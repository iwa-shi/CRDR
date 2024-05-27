from typing import List, Dict, Optional, Union, Any, Tuple
import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from torch import Tensor

from src.models.subnet import build_subnet
from src.utils.registry import MODEL_REGISTRY
from src.utils.img_utils import calc_psnr, calc_ms_ssim, imwrite


from .interpca_hyperprior_model import InterpCaHyperpriorModel


@MODEL_REGISTRY.register()
class BetaCondInterpCaHyperpriorModel(InterpCaHyperpriorModel):
    def __init__(self, opt):
        super().__init__(opt)
        self.max_beta: float = opt.subnet.decoder.max_beta

    def sample_beta(self) -> float:
        i = np.random.randint(0, 101)
        beta = self.max_beta * (float(i) / 100.0)
        return beta

    def run_model(
        self,
        real_images: Tensor,
        rate_ind: Optional[Union[float, Tensor]] = None,
        beta: Optional[float] = None,
        is_train: bool = True,
    ) -> dict:
        N, _, H, W = real_images.size()
        num_pixel = H * W
        real_images = self.data_preprocess(real_images, is_train=is_train)
        if rate_ind is None:
            if not is_train:
                raise ValueError('"rate_ind" must be specified if is_train=False')
            rate_ind = self.sample_rate_ind()

        if beta is None:
            if not is_train:
                raise ValueError('"beta" must be specified if is_train=False')
            beta = self.sample_beta()

        out_dict = self.forward(real_images, rate_ind, beta, is_train=is_train)

        fake_images = out_dict["fake_images"]
        rate_summary_dict = self.get_rate_summary_dict(out_dict, num_pixel)
        real_images, fake_images = self.data_postprocess(
            real_images, fake_images, size=(H, W), is_train=is_train
        )
        return dict(
            real_images=real_images,
            fake_images=fake_images,
            y_hat=out_dict["quantized_code"]["y"],
            z_hat=out_dict["quantized_code"]["z"],
            rate_ind=rate_ind,
            beta=beta,
            **rate_summary_dict,
            **out_dict.get("others", {}),
        )

    def forward(
        self,
        real_images: Tensor,
        rate_ind: Union[Tensor, float],
        beta: float,
        is_train: bool = True,
    ) -> dict:
        y = self.encoder(real_images, rate_ind)
        z = self.hyperencoder(y)
        z_hat, z_likelihood = self.entropy_model_z(z, is_train=is_train)
        hyper_out = self.hyperdecoder(z_hat)

        y_hat, y_likelihood = self.entropy_model_y(y, hyper_out, is_train=is_train)
        fake_images = self.decoder(y_hat, rate_ind, beta=beta)
        if not is_train:
            fake_images = torch.clamp(fake_images, min=-1.0, max=1.0)
        with torch.no_grad():
            _, z_q_likelihood = self.entropy_model_z(z, is_train=False)
            _, y_q_likelihood = self.entropy_model_y(y, hyper_out, is_train=False)

        return {
            "fake_images": fake_images,
            "likelihoods": {
                "y": y_likelihood,
                "z": z_likelihood,
            },
            "latent_code": {
                "y": y,
                "z": z,
            },
            "quantized_code": {
                "y": y_hat,
                "z": z_hat,
            },
            "q_likelihoods": {
                "y": y_q_likelihood,
                "z": z_q_likelihood,
            },
        }

    @torch.no_grad()
    def decompress(
        self, string_list: List, beta: float = 0.0
    ) -> Tuple[Tensor, Tensor, Tensor]:
        header_str = string_list[0]
        latent_z_str = string_list[1]
        latent_y_str = string_list[2]

        header_dict = self.header_handler.decode(header_str)

        H, W = header_dict["img_size"]
        rate_ind = header_dict["rate_ind"]

        padH = int(np.ceil(H / self.model_stride)) * self.model_stride
        padW = int(np.ceil(W / self.model_stride)) * self.model_stride
        zH, zW = padH // self.model_stride, padW // self.model_stride

        z_hat = self.entropy_model_z.decompress([latent_z_str], (zH, zW))

        hyper_out = self.hyperdecoder(z_hat)

        means_hat, scales_hat = hyper_out.chunk(2, 1)
        indexes = self.entropy_model_y.build_indexes(scales_hat)
        y_hat = self.entropy_model_y.decompress(
            [latent_y_str], indexes, means=means_hat
        )

        fake_img = self.decoder(y_hat.to(self.device), rate_ind, beta=beta)
        fake_img = self.data_postprocess(fake_img, size=(H, W), is_train=False)
        return fake_img, z_hat, y_hat

    @torch.no_grad()
    def validation(
        self,
        dataloader,
        max_sample_size: int,
        beta: Optional[float] = None,
        save_img: bool = False,
        save_dir: str = "",
        use_tqdm: bool = False,
    ) -> pd.DataFrame:
        """

        Args:
            dataloader ([type]): [description]
            max_sample_size (int): [description]
            save_img (bool, optional): [description]. Defaults to False.
            save_dir (str, optional): [description]. Defaults to ''.
            use_tqdm (bool, optional): [description]. Defaults to False.

        Returns:
            pd.DataFrame: [description]
        """
        score_list = []

        sample_size = min(len(dataloader), max_sample_size)

        if use_tqdm:
            pbar = tqdm(total=sample_size, ncols=60)

        if save_img:
            assert os.path.exists(save_dir), f'save_dir: "{save_dir}" does not exist.'

        beta = self.max_beta / 2.0 if beta is None else beta

        for idx, data in enumerate(dataloader):
            score_dict: dict[str, Any] = {"idx": idx + 1}

            for rate_ind in range(self.rate_level):
                out_dict = self.run_model(
                    **data,
                    rate_ind=rate_ind,
                    beta=beta,
                    is_train=False,
                )
                score_dict.update(
                    {
                        f"bpp_{rate_ind+1}": out_dict["bpp"].item(),
                        f"psnr_{rate_ind+1}": calc_psnr(
                            out_dict["real_images"], out_dict["fake_images"], 255
                        ),
                        f"ms_ssim_{rate_ind+1}": calc_ms_ssim(
                            out_dict["real_images"], out_dict["fake_images"]
                        ),
                    }
                )

                if save_img:
                    fake_path = os.path.join(
                        save_dir, f"sample_{idx+1}_fake_lv{rate_ind+1}.jpg"
                    )
                    imwrite(fake_path, out_dict["fake_images"])
                    if rate_ind == 0:
                        real_path = os.path.join(save_dir, f"sample_{idx+1}_real.jpg")
                        imwrite(real_path, out_dict["real_images"])

            score_list.append(score_dict.copy())

            if use_tqdm:
                pbar.update(1)
            if idx + 1 == sample_size:
                break
        if use_tqdm:
            pbar.close()

        return pd.json_normalize(score_list)
