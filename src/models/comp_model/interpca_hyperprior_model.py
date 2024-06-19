from typing import Dict, Optional, Union, List, Tuple, Any


import os

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from tqdm import tqdm

from .hyperprior_model import HyperpriorModel
from src.utils.registry import MODEL_REGISTRY
from src.utils.img_utils import calc_psnr, calc_ms_ssim, imwrite
from src.utils.codec_utils import MultiRateHeaderHandler


@MODEL_REGISTRY.register()
class InterpCaHyperpriorModel(HyperpriorModel):
    def __init__(self, opt):
        self.rate_level = opt.subnet.encoder.rate_level
        assert opt.subnet.encoder.rate_level == opt.subnet.decoder.rate_level
        super().__init__(opt)
        self.batch_rate_ind_sample = opt.get("batch_rate_ind_sample", False)
        if self.batch_rate_ind_sample:
            raise NotImplementedError("batch_rate_ind_sample is not supported yet.")

    def sample_rate_ind(self, num_sample=1) -> Tensor:
        return torch.randint(self.rate_level, (num_sample,))

    def run_model(
        self,
        real_images,
        rate_ind: Optional[Union[Tensor, float]] = None,
        is_train: bool = True,
    ):
        N, _, H, W = real_images.size()
        num_pixel = H * W
        real_images = self.data_preprocess(real_images, is_train=is_train)
        if rate_ind is None:
            if not (is_train):
                raise ValueError('"rate_ind" must be specified if is_train=False')
            rate_ind = self.sample_rate_ind(
                num_sample=N if self.batch_rate_ind_sample else 1
            )

        out_dict = self.forward(real_images, rate_ind, is_train=is_train)

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
            **rate_summary_dict,
            **out_dict.get("others", {}),
        )

    def forward(
        self, real_images, rate_ind: Union[float, Tensor], is_train: bool = True
    ):
        y = self.encoder(real_images, rate_ind)
        z = self.hyperencoder(y)
        z_hat, z_likelihood = self.entropy_model_z(z, is_train=is_train)
        hyper_out = self.hyperdecoder(z_hat)
        y_hat, y_likelihood = self.entropy_model_y(y, hyper_out, is_train=is_train)

        fake_images = self.decoder(y_hat, rate_ind)
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

    def codec_setup(self):
        super().codec_setup()
        self.header_handler = MultiRateHeaderHandler(use_non_zero_ind=False)

    @torch.no_grad()
    def compress(self, real_images: Tensor, rate_ind: Union[Tensor, float]) -> Dict:
        N, _, H, W = real_images.shape
        assert N == 1, f"In compress mode, batchsize must be 1, but {N}"

        real_images = self.data_preprocess(real_images, is_train=False)
        y = self.encoder(real_images, rate_ind)
        z = self.hyperencoder(y)
        y = y.cpu()
        z = z.cpu()

        z_hat, z_likelihood = self.entropy_model_z(z, is_train=False)
        z_str = self.entropy_model_z.compress(z)

        hyper_out = self.hyperdecoder(z_hat)
        means_hat, scales_hat = hyper_out.chunk(2, 1)
        indexes = self.entropy_model_y.build_indexes(scales_hat)
        y_str = self.entropy_model_y.compress(y, indexes, means=means_hat)

        hyper_out = self.hyperdecoder(z_hat)
        y_hat, y_likelihood = self.entropy_model_y(y, hyper_out, is_train=False)

        header_str = self.header_handler.encode((H, W), y_hat, rate_ind=rate_ind)

        pred_y_bitcost, pred_y_bpp = self.likelihood_to_bit(y_likelihood, H * W)
        pred_z_bitcost, pred_z_bpp = self.likelihood_to_bit(z_likelihood, H * W)

        return {
            "string_list": [header_str, z_str[0], y_str[0]],
            "z_hat": z_hat,
            "y_hat": y_hat,
            "z_likelihood": z_likelihood,
            "y_likelihood": y_likelihood,
            "pred_y_bit": pred_y_bitcost.item(),
            "pred_y_bpp": pred_y_bpp.item(),
            "pred_z_bit": pred_z_bitcost.item(),
            "pred_z_bpp": pred_z_bpp.item(),
        }

    @torch.no_grad()
    def decompress(self, string_list: List) -> Tuple[Tensor, Tensor, Tensor]:
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

        fake_img = self.decoder(y_hat.to(self.device), rate_ind)
        fake_img = self.data_postprocess(fake_img, size=(H, W), is_train=False)
        return fake_img, z_hat, y_hat

    def validation(
        self,
        dataloader,
        max_sample_size: int,
        save_img: bool = False,
        save_dir: str = "",
        use_tqdm: bool = False,
    ) -> pd.DataFrame:
        score_list = []

        sample_size = min(len(dataloader), max_sample_size)

        if use_tqdm:
            pbar = tqdm(total=sample_size, ncols=60)

        if save_img:
            assert os.path.exists(save_dir), f'save_dir: "{save_dir}" does not exist.'

        for idx, data in enumerate(dataloader):
            score_dict: Dict[str, Any] = {"idx": idx + 1}

            for rate_ind in range(self.rate_level):
                with torch.no_grad():
                    out_dict = self.run_model(
                        **data, rate_ind=float(rate_ind), is_train=False
                    )
                    psnr = calc_psnr(out_dict["real_images"], out_dict["fake_images"], 255)
                    ms_ssim = calc_ms_ssim(out_dict["real_images"], out_dict["fake_images"])

                score_dict.update({
                    f"bpp_{rate_ind+1}": out_dict["bpp"].mean().item(),
                    f"psnr_{rate_ind+1}": psnr,
                    f"ms_ssim_{rate_ind+1}": ms_ssim,
                })

                if save_img:
                    fake_path = os.path.join(
                        save_dir, f"sample_{idx+1}_fake_q{rate_ind}.jpg"
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
