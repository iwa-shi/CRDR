from typing import Dict, Optional, Union, List, Tuple

import os

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from compressai.models import get_scale_table
from tqdm import tqdm

from .base_model import BaseModel
from src.models.subnet import build_subnet
from src.utils.registry import MODEL_REGISTRY
from src.utils.img_utils import calc_psnr, calc_ms_ssim, imwrite
from src.utils.codec_utils import HeaderHandler


@MODEL_REGISTRY.register()
class HyperpriorModel(BaseModel):
    def _build_subnets(self):
        self.encoder = build_subnet(self.opt.subnet.encoder, subnet_type="encoder")
        self.decoder = build_subnet(self.opt.subnet.decoder, subnet_type="decoder")
        self.hyperencoder = build_subnet(
            self.opt.subnet.hyperencoder, subnet_type="hyperencoder"
        )
        self.hyperdecoder = build_subnet(
            self.opt.subnet.hyperdecoder, subnet_type="hyperdecoder"
        )
        self.entropy_model_z = build_subnet(
            self.opt.subnet.entropy_model_z, subnet_type="entropy_model"
        )
        self.entropy_model_y = build_subnet(
            self.opt.subnet.entropy_model_y, subnet_type="entropy_model"
        )

    def run_model(
        self, real_images, is_train: bool = True, 
    ):
        N, _, H, W = real_images.size()
        num_pixel = H * W
        real_images = self.data_preprocess(real_images, is_train=is_train)
        out_dict = self.forward(real_images, is_train=is_train)
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
            **rate_summary_dict,
            **out_dict.get("others", {}),
        )

    def get_rate_summary_dict(
        self, out_dict: Dict, num_pixel: int
    ) -> Dict[str, Tensor]:
        _, y_noisy_bpp = self.likelihood_to_bit(out_dict["likelihoods"]["y"], num_pixel)
        _, z_noisy_bpp = self.likelihood_to_bit(out_dict["likelihoods"]["z"], num_pixel)
        _, y_quantized_bpp = self.likelihood_to_bit(
            out_dict["q_likelihoods"]["y"], num_pixel
        )
        _, z_quantized_bpp = self.likelihood_to_bit(
            out_dict["q_likelihoods"]["z"], num_pixel
        )
        return dict(
            y_likelihood=out_dict["likelihoods"]["y"],
            z_likelihood=out_dict["likelihoods"]["z"],
            bpp=y_noisy_bpp + z_noisy_bpp,
            y_q_likelihood=out_dict["q_likelihoods"]["y"],
            z_q_likelihood=out_dict["q_likelihoods"]["z"],
            qbpp=y_quantized_bpp + z_quantized_bpp,
        )

    def likelihood_to_bit(
        self, likelihood: Tensor, num_pixel: int
    ) -> Tuple[Tensor, Tensor]:
        # bitcost = -(torch.log(likelihood).sum()) / np.log(2)
        sum_dims = tuple(range(1, likelihood.ndim))
        bitcost = -(torch.log(likelihood).sum(dim=sum_dims)) / np.log(2)
        return bitcost, bitcost / num_pixel

    def forward(self, real_images, is_train: bool = True):
        y = self.encoder(real_images)
        z = self.hyperencoder(y)
        z_hat, z_likelihood = self.entropy_model_z(z, is_train=is_train)
        hyper_out = self.hyperdecoder(z_hat)
        y_hat, y_likelihood = self.entropy_model_y(y, hyper_out, is_train=is_train)
        fake_images = self.decoder(y_hat)
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
        self.header_handler = HeaderHandler(use_non_zero_ind=False)
        self.entropy_model_z.update(force=True)
        scale_table = get_scale_table()
        self.entropy_model_y.update_scale_table(scale_table, force=True)

        # use cpu in models involving entropy coding
        self.entropy_model_z.to("cpu")
        self.entropy_model_y.to("cpu")
        self.hyperdecoder.to("cpu")

        self.yC = self.encoder.latent_ch
        self.zC = self.hyperencoder.latent_ch
        enc_stride = 2**self.encoder.num_downscale
        hyper_stride = 2**self.hyperencoder.num_downscale
        self.y_stride = enc_stride
        self.model_stride = enc_stride * hyper_stride

    @torch.no_grad()
    def compress(self, real_images):
        N, _, H, W = real_images.shape
        assert N == 1, f"In compress mode, batchsize must be 1, but {N}"

        real_images = self.data_preprocess(real_images, is_train=False)
        y = self.encoder(real_images)
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

        header_str = self.header_handler.encode((H, W), y_hat)

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

        fake_img = self.decoder(y_hat.to(self.device))
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

        for idx, data in enumerate(dataloader):
            with torch.no_grad():
                out_dict = self.run_model(**data, is_train=False)
            score_list.append(
                {
                    "idx": idx + 1,
                    "bpp": out_dict["bpp"].item(),
                    "psnr": calc_psnr(
                        out_dict["real_images"], out_dict["fake_images"], 255
                    ),
                    "ms_ssim": calc_ms_ssim(
                        out_dict["real_images"], out_dict["fake_images"]
                    ),
                }
            )

            if save_img:
                fake_path = os.path.join(save_dir, f"sample_{idx+1}_fake.jpg")
                imwrite(fake_path, out_dict["fake_images"])
                real_path = os.path.join(save_dir, f"sample_{idx+1}_real.jpg")
                imwrite(real_path, out_dict["real_images"])

            if use_tqdm:
                pbar.update(1)

            if idx + 1 == sample_size:
                break
        if use_tqdm:
            pbar.close()

        return pd.json_normalize(score_list)
