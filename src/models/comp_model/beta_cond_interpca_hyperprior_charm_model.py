from typing import List, Dict, Optional, Union, Any, Tuple
import numpy as np
import torch
from torch import Tensor

from src.models.subnet import build_subnet
from src.utils.registry import MODEL_REGISTRY


from .beta_cond_interpca_hyperprior_model import BetaCondInterpCaHyperpriorModel


@MODEL_REGISTRY.register()
class BetaCondInterpCaHyperpriorCharmModel(BetaCondInterpCaHyperpriorModel):
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
        self.context_model = build_subnet(
            self.opt.subnet.context_model, subnet_type="context_model"
        )

    def forward(
        self,
        real_images: Tensor,
        rate_ind: Union[float, Tensor],
        beta: float,
        is_train: bool = True,
    ) -> dict:
        y = self.encoder(real_images, rate_ind)
        z = self.hyperencoder(y)
        z_hat, z_likelihood = self.entropy_model_z(z, is_train=is_train)
        hyper_out = self.hyperdecoder(z_hat)

        y_hat, y_likelihood, y_q_likelihood = self.context_model(
            y,
            hyper_out,
            self.entropy_model_y,
            is_train=is_train,
            calc_q_likelihood=True,
        )

        fake_images = self.decoder(y_hat, rate_ind, beta=beta)
        if not is_train:
            fake_images = torch.clamp(fake_images, min=-1.0, max=1.0)
        with torch.no_grad():
            _, z_q_likelihood = self.entropy_model_z(z, is_train=False)

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
        self.context_model.to("cpu")

    @torch.no_grad()
    def compress(self, real_images: Tensor, rate_ind: Union[int, float]) -> Dict:
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
        y_str, y_hat, y_likelihood = self.context_model.forward_compress(
            y, hyper_out, self.entropy_model_y
        )

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
    def decompress(
        self, string_list: List, beta: float = 0.0
    ) -> Tuple[Tensor, Tensor, Tensor]:
        assert (
            len(string_list) == 3
        ), f"String list length should be 3 (header, z, and y),\
                                             but got {len(string_list)}"
        header_str = string_list[0]
        latent_z_str = string_list[1]
        latent_y_str = string_list[2]

        header_dict = self.header_handler.decode(header_str)
        H, W = header_dict["img_size"]
        rate_ind = header_dict["rate_ind"]
        padH = int(np.ceil(H / self.model_stride)) * self.model_stride
        padW = int(np.ceil(W / self.model_stride)) * self.model_stride
        zH, zW = padH // self.model_stride, padW // self.model_stride

        z_symbol = self.entropy_model_z.decompress([latent_z_str], (zH, zW))
        z_hat = self.entropy_model_z.dequantize(z_symbol)
        hyper_out = self.hyperdecoder(z_hat)

        y_hat, y_symbol = self.context_model.forward_decompress(
            latent_y_str, hyper_out, self.entropy_model_y
        )

        fake_img = self.decoder(y_hat.to(self.device), rate_ind, beta=beta)
        fake_img = self.data_postprocess(fake_img, size=(H, W), is_train=False)
        return fake_img, z_hat, y_hat
