"""
"Channel-wise autoregressive entropy models for learned image compression", ICIP2020
from https://github.com/tensorflow/compression/blob/master/models/ms2020.py
"""
from typing import Tuple, Optional, List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from compressai.ans import RansDecoder
from compressai.entropy_models import GaussianConditional

from src.utils.registry import CONTEXTMODEL_REGISTRY
from .base_context_model import BaseContextModel

def get_actv(actv: str):
    actv_dict = dict(
        relu  = nn.ReLU(inplace=True),
        gelu  = nn.GELU(),
        lrelu = nn.LeakyReLU(0.2),
    )
    return actv_dict[actv]

class SliceTransform(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, actv: str='relu'):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_ch, 224, kernel_size=5, padding=2, stride=1),
            get_actv(actv),
            nn.Conv2d(224, 128, kernel_size=5, padding=2, stride=1),
            get_actv(actv),
            nn.Conv2d(128, out_ch, kernel_size=3, padding=1, stride=1),
        )

    def forward(self, x):
        return self.model(x)


@CONTEXTMODEL_REGISTRY.register()
class Minnen20CharmContextModel(BaseContextModel):
    def __init__(self,
                 num_slices: int,
                 bottleneck_y: int,
                 hyper_out_ch: int, 
                 max_support_slices: int=5,
                 slice_transform_kwargs: Dict={},
                 crop_gaussian_params: bool=False) -> None:
        """Minnen et al. ICIP2020. Channel Autoregressive Context Model\\
        
        Args:
            num_slices (int): 
            bottleneck_y (int): 
            hyper_out_ch (int): hyper_out_mean + hyper_out_scale
            max_support_slices (int, optional):Defaults to 5.
        """
        super().__init__()
        assert bottleneck_y % num_slices == 0, f'bottleneck_y % num_slices must be 0, but got {bottleneck_y} and {num_slices}'
        assert (max_support_slices == -1) or (1 <= max_support_slices <= num_slices), \
                f'Invalid max_support_slices. It should be -1 or [1, num_slices({num_slices})]'

        slice_ch = bottleneck_y // num_slices
        hyper_mean_ch = hyper_scale_ch = hyper_out_ch // 2

        self.slice_ch = slice_ch
        self.num_slices = num_slices
        self.max_support_slices = max_support_slices
        self.crop_gaussian_params = crop_gaussian_params

        self.mean_slice_transforms = nn.ModuleList()
        self.scale_slice_transforms = nn.ModuleList()
        self.lrp_slice_transforms = nn.ModuleList()

        for slice_ind in range(num_slices):
            if max_support_slices == -1:
                num_slice = slice_ind
            else:
                num_slice = min(slice_ind, max_support_slices) 
            support_slices_ch = slice_ch * num_slice
            self.mean_slice_transforms.append(
                SliceTransform(in_ch=support_slices_ch+hyper_mean_ch, out_ch=slice_ch, **slice_transform_kwargs))
            self.scale_slice_transforms.append(
                SliceTransform(in_ch=support_slices_ch+hyper_scale_ch, out_ch=slice_ch, **slice_transform_kwargs))
            self.lrp_slice_transforms.append(
                SliceTransform(in_ch=support_slices_ch+hyper_mean_ch+slice_ch, out_ch=slice_ch, **slice_transform_kwargs))

    def forward(self, 
                y: Tensor,
                hyper_out: Tensor,
                entropy_model_y: GaussianConditional,
                is_train: bool,
                calc_q_likelihood: bool=True):
        # y: tensor(N, C, H, W) -> y_slices: list of tensor(N, C/num_slices, H, W)
        y_shape = y.shape[2:]
        y_slices = list(torch.chunk(y, chunks=self.num_slices, dim=1))
        hyper_mean, hyper_scale = torch.chunk(hyper_out, chunks=2, dim=1)

        y_hat_slice_list = []
        y_likelihood_list = []
        y_q_likelihood_list = []

        for slice_ind, y_slice in enumerate(y_slices):
            support_slices = (y_hat_slice_list if self.max_support_slices < 0 else 
                            y_hat_slice_list[:self.max_support_slices])

            # Estimate mean & scale from slices and hyper_out
            mean_support = torch.cat([hyper_mean] + support_slices, dim=1)
            scale_support = torch.cat([hyper_scale] + support_slices, dim=1)

            mu = self.mean_slice_transforms[slice_ind](mean_support)
            sigma = self.scale_slice_transforms[slice_ind](scale_support)
            if self.crop_gaussian_params:
                mu = mu[:, :, :y_shape[0], :y_shape[1]]
                sigma = sigma[:, :, :y_shape[0], :y_shape[1]]

            # Calculate likelihoods
            y_hat_slice, y_likelihood_slice = entropy_model_y(y_slice, torch.cat([mu, sigma], dim=1), is_train=is_train)
            y_likelihood_list.append(y_likelihood_slice)

            if calc_q_likelihood:
                with torch.no_grad():
                    _, y_q_likelihood_slice = entropy_model_y(y_slice, torch.cat([mu, sigma], dim=1), is_train=False)
                y_q_likelihood_list.append(y_q_likelihood_slice)

            # Latent Residual Predictor
            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_slice_transforms[slice_ind](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)

            y_hat_slice = y_hat_slice + lrp # add estimated residual
            y_hat_slice_list.append(y_hat_slice)

        y_hat = torch.cat(y_hat_slice_list, dim=1)
        y_likelihood = torch.cat(y_likelihood_list, dim=1)

        if calc_q_likelihood:
            y_q_likelihood = torch.cat(y_q_likelihood_list, dim=1)
            return y_hat, y_likelihood, y_q_likelihood

        return y_hat, y_likelihood

    def forward_compress(self, 
                y: Tensor,
                hyper_out: Tensor,
                entropy_model_y: GaussianConditional) -> Tuple[List[bytes], Tensor, Tensor]:
        # y: tensor(N, C, H, W) -> y_slices: list of tensor(N, C/num_slices, H, W)
        y_slices = list(torch.chunk(y, chunks=self.num_slices, dim=1))
        hyper_mean, hyper_scale = torch.chunk(hyper_out, chunks=2, dim=1)

        y_hat_slice_list = []
        y_likelihood_list = []
        scale_slice_list = []
        mean_slice_list = []

        for slice_ind, y_slice in enumerate(y_slices):
            support_slices = (y_hat_slice_list if self.max_support_slices < 0 else 
                            y_hat_slice_list[:self.max_support_slices])

            # Estimate mean & scale from slices and hyper_out
            mean_support = torch.cat([hyper_mean] + support_slices, dim=1)
            scale_support = torch.cat([hyper_scale] + support_slices, dim=1)

            mu = self.mean_slice_transforms[slice_ind](mean_support)
            sigma = self.scale_slice_transforms[slice_ind](scale_support)

            mean_slice_list.append(mu)
            scale_slice_list.append(sigma)
            # Calculate likelihoods
            y_hat_slice, y_likelihood_slice = entropy_model_y(y_slice, torch.cat([mu, sigma], dim=1), is_train=False)

            # Latent Residual Predictor
            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_slice_transforms[slice_ind](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice = y_hat_slice + lrp # add estimated residual

            y_hat_slice_list.append(y_hat_slice)
            y_likelihood_list.append(y_likelihood_slice)

        y_hat = torch.cat(y_hat_slice_list, dim=1)
        y_likelihood = torch.cat(y_likelihood_list, dim=1)
        y_mean = torch.cat(mean_slice_list, dim=1)
        y_scale = torch.cat(scale_slice_list, dim=1)

        indexes = entropy_model_y.build_indexes(y_scale)
        y_str = entropy_model_y.compress(y, indexes=indexes, means=y_mean)

        return y_str, y_hat, y_likelihood


    def forward_decompress(self, 
                y_str: bytes,
                hyper_out: torch.Tensor,
                entropy_model_y: GaussianConditional) -> Tuple[Tensor, Tensor]:

        cdf = entropy_model_y._quantized_cdf.tolist()
        cdf_lengths = entropy_model_y._cdf_length.tolist()
        offsets = entropy_model_y._offset.tolist()

        rans_decoder = RansDecoder()
        rans_decoder.set_stream(y_str)

        hyper_mean, hyper_scale = torch.chunk(hyper_out, chunks=2, dim=1)

        y_symbol_slice_list = []
        y_hat_slice_list = []

        for slice_ind in range(self.num_slices):
            support_slices = (y_hat_slice_list if self.max_support_slices < 0 else 
                            y_hat_slice_list[:self.max_support_slices])

            # Estimate mean & scale from slices and hyper_out
            mean_support = torch.cat([hyper_mean] + support_slices, dim=1)
            scale_support = torch.cat([hyper_scale] + support_slices, dim=1)

            mu = self.mean_slice_transforms[slice_ind](mean_support)
            sigma = self.scale_slice_transforms[slice_ind](scale_support)

            # decode values from stream
            indexes = entropy_model_y.build_indexes(sigma)
            read_val = rans_decoder.decode_stream(
                indexes.reshape(-1).int().tolist(), cdf, cdf_lengths, offsets
            )
            y_symbol_slice = torch.Tensor(read_val).reshape(sigma.size())
            y_hat_slice = entropy_model_y.dequantize(y_symbol_slice, mu)

            # Latent Residual Predictor
            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_slice_transforms[slice_ind](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice = y_hat_slice + lrp # add estimated residual

            y_hat_slice_list.append(y_hat_slice)
            y_symbol_slice_list.append(y_symbol_slice)

        y_hat = torch.cat(y_hat_slice_list, dim=1)
        y_symbol = torch.cat(y_symbol_slice_list, dim=1).int()

        return y_hat, y_symbol
