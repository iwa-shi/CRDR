from typing import Dict, Tuple, Set, Union, List

from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from compressai.models.utils import update_registered_buffers

from src.models.subnet.entropy_model.entropy_bottleneck import EntropyBottleneck
from compressai.entropy_models import GaussianConditional
from src.utils.logger import get_root_logger, log_dict_items

class BaseModel(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.device = opt.device

        # NOTE: In CompressAI, the input_image's range is [0, 1]. 
        # However, in our implementation, the input_image's range is always [-1, 1].
        # The option "convert_img_range_to_01" enables you to train models
        # with the same settings as CompressAI, and use the pre-trained CompressAI models.
        self.convert_img_range = opt.get('convert_img_range_to_01', False)
        log_dict_items({'convert_img_range_to_01': self.convert_img_range}, level='DEBUG', indent=True)

        self._build_subnets()
        self.stride = 64 # TODO: avoid hard coding

    def _build_subnets(self) -> None:
        raise NotImplementedError()

    def data_preprocess(self, real_images: Tensor, is_train: bool=True) -> Tensor:
        out = real_images
        if self.convert_img_range: # [-1, 1] -> [0, 1]
            out = (out + 1.) / 2.
        if not(is_train):
            out = self.pad_images(out)
            assert isinstance(out, torch.Tensor)
        out = out.to(self.device)
        return out
    
    def data_postprocess(self, *images:Tensor, size: Tuple[int, int], is_train: bool) -> Union[Tensor, Tuple[Tensor, ...]]:
        H, W = size
        out = []
        for img in images:
            if self.convert_img_range: # [0, 1] -> [-1, 1]
                img = (img - 0.5) * 2.
            if not(is_train):
                img = self._crop_image(img, H, W)
                img = img.clamp(-1, 1)
            out.append(img)
        if len(out) == 1:
            return out[0]
        return tuple(out)
    
    def run_model(self):
        raise NotImplementedError()

    def forward(self):
        raise NotImplementedError()

    def validation(self):
        raise NotImplementedError()

    def aux_loss(self):
        """Return the aggregated loss over the auxiliary entropy bottleneck
        module(s).

        Returns:
            torch.Tensor: auxiliary loss
        """
        aux_loss = sum(
            m.loss() for m in self.modules() if (isinstance(m, EntropyBottleneck))
        )
        return aux_loss

    def load_state_dict(self, state_dict, strict: bool=True):
        if hasattr(self, 'entropy_model_z') and isinstance(self.entropy_model_z, EntropyBottleneck):
            update_registered_buffers(
                self.entropy_model_z,
                "entropy_model_z",
                ["_quantized_cdf", "_offset", "_cdf_length"],
                state_dict,
            )
        if hasattr(self, 'entropy_model_y') and isinstance(self.entropy_model_y, GaussianConditional):
            update_registered_buffers(
                self.entropy_model_y,
                'entropy_model_y',
                ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
                state_dict,
            )

        return super().load_state_dict(state_dict, strict=strict)

    def load_learned_weight(self, ckpt_path: str) -> None:
        logger = get_root_logger()
        logger.info(f'load checkpoint: {ckpt_path}')
        ckpt = torch.load(ckpt_path, map_location=self.device)
        state_dict = ckpt['comp_model']
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k
            if 'module.' in name:
                name = k[7:]  # remove `module.`nvidia
            new_state_dict[name] = v
    
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in new_state_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        out = self.load_state_dict(model_dict)
        logger.debug(f'load_state_dict return: {out}')

        for m in self.children():
            if isinstance(m, EntropyBottleneck):
                m.update(force=False)

    def separate_aux_parameters(self) -> Tuple[Dict, Dict]:
        parameters = set(n for n, p in self.named_parameters() if not n.endswith(".quantiles") and p.requires_grad)
        aux_parameters = set(n for n, p in self.named_parameters() if n.endswith(".quantiles") and p.requires_grad)
        all_params = set(n for n, p in self.named_parameters() if p.requires_grad)
        
        ## Sanity check
        inter_params = parameters & aux_parameters
        union_params = parameters | aux_parameters
        assert len(inter_params) == 0
        assert len(union_params) - len(all_params) == 0

        params_dict = dict(self.named_parameters())
        parameters = {n: params_dict[n] for n in sorted(parameters)}
        aux_parameters = {n: params_dict[n] for n in sorted(aux_parameters)}

        return parameters, aux_parameters

    def pad_images(self, *images:Tensor) -> Union[Tensor, Tuple[Tensor, ...]]:
        out = []
        for img in images:
            out.append(self._pad_image(img, self.stride, mode='reflect'))
        if len(out) == 1:
            return out[0]
        return tuple(out)

    @staticmethod
    def _pad_image(x: Tensor, stride: int, mode: str='reflect') -> Tensor:
        _, _, H, W = x.size()
        padW = int(np.ceil(W / stride) * stride - W)
        padH = int(np.ceil(H / stride) * stride - H)
        if padH == 0 and padW == 0:
            return x
        return F.pad(x, (0, padW, 0, padH), mode=mode)

    def crop_images(self, *images:Tensor, size: Tuple[int, int]) -> Union[Tensor, Tuple[Tensor, ...]]:
        assert isinstance(size, tuple)
        assert len(size) == 2, f'size shoule be (H, W), but len(size)={len(size)}'
        H, W = size
        out = []
        for img in images:
            out.append(self._crop_image(img, H, W))
        if len(out) == 1:
            return out[0]
        return tuple(out)

    @staticmethod
    def _crop_image(x: Tensor, H: int, W: int) -> Tensor:
        return x[:, :, :H, :W]

    def codec_setup(self):
        raise NotImplementedError()