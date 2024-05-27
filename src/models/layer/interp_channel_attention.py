"""
Unofficial implementation of Interpolation Channel Attention Layer
Z.Sun et al. "Interpolation Variable Rate Image Compression", ACMMM2022
"""

from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor


class InterpChAtt(nn.Module):
    def __init__(self,
                 ch: int,
                 rate_level: int,
                 actv: str='identity',
                 use_interp: bool=False,
                 use_bias: bool=False) -> None:
        super().__init__()
        weight_vector = torch.ones(rate_level, 1, ch, 1, 1, dtype=torch.float32)
        if actv == 'softplus': # initialize so that the output becomes 1
            weight_vector *= np.log(np.e - 1)
        self.weight = nn.Parameter(weight_vector, requires_grad=True)

        if use_bias:
            bias_vector = torch.zeros(rate_level, 1, ch, 1, 1, dtype=torch.float32)
            self.bias = nn.Parameter(bias_vector, requires_grad=True)

        actv_dict = dict(relu=nn.ReLU(), softplus=nn.Softplus(), identity=nn.Identity())
        self.actv = actv_dict[actv]
        self.use_interp = use_interp
        self.use_bias = use_bias
        self.rate_level = rate_level

    def get_interp_val(self, tensor: Tensor, ind: Tensor) -> Tensor:
        """
        ```
            1 - alpha    alpha
        N -------------|------- N+1
        l             ind       r
        ```
        """
        l = torch.floor(ind.float())
        r = l + 1.
        r = torch.minimum(r, torch.tensor(self.rate_level - 1, device=r.device))
        alpha = (r - ind).reshape(-1, *[1] * (tensor.ndim - 1)) # [N, 1, 1, 1, 1]
        out = (tensor[l.long()] * alpha) + (tensor[r.long()] * (1 - alpha))
        return out.squeeze(1)

    def forward(self, x: Tensor, rate_ind: Union[float, Tensor]):
        # assert 0 <= rate_ind < self.rate_level, f'rate_ind = {rate_ind} should be in [0, self.rate_level)'
        if not isinstance(rate_ind, Tensor):
            rate_ind = torch.tensor([rate_ind], dtype=torch.float)
        rate_ind = rate_ind.to(x.device)

        # check shape
        assert rate_ind.ndim == 1, f'rate_ind should be 1D tensor, but {rate_ind.ndim}D tensor is given'
        if rate_ind.shape[0] > 1:
            assert rate_ind.shape[0] == x.shape[0]
        # check value
        assert rate_ind.min() >= 0
        assert rate_ind.max() <= self.rate_level - 1

        weight = self.get_interp_val(self.weight, rate_ind) if self.use_interp else self.weight[rate_ind.long()]
        x = self.actv(weight) * x
        if self.use_bias:
            bias = self.get_interp_val(self.bias, rate_ind) if self.use_interp else self.bias[rate_ind.long()]
            x = x + bias
        return x