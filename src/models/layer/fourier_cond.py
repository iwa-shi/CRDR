"""
My unofficial implementation 
FourierCond from "Multi-Realism Image Compression with a Conditional Generator (CVPR2023)"
"""
import torch
import torch.nn as nn
from torch import Tensor

import numpy as np
from typing import Union

class FourierEmbedding(object):
    def __init__(self, L: int, max_beta: float, use_pi: bool=True, include_x: bool=False) -> None:
        self.L = L
        self.max_beta = max_beta
        self.freq = torch.pow(torch.Tensor([2]), torch.arange(L)) # [2^0, 2^1, 2^2, ..., 2^(L-1)]
        if use_pi:
            self.freq = self.freq * np.pi
        self.include_x = include_x

    def embed(self, beta: Union[int, float, Tensor]) -> Tensor:
        if isinstance(beta, (int, float)):
            beta = torch.Tensor([beta]).float()
        assert isinstance(beta, Tensor)
        assert beta.ndim == 1
        assert 0 <= beta.min() <= self.max_beta
        assert 0 <= beta.max() <= self.max_beta

        norm_beta = (beta / self.max_beta) # [0, 1]
        norm_beta = (norm_beta - 0.5) * 2 # [-1, 1]

        sin_tensor = torch.sin(norm_beta * self.freq)
        cos_tensor = torch.cos(norm_beta * self.freq)
        out = torch.cat([sin_tensor, cos_tensor], dim=0) # [2L]
        if self.include_x:
            out = torch.cat([norm_beta, out], dim=0) # [2L+1]
        return out.unsqueeze(0).detach() # [1, 2L] or [1, 2L+1]


if __name__ == '__main__':
    emb = FourierEmbedding(L=4, max_beta=5.12)
    print(emb.freq)
    print(emb.freq.shape)

    print(emb.embed(2.56))
    print(emb.embed(1.28))
    print(emb.embed(0.))
    print(emb.embed(0.).shape)