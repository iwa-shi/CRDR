import torch
from torch import Tensor

def ste_round(x: Tensor) -> Tensor:
    return (torch.round(x) - x).detach() + x