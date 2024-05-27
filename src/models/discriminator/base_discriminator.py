import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseDiscriminator(nn.Module):
    def __init__(self) -> None:
        super().__init__()