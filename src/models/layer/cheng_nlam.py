import torch
import torch.nn as nn
import torch.nn.functional as F

class ChengNLAM(nn.Module):
    """Cheng CVPR2020 Simple Attention Module
    """
    def __init__(self, ch: int, padding_mode: str='zeros'):
        super().__init__()
        self.trunk_block = nn.Sequential(
            NLAMResBlock(ch, ch, padding_mode=padding_mode),
            NLAMResBlock(ch, ch, padding_mode=padding_mode),
            NLAMResBlock(ch, ch, padding_mode=padding_mode),
        )
        self.attention_block = nn.Sequential(
            NLAMResBlock(ch, ch, padding_mode=padding_mode),
            NLAMResBlock(ch, ch, padding_mode=padding_mode),
            NLAMResBlock(ch, ch, padding_mode=padding_mode),
        )
        self.conv = nn.Conv2d(ch, ch, kernel_size=1, stride=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        trunk = self.trunk_block(x)
        attn = self.attention_block(x)
        attn = self.conv(attn)
        attn = self.sigmoid(attn)
        x = x + trunk * attn
        return x

class NLAMResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, padding_mode: str='zeros'):
        super().__init__()
        mid_ch = out_ch // 2
        self.c1 = nn.Conv2d(in_ch, mid_ch, kernel_size=1, stride=1, padding=0)
        self.c2 = nn.Conv2d(mid_ch, mid_ch, kernel_size=3, stride=1, padding=1, padding_mode=padding_mode)
        self.c3 = nn.Conv2d(mid_ch, out_ch, kernel_size=1, stride=1, padding=0)
        self.actv = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.c1(x)
        out = self.actv(out)
        out = self.c2(out)
        out = self.actv(out)
        out = self.c3(out)
        out = out + x
        return out