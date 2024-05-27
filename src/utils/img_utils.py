from __future__ import annotations
import math

from typing import TypeVar

import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pytorch_msssim import ssim, ms_ssim

from typing import Dict, Generator, List, Union, Tuple

from torch.types import Number

Array_or_Tensor = TypeVar("Array_or_Tensor", np.ndarray, torch.Tensor)

def torch2npimg(img_rgb:torch.Tensor, out_mode:str ='rgb') -> np.ndarray:
    """
    convert torch img to numpy img
    
    Args:
        img_rgb (torch.Tensor): image tensor *RGB* [-1.0, 1.0] or [0, 255]
        out_mode (str): 'rgb' or 'bgr' (default='rgb')
    
    Returns:
        img (np.array): image array [0, 255] uint8
    """
    assert isinstance(img_rgb, torch.Tensor)
    img = img_rgb.clone()

    if torch.max(img) <= 1.0:
        img = cvt_range_to_255(img)

    if img.dim() == 4: # if shape is [N, C, H, W]
        assert img.size(0) == 1, f'batch size must be 1, but {img.size(0)}'
        img = img.squeeze(0)

    img_np = img.detach().cpu().numpy() # torch.Tensor -> numpy
    img_np = img_np.transpose(1, 2, 0) # [CHW] -> [HWC]
    if out_mode == 'bgr':
        img_np = img_np[..., ::-1]
    return img_np.astype(np.uint8)


def npimg2torch(img: np.ndarray, out_dim: int=3, input_range: int=1) -> torch.Tensor:
    assert isinstance(img, np.ndarray), f'input must be np.ndarray. not {type(img)}'
    assert img.ndim == 3, f'input array must be [H, W, C]'
    assert img.shape[2] == 1 or img.shape[2] == 3, 'channel must be 1 or 3'
    assert out_dim in [3, 4]
    assert input_range in [1, 255]
    if input_range == 255:
        img = img.astype(np.float32)
        img = img / 255.
        img = (img - 0.5) * 2.
    img = img.transpose(2, 0, 1)
    img_torch = torch.from_numpy(img.copy())
    if out_dim == 4: # [C, H, W] -> [1, C, H, W]
        img_torch = img_torch.unsqueeze(0)
    return img_torch


def cvt_range_to_255(img: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """
    convert img [0.0, 1.0] to [0, 255]

    Args:
        img (np.array or torch.Tensor) [0, 1]
    Rerutns:
        img (np.array or torch.Tensor) [0, 255]
    """
    # assert img.min() >= 0, 'All elements must be greater than or equal to 0'
    # assert img.max() <= 1, 'All elements must be smaller than or equal to 1'
    # if img.min() < 0:
        # img = (img + 1.) / 2.
    # return img * 255.
    return (img + 1.) / 2. * 255.
    

def imwrite(path:str, img_rgb: Union[np.ndarray, torch.Tensor]):
    """
    save image

    Args:
        path (str): save path
        img_rgb (torch.Tensor or np.array): image (**RGB**) to save ([-1.0, 1.0], or [0, 255]) 
    """
    if isinstance(img_rgb, torch.Tensor):
        img = torch2npimg(img_rgb, out_mode='bgr')
    elif isinstance(img_rgb, np.ndarray):
        if np.max(img_rgb) <= 1.0:
            img_rgb = cvt_range_to_255(img_rgb)
        if img_rgb.ndim == 3:
            img = img_rgb[..., ::-1]
        else:
            img = img_rgb
    else:
        raise ValueError(f'img must be torch.Tensor or np.array, but it is {type(img_rgb)}')
    
    cv2.imwrite(path, img)


def calc_psnr(real: Union[np.ndarray, torch.Tensor], fake :Union[np.ndarray, torch.Tensor], data_range: int=255) -> float:
    """
    PSNR
    Args:
        real (np.array or torch.Tensor)
        fake (np.array or torch.Tensor)

    Return:
        psnr
    """
    # assert data_range in [1, 255]
    assert data_range == 255, 'data_range=1 was removed'

    if real.max() <= 1.0:
        real = cvt_range_to_255(real)
        fake = cvt_range_to_255(fake)

    if isinstance(real, torch.Tensor):
        real = real.detach().cpu().numpy()
        fake = fake.detach().cpu().numpy()
    
    assert isinstance(real, np.ndarray)
    assert isinstance(fake, np.ndarray)

    # if data_range == 255:
    real = real.astype(np.uint8).astype(np.float)
    fake = fake.astype(np.uint8).astype(np.float)
    mse = np.mean(np.power(real - fake, 2))

    # return 10.0 * np.log10((data_range ** 2) / mse)
    return 10.0 * math.log10((float(data_range) ** 2) / mse)


def calc_ms_ssim(real: Union[np.ndarray, torch.Tensor], fake: Union[np.ndarray, torch.Tensor]) -> float:
    """
    MS-SSIM
    Args:
        real (np.array or torch.Tensor)
        fake (np.array or torch.Tensor)

    Return:
        ms_ssim or -1
    """
    if real.max() <= 1.0:
        real = cvt_range_to_255(real)
        fake = cvt_range_to_255(fake)

    if isinstance(real, np.ndarray):
        real = npimg2torch(real, out_dim=4)
    if isinstance(fake, np.ndarray):
        fake = npimg2torch(fake, out_dim=4)

    real = real.cpu().detach().int().float()
    fake = fake.cpu().detach().int().float()
    try:
        val = ms_ssim(real, fake, data_range=255, size_average=True).item()
    except:
        import traceback
        traceback.print_exc()
        return -1
    return float(val)


def pad_image(x: torch.Tensor, stride: int, mode: str='reflect') -> torch.Tensor:
    """

    Args:
        x (torch.Tensor): input image
        stride (int):
        mode (str): padding mode. ['constant', 'reflect', 'replicate', 'circular']  default is 'reflect'

    Returns:
        x_ (ntorch.Tensor): height % stride = 0 and width % stride = 0
    """
    _, _, H, W = x.size()
    padW = int(np.ceil(W / stride) * stride - W)
    padH = int(np.ceil(H / stride) * stride - H)
    if padH == 0 and padW == 0:
        return x
    left, right = padW // 2, int(np.ceil(padW / 2))
    top, bottom = padH // 2, int(np.ceil(padH / 2))
    return F.pad(x, (left, right, top, bottom), mode=mode)
    # return F.pad(x, (left, right, top, bottom), mode='constant')


def crop_image(x: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """
    Crop compressed image

    Args:
        x (torch.Tensor): compressed image
        H (int): original height
        W (int): original width
    """
    _, _, H_, W_ = x.size()
    padW = W_ - W
    padH = H_ - H
    left = padW // 2
    top = padH // 2
    return x[:, :, top:top+H, left:left+W]

