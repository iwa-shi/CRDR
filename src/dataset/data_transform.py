from typing import Any, Optional, Tuple, Union, List
from PIL import Image

import numpy as np
import torch
import torchvision.transforms as T


class DataTransform(object):
    def __init__(self):
        self.transforms = []

    def __call__(self, img: np.ndarray) -> torch.Tensor:
        for trfm in self.transforms:
            img = trfm(img)
        return img


class PilTrainTransform(DataTransform):
    def __init__(self,
                 img_size: Union[Tuple, int],
                 resize_range: Optional[Tuple[float, float]]=None,
                 interpolation: str='bicubic') -> None:
        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        assert isinstance(img_size, tuple)
        assert len(img_size) == 2

        if resize_range is None:
            self.transforms = []
        else:
            _fmin, _fmax = resize_range
            self.transforms = [PilRandomResize(_fmin, _fmax, crop_size=img_size[0], interpolation=interpolation)]

        self.transforms += [
            T.RandomCrop(img_size, pad_if_needed=True, padding_mode='reflect'),
            T.RandomHorizontalFlip(0.5),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]

    def __call__(self, img: Image.Image) -> torch.Tensor:
        for trfm in self.transforms:
            img = trfm(img)
        return img



class PilEvalTransform(DataTransform):
    def __init__(self) -> None:
        self.transforms = [T.ToTensor(), T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]


class PilRandomResize(object):
    def __init__(self, f_min: float, f_max: float, crop_size: int=256, interpolation: str='bicubic'):
        super().__init__()
        self.scale_min = f_min
        self.scale_max = f_max
        self.crop_size = crop_size
        self.interpolation = dict(
                bicubic=Image.BICUBIC,
                bilinear=Image.BILINEAR
        )[interpolation]

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        shortest_side_length = min(h, w)
        minimum_scale_factor = float(self.crop_size) / float(shortest_side_length)
        scale_low = max(minimum_scale_factor, self.scale_min)
        scale_high = max(scale_low, self.scale_max)
        scale = np.random.uniform(scale_low, scale_high)
        img = img.resize((int(w*scale), int(h*scale)), self.interpolation)
        return img

