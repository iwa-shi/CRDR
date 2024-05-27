from tracemalloc import is_tracing
from typing import Dict, List

import torch
from torch.utils.data import Dataset

from PIL import Image

from .data_transform import *


class BaseImageDataset(Dataset):
    def __init__(self,
                 img_path_list: List[str],
                 is_train: bool=True,
                 image_size: int=256,
                 resize_range:Optional[Tuple[float, float]]=None,
                 interpolation: str='bicubic') -> None:
        super().__init__()
        self.is_train = is_train
        self.img_path_list = img_path_list
        if is_train:
            self.transform = PilTrainTransform((image_size, image_size), resize_range=resize_range, interpolation=interpolation) 
        else:
            self.transform = PilEvalTransform()

    def __len__(self) -> int:
        return len(self.img_path_list)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        img_path = self.img_path_list[index]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        return {'real_images':img}
