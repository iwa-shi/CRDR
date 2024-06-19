from typing import Dict, List

import os
from glob import glob

from .base_dataset import BaseImageDataset
from src.utils.registry import DATASET_REGISTRY


def get_openimage_path_list(root_dir: str, subset_list: List[int]):
    img_path_list = []
    for subset_id in subset_list:
        subset_dir = os.path.join(root_dir, f'train_{subset_id}')
        assert os.path.exists(subset_dir), f'openimage subset "train_{subset_id}" does not exist!'
        img_path_list.extend(glob(os.path.join(subset_dir, '*.jpg')))
    return img_path_list


@DATASET_REGISTRY.register()
class OpenImageImageDataset(BaseImageDataset):
    def __init__(self, root_dir: str, subset_list: List, is_train: bool=False, 
            image_size: int=256, **kwargs) -> None:
        """OpenImage Image dataset
        """
        if not(is_train):
            img_path_list = glob(os.path.join(root_dir, 'validation', '*.jpg'))
        else:
            img_path_list = get_openimage_path_list(root_dir, subset_list)
        img_path_list.sort()
        super().__init__(img_path_list, is_train, image_size, **kwargs)

