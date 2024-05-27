from typing import Dict

import os
from glob import glob

import cv2
import numpy as np

# from .data_transform import *
from .base_dataset import BaseImageDataset
from src.utils.logger import get_root_logger, IndentedLog
from src.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class KodakImageDataset(BaseImageDataset):
    def __init__(self, root_dir: str, is_train: bool=False, 
            image_size: int=256) -> None:
        """Kodak Image dataset
        """
        assert not(is_train), f'Kodak dataset should not be train dataset, but is_train: {is_train}'
        img_path_list = glob(os.path.join(root_dir, f'*.png'))
        img_path_list.sort()
        super().__init__(img_path_list, is_train, image_size)


