from typing import List, Tuple, Dict, Union

import itertools
from tqdm import tqdm
import tempfile

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F

class HeaderHandler:
    def __init__(self, use_non_zero_ind: bool=False):
        self.use_non_zero_ind = use_non_zero_ind
    
    @staticmethod
    def check_img_size(img_size):
        assert len(img_size) == 2
        assert isinstance(img_size[0], int)
        assert isinstance(img_size[1], int)

    def encode(self, img_size: Tuple[int, int], y_hat: torch.Tensor) -> bytes:
        self.check_img_size(img_size)
        max_val = int(torch.max(torch.abs(y_hat)))
        info_list = [
            np.array(list(img_size), dtype=np.uint16),
            np.array(max_val, dtype=np.uint8),
        ]
        if self.use_non_zero_ind:
            non_zero_ind_binary = self.encode_non_zero_ind(y_hat)
            info_list.append(non_zero_ind_binary)

        with tempfile.TemporaryFile() as f:
            for info in info_list:
                f.write(info.tobytes())
            f.seek(0)
            header_str = f.read()
        return header_str
    
    def decode(self, header_byte_string: bytes) -> Dict:
        img_size_buffer = header_byte_string[:4]
        img_size = np.frombuffer(img_size_buffer, dtype=np.uint16)
        H, W = int(img_size[0]), int(img_size[1])
        max_sample_buffer = header_byte_string[4:5]
        max_sample = np.frombuffer(max_sample_buffer, dtype=np.uint8)
        max_sample = int(max_sample)
        out_dict = {
            'img_size': (H, W),
            'max_sample': max_sample,
        }
        if self.use_non_zero_ind:
            non_zero_ind_buffer = header_byte_string[5:]
            non_zero_ind_binary = np.frombuffer(non_zero_ind_buffer, dtype=np.uint32)
            non_zero_ind = self.decode_nonzero_ind(non_zero_ind_binary)
            out_dict['non_zero_ind'] = non_zero_ind
        return out_dict

    def encode_non_zero_ind(self, y_hat: torch.Tensor) -> np.ndarray:
        y_hat_np = y_hat.cpu().detach().numpy()
        yC = y_hat_np.shape[1]
        sum_per_channel = np.sum(np.abs(y_hat_np), axis=(2, 3)).reshape(-1)
        sign = np.sign(sum_per_channel)
        nonzero_ind_list = []
        for l in np.split(sign, yC // 32):
            bin_str = ''.join([str(int(n)) for n in l])
            num = int(bin_str, 2)
            nonzero_ind_list.append(num)
        return np.array(nonzero_ind_list).astype(np.uint32)

    def decode_nonzero_ind(self, non_zero_ind_binary: np.ndarray) -> np.ndarray:
        arr_bin = [list(format(n, '032b')) for n in non_zero_ind_binary]
        arr_bin = list(itertools.chain.from_iterable(arr_bin))
        non_zero_ind = []
        for ind, val in enumerate(arr_bin):
            if val == '1':
                non_zero_ind.append(ind)
        non_zero_ind = np.array(non_zero_ind)
        return non_zero_ind


class MultiRateHeaderHandler(HeaderHandler):
    def encode(self, img_size: Tuple[int, int], y_hat: Tensor, rate_ind: Union[Tensor, float]) -> bytes:
        if isinstance(rate_ind, torch.Tensor):
            assert rate_ind.numel() == 1
            rate_ind = float(rate_ind.item())
        rate_ind = int(rate_ind * 16)
        self.check_img_size(img_size)
        max_val = int(torch.max(torch.abs(y_hat)))
        info_list = [
            np.array(list(img_size), dtype=np.uint16),
            np.array(max_val, dtype=np.uint8),
            np.array(rate_ind, dtype=np.uint8),
        ]
        if self.use_non_zero_ind:
            non_zero_ind_binary = self.encode_non_zero_ind(y_hat)
            info_list.append(non_zero_ind_binary)

        with tempfile.TemporaryFile() as f:
            for info in info_list:
                f.write(info.tobytes())
            f.seek(0)
            header_str = f.read()
        return header_str
    
    def decode(self, header_byte_string: bytes) -> Dict:
        img_size_buffer = header_byte_string[:4]
        img_size = np.frombuffer(img_size_buffer, dtype=np.uint16)
        H, W = int(img_size[0]), int(img_size[1])
        max_sample_buffer = header_byte_string[4:5]
        max_sample = np.frombuffer(max_sample_buffer, dtype=np.uint8)
        max_sample = int(max_sample)
        rate_ind_buffer = header_byte_string[5:6]
        rate_ind = np.frombuffer(rate_ind_buffer, dtype=np.uint8)
        rate_ind = float(rate_ind) / 16
        out_dict = {
            'img_size': (H, W),
            'max_sample': max_sample,
            'rate_ind': rate_ind,
        }
        if self.use_non_zero_ind:
            non_zero_ind_buffer = header_byte_string[5:]
            non_zero_ind_binary = np.frombuffer(non_zero_ind_buffer, dtype=np.uint32)
            non_zero_ind = self.decode_nonzero_ind(non_zero_ind_binary)
            out_dict['non_zero_ind'] = non_zero_ind
        return out_dict


def save_byte_strings(save_path: str, string_list: List) -> None:
    with open(save_path, 'wb') as f:
        for string in string_list:
            length = len(string)
            f.write(np.array(length, dtype=np.uint32).tobytes())
            f.write(string)

def load_byte_strings(load_path: str) -> List[bytes]:
    out_list = []
    with open(load_path, 'rb') as f:
        head = f.read(4)
        while head != b'':
            length = int(np.frombuffer(head, dtype=np.uint32)[0])
            out_list.append(f.read(length))
            head = f.read(4)
    return out_list


class IamgeInformation:
    def __init__(self, img_size: Tuple[int, int], max_sample: int, y_stride: int=16, z_stride: int=4) -> None:
        self.H, self.W = img_size
        self.max_sample = max_sample
        model_stride = y_stride * z_stride
        padH = int(np.ceil(self.H / model_stride) * model_stride)
        padW = int(np.ceil(self.W / model_stride) * model_stride)
        self.yH = padH // y_stride
        self.yW = padW // y_stride
        self.zH = padH // model_stride
        self.zW = padW // model_stride