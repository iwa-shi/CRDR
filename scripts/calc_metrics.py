import argparse
import json
import os
import shutil

import tempfile
import time
import traceback
from abc import ABCMeta, abstractmethod
from concurrent import futures
from typing import Dict, List, Tuple

import cv2
import lpips
import numpy as np
import skimage
import torch
import torchvision.transforms as T
from DISTS_pytorch import DISTS
from PIL import Image
from pytorch_fid.fid_score import (
    calculate_fid_given_paths,
)
from glob import glob
from tqdm import tqdm
from torch.utils.data import Dataset

from src.utils.options import TestConfig
from src.utils.logger import get_root_logger, bolded_log


class CustomConfig(TestConfig):
    @classmethod
    def get_opt(cls) -> "CustomConfig":
        arg_dict = cls.arg_parse()
        return cls(arg_dict)

    @staticmethod
    def arg_parse() -> Dict:
        parser = argparse.ArgumentParser()
        parser.add_argument("--real_dir", type=str)
        parser.add_argument("--fake_dir", type=str)
        parser.add_argument("-d", "--device", type=str, default="cuda:0")

        args = parser.parse_args()
        out_dict = vars(args)  # argparse.Namespace -> Dict
        return out_dict


class ImageDataset(Dataset):
    def __init__(
        self, real_path_list: List[str], fake_path_list: List[str], scale: str
    ) -> None:
        super().__init__()
        assert scale in ["0_1", "-1_1"]
        self.real_path_list = real_path_list
        self.fake_path_list = fake_path_list
        transform_list = [T.ToTensor()]
        if scale == "-1_1":
            transform_list.append(T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]))
        self.transform = T.Compose(transform_list)

    def __len__(self) -> int:
        return len(self.real_path_list)

    def read_img(self, path):
        img = Image.open(path).convert("RGB")
        return self.transform(img)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        real_path = self.real_path_list[index]
        fake_path = self.fake_path_list[index]
        return self.read_img(real_path), self.read_img(fake_path)


def get_dataloader(real_path_list: List[str], fake_path_list: List[str], scale: str):
    dataset = ImageDataset(real_path_list, fake_path_list, scale)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, drop_last=False, shuffle=False, num_workers=1
    )
    return dataloader


class BaseMetric(metaclass=ABCMeta):
    def __init__(self, opt, metric_name: str):
        self.opt = opt
        self.metric_name = metric_name
        self.logger = get_root_logger()

    @staticmethod
    def get_real_fake_path_list(
        real_dir: str, fake_dir: str
    ) -> Tuple[List[str], List[str]]:
        assert os.path.exists(real_dir)
        assert os.path.exists(fake_dir)
        real_path_list = glob(os.path.join(real_dir, f"*.png"))
        fake_path_list = glob(os.path.join(fake_dir, f"*.png"))
        assert len(real_path_list) == len(fake_path_list)
        real_path_list.sort()
        fake_path_list.sort()
        for r, f in zip(real_path_list, fake_path_list):
            assert os.path.basename(r) == os.path.basename(f)
        return real_path_list, fake_path_list

    @abstractmethod
    def calc_metric(self, real_path_list, fake_path_list) -> float:
        raise NotImplementedError()

    def run(self, real_dir: str, fake_dir: str) -> float:
        real_path_list, fake_path_list = self.get_real_fake_path_list(
            real_dir, fake_dir
        )
        num_img = len(real_path_list)
        img_avg_value = self.calc_metric(real_path_list, fake_path_list)
        self.logger.info(f"{num_img} images: {self.metric_name}: {img_avg_value:.4}")
        return img_avg_value


class PSNRMetric(BaseMetric):
    def __init__(self, opt):
        super().__init__(opt, "PSNR")

    @staticmethod
    def read_img(img_path: str) -> np.ndarray:
        """return np RGB [0, 1]"""
        img_np = np.asarray(Image.open(img_path).convert("RGB"), dtype=np.float32)
        return img_np

    def calc_metric(self, real_path_list, fake_path_list) -> float:
        self.num_dims = 0
        self.sqerror_values = []
        self.img_psnr_values = []
        max_workers = 8
        future_list = []
        with futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future = executor.submit(self.monitor_progress, total=len(real_path_list))
            future_list.append(future)
            for real_path, fake_path in zip(real_path_list, fake_path_list):
                future = executor.submit(
                    self._calc_one_image, real_path=real_path, fake_path=fake_path
                )
                future_list.append(future)
            _ = futures.as_completed(fs=future_list)
        # mse = np.sum(self.sqerror_values) / self.num_dims
        # pix_avg_psnr = 20. * np.log10(255.) - 10. * np.log10(mse)
        img_avg_psnr = np.mean(self.img_psnr_values)
        return img_avg_psnr

    def monitor_progress(self, total):
        strt = time.time()
        n = len(self.sqerror_values)
        while n < total:
            print(
                f"\r{n:6} / {total} imgs {time.time()-strt:5.2f}s", flush=True, end=""
            )
            time.sleep(0.1)
            n = len(self.sqerror_values)
        print("\r", end="")

    def _calc_one_image(self, real_path, fake_path):
        assert os.path.basename(fake_path) == os.path.basename(real_path)
        image0 = self.read_img(real_path)
        image1 = self.read_img(fake_path)
        self.num_dims += image0.size
        sqerror = np.sum(np.square(image1 - image0))
        self.sqerror_values.append(sqerror)
        _mse = sqerror / image0.size
        self.img_psnr_values.append(20.0 * np.log10(255.0) - 10.0 * np.log10(_mse))


class LPIPSMetric(BaseMetric):
    def __init__(self, opt):
        super().__init__(opt, "LPIPS")
        self.device = opt.device
        self.lpips_fn = lpips.LPIPS(net="alex").to(self.device)
        self.wandb_save_type = "pix_avg"

    @torch.no_grad()
    def calc_metric(self, real_path_list, fake_path_list) -> float:
        lpips_list = []
        dataloader = get_dataloader(real_path_list, fake_path_list, scale="-1_1")

        for img_real, img_fake in tqdm(
            dataloader, total=len(dataloader), leave=False, ncols=80
        ):
            _, _, H, W = img_real.size()
            assert img_fake.shape == img_real.shape
            dist = self.lpips_fn.forward(
                img_fake.to(self.device), img_real.to(self.device)
            )  # calc LPIPS
            lpips_list.append(dist.item())
        return np.mean(lpips_list)


class DISTSMetric(BaseMetric):
    def __init__(self, opt):
        super().__init__(opt, "DISTS")
        self.device = opt.device
        self.dists_fn = DISTS().to(self.device)

    @torch.no_grad()
    def calc_metric(self, real_path_list, fake_path_list) -> float:
        dists_list = []
        dataloader = get_dataloader(real_path_list, fake_path_list, scale="0_1")

        for img_real, img_fake in tqdm(
            dataloader, total=len(dataloader), leave=False, ncols=80
        ):
            _, _, H, W = img_real.size()
            assert img_fake.shape == img_real.shape
            # calculate DISTS between X, Y (a batch of RGB images, data range: 0~1)
            dist = self.dists_fn.forward(
                img_fake.to(self.device), img_real.to(self.device), require_grad=False
            )
            dists_list.append(dist.item())
        return np.mean(dists_list)


class FIDMetric(BaseMetric):
    def __init__(self, opt):
        self.use_hific_patch = True
        self.feature_dims = 2048
        self.fid_num_workers = 8
        self.fid_batch_size = 100
        super().__init__(opt, metric_name="FID")
        self.device = opt.device

    def calc_metric(
        self, real_path_list: List[str], fake_path_list: List[str]
    ) -> float:
        np.random.seed(0)
        num_img = len(real_path_list)
        if len(real_path_list) < 50:
            self.logger.error(f"num_img (={num_img}) is too small to calc FID")
            return -1

        tmp_dir = tempfile.mkdtemp()
        fake_patch_dir = os.path.join(tmp_dir, "fake_patches")
        real_patch_dir = os.path.join(tmp_dir, "real_patches")
        os.makedirs(fake_patch_dir)
        os.makedirs(real_patch_dir)

        try:
            self.save_hific_fid_patches(
                save_dir=real_patch_dir, img_path_list=real_path_list, max_workers=8
            )
            self.save_hific_fid_patches(
                save_dir=fake_patch_dir, img_path_list=fake_path_list, max_workers=8
            )
            fid_val = calculate_fid_given_paths(
                [fake_patch_dir, real_patch_dir],
                batch_size=self.fid_batch_size,
                device=self.device,
                dims=self.feature_dims,
                num_workers=self.fid_num_workers,
            )
        finally:
            shutil.rmtree(fake_patch_dir)
        return fid_val

    def save_hific_fid_patches(
        self, save_dir: str, img_path_list: List[str], max_workers: int = 8
    ):
        future_list = []
        with futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future = executor.submit(self.monitor_progress, dir_name=save_dir)
            future_list.append(future)
            for fake_path in img_path_list:
                future = executor.submit(
                    FIDMetric._save_hific_fid_patches,
                    save_dir=save_dir,
                    img_path=fake_path,
                )
                future_list.append(future)
            _ = futures.as_completed(fs=future_list)

    @staticmethod
    def monitor_progress(dir_name):
        retry_count, max_retry = 0, 5
        n = len(os.listdir(dir_name))
        while True:
            print(f"\r{n:6} patches", flush=True, end="")
            time.sleep(0.2)
            m = len(os.listdir(dir_name))
            if m - n == 0:
                if m > 0 or retry_count == max_retry:
                    return
                else:
                    retry_count += 1
            else:
                retry_count = 0
            n = m

    @staticmethod
    def _save_hific_fid_patches(save_dir: str, img_path: str):
        assert os.path.exists(img_path)
        img_name = os.path.basename(img_path).split(".")[0]
        img = cv2.imread(img_path)
        out = FIDMetric.crop_hific_fid_patches(img, 256)
        for i in range(out.shape[0]):
            save_path = os.path.join(save_dir, img_name + f"_{i:04}.png")
            cv2.imwrite(save_path, out[i, :, :, :])

    @staticmethod
    def crop_hific_fid_patches(img: np.ndarray, patch_size: int) -> np.ndarray:
        p = patch_size
        H, W = img.shape[:2]
        ## crop
        out1 = skimage.util.view_as_blocks(
            img[: H // p * p, : W // p * p, :], (p, p, 3)
        ).reshape(-1, p, p, 3)
        ## shift and crop
        o = p // 2
        sH, sW = H - o, W - o
        out2 = skimage.util.view_as_blocks(
            img[o : o + sH // p * p, o : o + sW // p * p, :], (p, p, 3)
        ).reshape(-1, p, p, 3)
        return np.concatenate([out1, out2], axis=0)


def retrieve_bitrate(fake_dir: str) -> float:
    rate_json = os.path.join(fake_dir, "_avg_bitrate.json")
    assert os.path.exists(rate_json)
    with open(rate_json, "r") as f:
        rate_dict = json.load(f)
    return rate_dict["avg_bpp"]


def main():
    opt = CustomConfig.get_opt()
    logger = get_root_logger(log_level="INFO")
    real_dir = opt.real_dir
    fake_dir = opt.fake_dir

    metrics_dict = {
        "PSNR": PSNRMetric(opt),
        "FID": FIDMetric(opt),
        "LPIPS": LPIPSMetric(opt),
        "DISTS": DISTSMetric(opt),
    }

    logger.info("Calculate " + ", ".join(list(metrics_dict.keys())))

    out_dict = {
        "bpp": retrieve_bitrate(fake_dir),
    }

    for metrics_name, met_obj in metrics_dict.items():
        bolded_log(msg=f"Calc {metrics_name}", level="INFO", new_line=True)
        try:
            met_val = met_obj.run(real_dir, fake_dir)
            out_dict[metrics_name] = met_val
        except KeyboardInterrupt:
            traceback.print_exc()
            exit()
        except:
            logger.error(f"ERROR: skip {fake_dir}")
            traceback.print_exc()

    json_path = os.path.join(fake_dir, "_metrics.json")
    with open(json_path, "w") as f:
        json.dump(out_dict, f, indent=4)

    logger.info(f"Results: {fake_dir}")
    for k, v in out_dict.items():
        logger.info(f"{k:>7}: {v:.4f}")


if __name__ == "__main__":
    main()
