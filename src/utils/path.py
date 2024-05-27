import re
import os
import os.path as osp
from datetime import datetime
from glob import glob
from typing import Optional, Dict

def check_file_exist(filename: str, msg_tmpl='file "{}" does not exist'):
    if not osp.isfile(filename):
        raise FileNotFoundError(msg_tmpl.format(filename))


class PathHandler(object):
    def __init__(self, ckpt_root: str, exp: str) -> None:
        self.ckpt_root = ckpt_root
        self.exp = exp
        self.job_dir = os.path.join(self.ckpt_root, self.exp)

    def make_job_dir(self) -> None:
        job_dir = os.path.join(self.ckpt_root, self.exp)
        os.makedirs(os.path.join(job_dir, 'model'), exist_ok=True)
        os.makedirs(os.path.join(job_dir, 'sample'), exist_ok=True)

    def get_exp_path_dict(self) -> Dict:
        job_dir = os.path.join(self.ckpt_root, self.exp)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return {
            'ckpt_root': self.ckpt_root,
            'job_dir': job_dir,
            'model_dir': osp.join(job_dir, 'model'),
            'sample_dir': osp.join(job_dir, 'sample'),
            'log_loss_path': osp.join(job_dir, 'log_loss.csv'),
            'log_eval_path': osp.join(job_dir, 'eval_result.csv'),
            'log_msg_path': osp.join(job_dir, f'train_{timestamp}.log'),
            'sample_dir': osp.join(job_dir, 'sample'),
        }

    def get_ckpt_path(self, label: str, itr: int):
        model_dir = self.get_exp_path_dict()['model_dir']
        itr_str = self.iter2str(itr)
        return os.path.join(model_dir, f'{label}_iter{itr_str}.pth.tar')

    @staticmethod
    def iter2str(itr: int) -> str:
        if itr % 1000 == 0:
            return str(itr // 1000) + 'K'
        return str(itr)
