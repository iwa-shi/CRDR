from typing import Dict, List, Union

import os
import torch

from .logger import get_root_logger
from .path import PathHandler

class Saver(object):
    def __init__(self, ckpt_root: str, exp: str, save_step: int, keep_step: Union[int, List]):
        self.path_hander = PathHandler(ckpt_root, exp)
        self.save_step = save_step
        self.keep_step = keep_step
        if isinstance(keep_step, list):
            self.keep_step = set(keep_step)

    def _should_keep(self, itr: int):
        if isinstance(self.keep_step, int):
            return (itr % self.keep_step == 0)
        return (itr in self.keep_step)

    def save(self, network_dict: Dict, save_label: str, current_iter: int, keep: bool) -> None:
        """
        Save model to `{save_dir}/{save_label}_iter{current_iter}.pth.tar`

        Args:
            network_dict (Dict): {"key": model} e.g., {"comp_model": comp_model}
            save_label (str): "comp_model", "discriminator", "training_state", etc.
            current_iter (int):
            keep (bool): if False, delete the model saved at `current_iter - save_step`
        """
        self._save_network(network_dict, save_label, current_iter)
        last_iter = current_iter - self.save_step
        if last_iter == 0:
            return
        if not(keep) or not(self._should_keep(last_iter)):
            self._delete_network(save_label, last_iter)

    def _save_network(self, network_dict: Dict, save_label: str, current_iter: int) -> None:
        state_dict = {'iter': current_iter}
        for key, network in network_dict.items():
            state_dict[key] = network.state_dict()
        save_path = self._get_save_path(save_label, current_iter)
        logger = get_root_logger()
        logger.debug(f'saving {save_label} iter{current_iter} ("{save_path}")')
        torch.save(state_dict, save_path)

    def _delete_network(self, save_label: str, delete_iter: int) -> None:
        delete_path = self._get_save_path(save_label, delete_iter)
        logger = get_root_logger()
        logger.debug(f'deleting {save_label} iter{delete_iter} ("{delete_path}")')
        if os.path.exists(delete_path):
            os.remove(delete_path)
        else:
            logger.warning(f'Tried to delete checkpoint "{save_label}" iter{delete_iter}, but "{delete_path}" does not exist.')

    @staticmethod
    def get_save_path(save_dir: str, save_label: str, itr: int) -> str:
        itr_str = PathHandler.iter2str(itr)
        return os.path.join(save_dir, f'{save_label}_iter{itr_str}.pth.tar')

    def _get_save_path(self, save_label: str, itr: int) -> str:
        return self.path_hander.get_ckpt_path(save_label, itr)