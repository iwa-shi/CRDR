from typing import Dict, Optional

import os.path as osp
from copy import deepcopy
from src.utils.logger import IndentedLog, get_root_logger, log_dict_items

from src.utils.misc import import_modules
from src.utils.registry import LOSS_REGISTRY

# import all loss modules to register
import_modules('src.losses', osp.dirname(osp.abspath(__file__)), suffix='_loss.py')

def build_loss(opt: Dict, loss_name: Optional[str]=None):
    """Build loss class from registry

    Args:
        opt (Dict): must include 'type' key
        display_name (str): only for logging

    Returns:
        [type]: loss
    """
    opt = deepcopy(opt)
    loss_type = opt.pop('type')
    loss = LOSS_REGISTRY.get(loss_type, display_name=loss_name)(**opt)
    log_dict_items(opt, level='DEBUG', indent=True)
    return loss