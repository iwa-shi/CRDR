from typing import Dict

from os import path as osp
from copy import deepcopy

import torch.nn as nn

from src.utils.logger import log_dict_items
from src.utils.misc import import_modules

import_modules('src.models.discriminator', osp.dirname(osp.abspath(__file__)), suffix='_discriminator.py')

from src.utils.registry import DISCRIMINATOR_REGISTRY

def build_discriminator(discriminator_opt: Dict) -> nn.Module:
    """Build Discriminator from registry

    Args:
        discriminator_opt (Dict): must include 'type' key

    Returns:
        subet[nn.Module]: Discriminator
    """
    subnet_opt = deepcopy(discriminator_opt)
    network_type = subnet_opt.pop('type')
    subnet = DISCRIMINATOR_REGISTRY.get(network_type)(**subnet_opt)
    log_dict_items(subnet_opt, level='DEBUG', indent=True)
    return subnet
