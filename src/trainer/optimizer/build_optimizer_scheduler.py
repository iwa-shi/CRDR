from typing import Dict

from copy import deepcopy

import torch

from src.utils.registry import OPTIMIZER_REGISTRY, SCHEDULER_REGISTRY
from src.utils.logger import get_root_logger, IndentedLog, log_dict_items


def register_optimizers() -> None:
    OPTIMIZER_REGISTRY.register()(torch.optim.Adam)
    OPTIMIZER_REGISTRY.register()(torch.optim.SGD)

def register_schedulers() -> None:
    SCHEDULER_REGISTRY.register()(torch.optim.lr_scheduler.MultiStepLR)

register_optimizers()
register_schedulers()

def get_params_list(parameters_dict, paramwise_opt, base_lr):
    params = []
    all_keys = set(list(parameters_dict.keys()))
    sorted_keys = sorted(list(parameters_dict.keys()))

    for _opt in paramwise_opt:
        queries = _opt['keys']
        lr_mult = _opt['lr_mult'] # _opt.get('lr_mult', 1.)
        _params = []
        for k in sorted_keys:
            hit = [(q in k) for q in queries]
            if any(hit):
                all_keys.remove(k)
                v = parameters_dict[k]
                if v.requires_grad:
                    _params.append(v)
        params.append({'params': _params.copy(), 'lr': lr_mult*base_lr})
    
    _params = []
    for k in sorted_keys:
        if k not in all_keys:
            continue
        v = parameters_dict[k]
        if v.requires_grad:
            _params.append(v)
    params.append({'params': _params.copy()})
    return params

def build_optimizer(parameters_dict: Dict, optimizer_opt: Dict):
    """Build optimizer from registry

    Args:
        parameters_dict (Dict): paramters of model
        optimizer_opt (Dict): must include 'type' key

    Returns:
        [type]: optimizer
    """
    optimizer_opt = deepcopy(optimizer_opt)
    _optimizer_opt = deepcopy(optimizer_opt) ## for logging
    optimizer_type = optimizer_opt.pop('type')
    paramwise_opt = optimizer_opt.pop('paramwise_opt', [])
    if paramwise_opt:
        params = get_params_list(parameters_dict, paramwise_opt, optimizer_opt['lr'])
    else:
        params = (v for v in parameters_dict.values() if v.requires_grad)
    
    optimizer = OPTIMIZER_REGISTRY.get(optimizer_type)(params=params, **optimizer_opt)
    log_dict_items(_optimizer_opt, level='DEBUG', indent=True)
    return optimizer


def build_scheduler(optimizer, scheduler_opt: Dict):
    scheduler_opt = deepcopy(scheduler_opt)
    scheduler_type = scheduler_opt.pop('type')
    scheduler = SCHEDULER_REGISTRY.get(scheduler_type)(optimizer, **scheduler_opt)
    log_dict_items(scheduler_opt, level='DEBUG', indent=True)
    return scheduler