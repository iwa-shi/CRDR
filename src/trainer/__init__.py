import os.path as osp
from copy import deepcopy

from src.utils.misc import import_modules
from src.utils.registry import TRAINER_REGISTRY
from src.utils.logger import bolded_log, log_dict_items

import_modules('src.trainer', osp.dirname(osp.abspath(__file__)), suffix='_trainer.py')

def build_trainer(opt):
    """

    Args:
        opt (Config):

    Returns:
        [type]: Trainer
    """
    bolded_log(msg='Trainer', new_line=True)
    if opt.get('trainer'):
        trainer_opt = deepcopy(opt['trainer'])
        trainer_type = trainer_opt.pop('type')
        trainer_cls = TRAINER_REGISTRY.get(trainer_type)
        if trainer_opt:
            log_dict_items(trainer_opt, level='DEBUG', indent=True)
        return trainer_cls(opt, **trainer_opt)

    raise ValueError('"trainer_type" key is not supported. Please use trainer.type')
