from copy import deepcopy
from glob import glob
import os
import os.path as osp

from src.utils.registry import MODEL_REGISTRY
from src.utils.misc import import_modules


import_modules('src.models.comp_model', osp.dirname(osp.abspath(__file__)), suffix='_model.py')

def build_model(opt):
    opt = deepcopy(opt)
    model = MODEL_REGISTRY.get(opt['model_type'])(opt)
    # logger = get_root_logger()
    # logger.info(f'Model [{model.__class__.__name__}] is created.')
    return model
