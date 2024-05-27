import importlib
from copy import deepcopy
from os import path as osp

from src.utils.registry import MODEL_REGISTRY
from src.utils.logger import get_root_logger
from .comp_model import *
from .subnet import *

__all__ = ['build_comp_model']

# automatically scan and import model modules for registry
# scan all the files under the 'models' folder and collect files ending with
# '_model.py'
# model_folder = osp.dirname(osp.abspath(__file__))
# model_filenames = [osp.splitext(osp.basename(v))[0] for v in scandir(model_folder) if v.endswith('_model.py')]
# import all the model modules
# _model_modules = [importlib.import_module(f'basicsr.models.{file_name}') for file_name in model_filenames]


def build_comp_model(opt):
    """Build model from options.
    Args:
        opt (Config): Configuration. It must contain:
            model_type (str): Model type.
    """
    opt = deepcopy(opt)
    model = MODEL_REGISTRY.get(opt['model_type'])(opt)
    # logger.info(f'Model [{model.__class__.__name__}] is created.')
    return model

def build_trained_comp_model(opt, ckpt_path: str):
    model = build_comp_model(opt)
    model.load_learned_weight(ckpt_path=ckpt_path)
    return model