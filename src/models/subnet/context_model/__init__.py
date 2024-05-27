from os import path as osp

from src.utils.misc import import_modules

import_modules('src.models.subnet.context_model', osp.dirname(osp.abspath(__file__)), suffix='_context_model.py')