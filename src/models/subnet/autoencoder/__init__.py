from os import path as osp

from src.utils.misc import import_modules


import_modules('src.models.subnet.autoencoder', osp.dirname(osp.abspath(__file__)), suffix='_autoencoder.py')