import importlib
import os.path as osp
from glob import glob

import subprocess
import time
from datetime import datetime

import torch
from tqdm import tqdm

class Color:
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    COLOR_DEFAULT = "\033[39m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    INVISIBLE = "\033[08m"
    REVERCE = "\033[07m"
    BG_BLACK = "\033[40m"
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"
    BG_MAGENTA = "\033[45m"
    BG_CYAN = "\033[46m"
    BG_WHITE = "\033[47m"
    BG_DEFAULT = "\033[49m"
    RESET = "\033[0m"


def import_modules(base_dir: str, search_dir: str, suffix: str = ".py"):
    filenames = [
        osp.splitext(osp.basename(v))[0]
        for v in glob(osp.join(search_dir, f"*{suffix}"))
    ]
    # print(filenames)
    for filename in filenames:
        importlib.import_module(f"{base_dir}.{filename}")


def dict2str(dic, level=0, indent_width=2, val_color="YELLOW", prefix=None):
    """dict to string for printing options.
    Args:
        opt (dict): Option dict.
        indent_level (int): Indent level. Default: 1.
    Return:
        (str): Option string for printing.
    """
    key_color_list = ["", Color.BLUE, Color.CYAN, Color.GREEN, Color.MAGENTA]
    _key_color = key_color_list[level % len(key_color_list)]
    _val_color = "" if val_color is None else getattr(Color, val_color.upper())

    if prefix is None:
        prefix = " " * indent_width

    def key_prefix(is_last):
        s = "┗" if is_last else "┣"
        return prefix + s + " " * (indent_width - 1)

    def val_prefix(is_last):
        s = " " if is_last else "┃"
        return prefix + s + " " * (indent_width - 1)

    msg = "\n"
    dict_len = len(dic)
    for i, (k, v) in enumerate(dic.items()):
        is_last_key = i == dict_len - 1
        key_str = key_prefix(is_last_key) + f"{_key_color}{k}{Color.RESET}: "
        if isinstance(v, dict):
            msg += key_str
            new_prefix = val_prefix(is_last_key)
            msg += dict2str(
                v, level + 1, indent_width=indent_width, prefix=new_prefix
            )
        elif isinstance(v, list):
            msg += key_str + "\n"
            new_prefix = val_prefix(is_last_key)
            for s in v:
                msg += new_prefix + f" - {_val_color}{s}{Color.RESET}\n"
        else:
            msg += key_str + f"{_val_color}{v}{Color.RESET}\n"
    return msg
