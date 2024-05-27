# from https://github.com/open-mmlab/mmcv/blob/master/mmcv/utils/config.py
import argparse
import copy
import os
import os.path as osp
from typing import Dict, List, Optional, Tuple

import yaml
from addict import Dict as Addict

from .path import PathHandler, check_file_exist

BASE_KEY = "_base_"
DELETE_KEY = "_delete_"
DEPRECATION_KEY = "_deprecation_"
RESERVED_KEYS = ["filename", "text"]


class ConfigDict(Addict):

    def __missing__(self, name):
        raise KeyError(name)

    def __getattr__(self, name):
        try:
            value = super(ConfigDict, self).__getattr__(name)
        except KeyError:
            ex = AttributeError(
                f"'{self.__class__.__name__}' object has no " f"attribute '{name}'"
            )
        except Exception as e:
            ex = e
        else:
            return value
        raise ex


# class Config:
class BaseConfig:
    """A facility for config and config files.
    It supports common file formats as configs: python/json/yaml. The interface
    is the same as a dict object and also allows access config values as
    attributes.
    Example:
        >>> cfg = Config(dict(a=1, b=dict(b1=[0, 1])))
        >>> cfg.a
        1
        >>> cfg.b
        {'b1': [0, 1]}
        >>> cfg.b.b1
        [0, 1]
        >>> cfg = Config.fromfile('tests/data/config/a.py')
        >>> cfg.filename
        "/home/kchen/projects/mmcv/tests/data/config/a.py"
        >>> cfg.item4
        'test'
        >>> cfg
        "Config [path: /home/kchen/projects/mmcv/tests/data/config/a.py]: "
        "{'item1': [1, 2], 'item2': {'a': 0}, 'item3': True, 'item4': 'test'}"
    """

    @staticmethod
    def _file2dict_yaml(filename: str) -> Tuple[Dict, str, List]:
        filename = osp.abspath(osp.expanduser(filename))
        check_file_exist(filename)
        loaded_yamls = [filename]
        fileExtname = osp.splitext(filename)[1]
        if fileExtname not in [".yaml"]:
            raise IOError("Only yaml type are supported now!")

        with open(filename) as f:
            cfg_dict: Dict = yaml.safe_load(f)

        cfg_text = filename + "\n"
        with open(filename, "r", encoding="utf-8") as f:
            cfg_text += f.read()

        if BASE_KEY in cfg_dict:
            cfg_dir = osp.dirname(filename)
            base_filename = cfg_dict.pop(BASE_KEY)
            base_filename = (
                base_filename if isinstance(base_filename, list) else [base_filename]
            )

            cfg_dict_list = list()
            cfg_text_list = list()
            for f in base_filename:
                _cfg_dict, _cfg_text, _loaded_yamls = BaseConfig._file2dict_yaml(
                    osp.join(cfg_dir, f)
                )
                cfg_dict_list.append(_cfg_dict)
                cfg_text_list.append(_cfg_text)
                loaded_yamls.extend(_loaded_yamls)

            base_cfg_dict = dict()
            for c in cfg_dict_list:
                duplicate_keys = base_cfg_dict.keys() & c.keys()
                if len(duplicate_keys) > 0:
                    raise KeyError(
                        "Duplicate key is not allowed among bases. "
                        f"Duplicate keys: {duplicate_keys}"
                    )
                base_cfg_dict.update(c)

            base_cfg_dict = BaseConfig._merge_a_into_b(cfg_dict, base_cfg_dict)
            cfg_dict = base_cfg_dict

            # merge cfg_text
            cfg_text_list.append(cfg_text)
            cfg_text = "\n".join(cfg_text_list)

        return cfg_dict, cfg_text, loaded_yamls

    @staticmethod
    def _merge_a_into_b(a: Dict, b: Dict) -> Dict:
        b = b.copy()
        for k, v in a.items():
            # if (1) v is dict, (2) both a and b have k, (3) v does not have DELETE_KEY
            if isinstance(v, dict) and (k in b) and not v.pop(DELETE_KEY, False):
                if not isinstance(b[k], dict):
                    raise TypeError(
                        f"{k}={v} in child config cannot inherit from base "
                        f"because {k} is a dict in the child config but is of "
                        f"type {type(b[k])} in base config. You may set "
                        f"`{DELETE_KEY}=True` to ignore the base config"
                    )
                b[k] = BaseConfig._merge_a_into_b(v, b[k])
            else:
                b[k] = v
        return b

    def __init__(self, cfg_dict=None, cfg_text=None, filename=None):
        if cfg_dict is None:
            cfg_dict = dict()
        elif not isinstance(cfg_dict, dict):
            raise TypeError("cfg_dict must be a dict, but " f"got {type(cfg_dict)}")
        for key in cfg_dict:
            if key in RESERVED_KEYS:
                raise KeyError(f"{key} is reserved for config file")

        super(BaseConfig, self).__setattr__("_cfg_dict", ConfigDict(cfg_dict))
        super(BaseConfig, self).__setattr__("_filename", filename)
        if cfg_text:
            text = cfg_text
        elif filename:
            with open(filename, "r") as f:
                text = f.read()
        else:
            text = ""
        super(BaseConfig, self).__setattr__("_text", text)

    @property
    def filename(self):
        return self._filename

    @property
    def text(self):
        return self._text

    def __repr__(self):
        return f"Config (path: {self.filename}): {self._cfg_dict.__repr__()}"

    def __len__(self):
        return len(self._cfg_dict)

    def __getattr__(self, name):
        return getattr(self._cfg_dict, name)

    def __getitem__(self, name):
        return self._cfg_dict.__getitem__(name)

    def __setattr__(self, name, value):
        if isinstance(value, dict):
            value = ConfigDict(value)
        self._cfg_dict.__setattr__(name, value)

    def __setitem__(self, name, value):
        if isinstance(value, dict):
            value = ConfigDict(value)
        self._cfg_dict.__setitem__(name, value)

    def __iter__(self):
        return iter(self._cfg_dict)

    def __getstate__(self):
        return (self._cfg_dict, self._filename, self._text)

    def __setstate__(self, state):
        _cfg_dict, _filename, _text = state
        super(BaseConfig, self).__setattr__("_cfg_dict", _cfg_dict)
        super(BaseConfig, self).__setattr__("_filename", _filename)
        super(BaseConfig, self).__setattr__("_text", _text)

    def dump(self, filename: str) -> None:
        cfg_dict = super(BaseConfig, self).__getattribute__("_cfg_dict").to_dict()
        with open(filename, "w") as f:
            yaml.dump(cfg_dict, f)


class TrainConfig(BaseConfig):
    @classmethod
    def get_opt(cls, config_dir: str) -> "TrainConfig":
        arg_dict = cls.arg_parse()
        filename = arg_dict["config_path"]
        cfg_dict, cfg_text, loaded_yamls = cls._file2dict_yaml(filename)
        cfg_dict["loaded_yamls"] = loaded_yamls

        arg_dict = cls._merge_a_into_b(arg_dict, cfg_dict)
        arg_dict["exp"] = os.path.basename(filename).split(".")[0]
        arg_dict["path"] = cls.get_path_dict(arg_dict)
        arg_dict["host"] = os.uname()[1]
        arg_dict["is_train"] = True
        return cls(arg_dict, cfg_text=cfg_text, filename=filename)

    @staticmethod
    def arg_parse() -> Dict:
        """NOTE: command line arguments are prioritized over yaml files"""
        parser = argparse.ArgumentParser()
        parser.add_argument("config_path", type=str)
        parser.add_argument("-d", "--device", type=str, default="cuda:0")
        parser.add_argument("-si", "--start_iter", type=int)
        parser.add_argument("-e", "--eval_step", type=int)
        parser.add_argument("-l", "--log_step", type=int)
        parser.add_argument("-s", "--save_step", type=int)
        parser.add_argument("-b", "--batch_size", type=int)
        parser.add_argument("-ti", "--total_iter", type=int)
        parser.add_argument("-nw", "--num_workers", type=int, default=8)
        parser.add_argument("-wb", "--use_wandb", action="store_true")
        parser.add_argument(
            "-dr",
            "--dry_run",
            action="store_true",
            help="If true, print config and models and exit",
        )
        parser.add_argument(
            "--debug",
            action="store_true",
            help='Set logger level to "DEBUG". All args will be printed.',
        )
        parser.add_argument(
            "--wandb_dryrun", action="store_true", help="Running wandb with dryrun mode"
        )

        args = parser.parse_args()

        if args.use_wandb and args.debug:
            raise ValueError(
                f"--debug and --use_wandb cannot be turned on at the same time."
            )
        if args.dry_run:
            args.device = "cpu"
            args.debug = True

        out_dict = vars(args)  # argparse.Namespace -> Dict

        nonetype_keys = [k for k, v in out_dict.items() if v is None]
        for k in nonetype_keys:
            del out_dict[k]

        if "batch_size" in out_dict:
            bs = out_dict.pop("batch_size")
            out_dict["dataset"] = {"batch_size": bs}

        return out_dict

    @staticmethod
    def get_path_dict(cfg_dict: Dict) -> Dict:
        ckpt_root = cfg_dict["ckpt_root"]
        exp_name = cfg_dict["exp"]

        assert os.path.exists(
            ckpt_root
        ), f'checkpoint_root directory "{ckpt_root}" does not exist.'
        path_handler = PathHandler(ckpt_root, exp_name)

        # path_handler.make_job_dir()
        path_dict = path_handler.get_exp_path_dict()
        return path_dict


class TestConfig(BaseConfig):
    @classmethod
    def get_opt(cls, config_dir: str, arg_dict: Optional[Dict] = None) -> "TestConfig":
        if not (arg_dict):
            arg_dict = cls.arg_parse()

        filename = arg_dict["config_path"]
        cfg_dict, cfg_text, loaded_yamls = cls._file2dict_yaml(filename)
        cfg_dict["loaded_yamls"] = loaded_yamls
        arg_dict = cls._merge_a_into_b(arg_dict, cfg_dict)
        arg_dict["exp"] = os.path.basename(filename).split(".")[0]
        arg_dict["path"] = cls.get_path_dict(arg_dict)
        arg_dict["host"] = os.uname()[1]
        arg_dict["is_train"] = False
        arg_dict["dataset"]["test_dataset"] = copy.deepcopy(
            cls.get_test_dataset_config(arg_dict)
        )
        return cls(arg_dict, cfg_text=cfg_text, filename=filename)

    @staticmethod
    def arg_parse() -> Dict:
        parser = argparse.ArgumentParser()
        parser.add_argument("config_path", type=str)
        parser.add_argument("iter", type=int)
        parser.add_argument("test_dataset_name", type=str)
        parser.add_argument("-s", "--sample_size", type=int, default=1000000)
        parser.add_argument("-d", "--device", type=str, default="cuda:0")

        store_true_args = ["notsave", "notgtmask", "debug"]
        for key in store_true_args:
            parser.add_argument(f"--{key}", action="store_true")

        args = parser.parse_args()
        out_dict = vars(args)  # argparse.Namespace -> Dict

        nonetype_keys = [k for k, v in out_dict.items() if v is None]
        for k in nonetype_keys:
            del out_dict[k]
        for key in store_true_args:
            if not (out_dict[key]):
                del out_dict[key]

        return out_dict

    @staticmethod
    def get_path_dict(cfg_dict: Dict) -> Dict:
        ckpt_root = cfg_dict["ckpt_root"]
        exp_name = cfg_dict["exp"]  # like "exp1-2_1"
        load_iter = cfg_dict["iter"]
        dataset_name = cfg_dict["test_dataset_name"]
        path_handler = PathHandler(ckpt_root, exp_name)
        path_dict = path_handler.get_exp_path_dict()
        model_path = path_handler.get_ckpt_path("comp_model", itr=load_iter)
        assert os.path.exists(model_path), f'model_path "{model_path}" does not exist.'
        sample_dir = os.path.join(
            path_dict["sample_dir"], f"{exp_name}_iter{load_iter//1000}K_{dataset_name}"
        )
        return {
            "ckpt_root": ckpt_root,
            "model_dir": path_dict["model_dir"],
            "sample_dir": sample_dir,
            "model_path": model_path,
        }

    @staticmethod
    def get_test_dataset_config(cfg_dict: Dict) -> Dict:
        test_dataset_config = copy.deepcopy(cfg_dict["dataset"]["eval_dataset"])
        test_dataset_name = cfg_dict["test_dataset_name"]
        test_dataset_config["name"] = test_dataset_name
        if "notgtmask" in cfg_dict:
            test_dataset_config["use_gt_mask"] = not (cfg_dict["notgtmask"])
        return test_dataset_config
