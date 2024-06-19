from collections import defaultdict
import copy
import logging
import os
import re
import pandas as pd

from python_log_indenter import IndentedLoggerAdapter

from typing import Dict, Optional, Union
from src.utils.misc import Color

initialized_logger = {}


class AvgMeter(object):
    def __init__(self):
        self.reset()

    def update(self, dic: Dict):
        for k, v in dic.items():
            self.sum_dict[k] += v
            self.count_dict[k] += 1

    def reset(self):
        self.sum_dict = defaultdict(float)
        self.count_dict = defaultdict(int)

    def get_avg_values(self) -> Dict:
        return {k: v / self.count_dict[k] for k, v in self.sum_dict.items()}


class CSVLogger(object):
    def __init__(self, log_path: str, resume: bool) -> None:
        """Logger

        Args:
            log_path (str):
            resume (bool):
        """
        self.log_path = log_path
        
        self.df = None
        if resume:
            if os.path.exists(log_path):
                self.df = pd.read_csv(log_path)
            else:
                logger = get_root_logger()
                logger.warning(f'Log file "{log_path}" not found.')

    def _save_log(self) -> None:
        if 'iter' in self.df:
            self.df = self.df.astype({'iter': int})
        self.df.to_csv(self.log_path, index=False)

    def update(self, log_dict:Dict) -> None:
        if self.df is None:
            columns = log_dict.keys()
            self.df = pd.DataFrame(columns=columns)
        self.df.loc[len(self.df), :] = pd.Series(log_dict)
        self._save_log()


class IndentedLog(object):
    def __init__(self, level="INFO", msg=None, new_line: bool=False):
        self.logger = get_root_logger()
        level = getattr(logging, level)
        if new_line:
            print()
        if msg is not None:
            self.logger.log(level, msg)

    def __enter__(self):
        self.logger.add()
        return self.logger

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.sub()


def indented_log(level: Union[int, str]="INFO", msg: Optional[str]=None, new_line: bool=False):
    """
    Example:
    ```
    @indented_log(level="INFO", msg="Indent begins...")
    def hoge():
        logger.info("Indented 1")
        logger.info("Indented 2")
    hoge()
    logger.info("No indent")
    ```

    The output is
    ```
    [INFO] Indent begins...
    [INFO]   Indented 1
    [INFO]   Indented 2
    [INFO] No indent
    ```
    """
    if isinstance(level, str):
        level = getattr(logging, level)
    def _receive_func(f):
        def _wrapper(*args, **kwargs):
            if new_line:
                print()
            logger = get_root_logger()
            if msg is not None:
                logger.log(level, msg)
            logger.add()
            out = f(*args, **kwargs)
            logger.sub()
            return out
        return _wrapper
    return _receive_func

def bolded_log(msg: str, level: Union[int, str]='INFO', new_line: bool=False, prefix: str='===== ', suffix: str=' ====='):
    if new_line:
        print()
    msg = f'{Color.BOLD}{prefix}{msg}{suffix}{Color.RESET}'
    logger = get_root_logger()
    if isinstance(level, str):
        level = getattr(logging, level)
    logger.log(level=level, msg=msg)


def log_dict_items(dic: Dict, level: Union[int, str]='INFO', indent: bool=True, key_color: Optional[str]=None, val_color: Optional[str]='YELLOW'):
    logger = get_root_logger()
    if isinstance(level, str):
        level = getattr(logging, level)
    if indent:
        logger.add()
    key_color = '' if key_color is None else getattr(Color, key_color.upper())
    val_color = '' if val_color is None else getattr(Color, val_color.upper())
    for k, v in dic.items():
        msg = f'{key_color}{k}{Color.RESET}: {val_color}{v}{Color.RESET}'
        logger.log(level=level, msg=msg)
    if indent:
        logger.sub()


class DelColorFormatter(logging.Formatter):
    def format(self, record):
        record = copy.deepcopy(record)
        record.msg = re.sub('\\033\[[0-9]*m', '', record.msg)
        return super().format(record)


class ColorStreamHandler(logging.StreamHandler):
    mapping = {
        "TRACE": "[ TRACE  ]",
        "DEBUG": "[  DEBUG ]",
        "INFO": "[  INFO  ]",
        "WARNING": f"{Color.RED}[ WARNING]{Color.RESET}",
        "WARN": f"{Color.RED}[  WARN  ]{Color.RESET}",
        "ERROR": f"{Color.BG_RED}[  ERROR  ]{Color.RESET}",
        "ALERT": f"{Color.BG_RED}[  ALERT  ]{Color.RESET}",
        "CRITICAL": f"{Color.BG_RED}[CRITICAL]{Color.RESET}",
    }
    def emit(self, record):
        record = copy.deepcopy(record)
        record.levelname = ColorStreamHandler.mapping[record.levelname]
        super().emit(record)


def get_root_logger(logger_name='basiccomp', log_level: Union[int, str]=logging.INFO, log_file=None):
    """Get the root logger.
    The logger will be initialized if it has not been initialized. By default a
    StreamHandler will be added. If `log_file` is specified, a FileHandler will
    also be added.
    Args:
        logger_name (str): root logger name. Default: 'basiccomp'.
        log_file (str | None): The log filename. If specified, a FileHandler
            will be added to the root logger.
        log_level (int or str): The root logger level. Note that only the process of
            rank 0 is affected, while other processes will set the level to
            "Error" and be silent most of the time.
    Returns:
        logging.Logger: The root logger.
    """
    if logger_name in initialized_logger:
        return initialized_logger[logger_name]
    if isinstance(log_level, str):
        log_level = getattr(logging, log_level)

    logger = IndentedLoggerAdapter(logging.getLogger(logger_name), spaces=2)
    logger.logger.setLevel(logging.DEBUG)

    format_str = '%(levelname)-10s %(message)s'
    stream_handler = ColorStreamHandler()
    stream_handler.setFormatter(logging.Formatter(format_str))
    stream_handler.setLevel(log_level)
    logger.logger.addHandler(stream_handler)
    logger.logger.propagate = False

    # add file handler
    if log_file is not None:
        file_handler = logging.FileHandler(log_file, 'w')
        file_format_str = '%(asctime)s %(levelname)-8s: %(message)s'
        file_handler.setFormatter(DelColorFormatter(file_format_str))
        file_handler.setLevel(logging.DEBUG)
        logger.logger.addHandler(file_handler)
    
    initialized_logger[logger_name] = logger
    return logger
