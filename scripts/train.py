import os

from src.utils.options import TrainConfig
from src.trainer import build_trainer
from src.utils.logger import get_root_logger, bolded_log
from src.utils.misc import dict2str
from src.utils.path import PathHandler

import src # To register classes

def set_msg_logger(opt):
    debug_mode = opt.debug
    log_level = 'DEBUG' if debug_mode else 'INFO'
    return get_root_logger(log_level=log_level, log_file=opt.path.log_msg_path)

def main():
    opt = TrainConfig.get_opt(config_dir='./config')
    path_handler = PathHandler(opt.ckpt_root, opt.exp)
    path_handler.make_job_dir()
    opt.dump(filename=os.path.join(path_handler.job_dir, 'config.yaml'))

    logger = set_msg_logger(opt)
    bolded_log('Config', level="DEBUG")
    logger.debug(dict2str(opt, level=0))

    trainer = build_trainer(opt)
    trainer.train_loop()

if __name__ == '__main__':
    main()