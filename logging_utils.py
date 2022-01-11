import logging
import os
import shutil
from typing import Dict, Tuple


def make_results_dir(exp_path: str, args: Dict) -> Tuple[str, str]:
    os.makedirs(exp_path, exist_ok=True)
    if args.mode == 'train' and os.path.exists(exp_path) and os.listdir(exp_path):
        if not args.force:
            raise FileExistsError(
                "{} is not empty. Please use --force to overwrite it".format(exp_path)
            )
        else:
            shutil.rmtree(exp_path)
            os.makedirs(exp_path)
    log_path = os.path.join(exp_path, 'logs')
    os.makedirs(log_path, exist_ok=True)
    return exp_path, log_path


def init_logger(path: str):
    formatter = logging.Formatter(
        "[%(asctime)s][%(name)s][%(levelname)s][%(filename)s>%(funcName)s] ==> %(message)s"
    )
    for mode in ['train', 'test', 'train_test', 'root']:
        file_path = os.path.join(path, mode + '.log')
        logger = logging.getLogger(mode)
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        handler = logging.FileHandler(file_path, mode='a')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
