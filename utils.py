import os
import torch
from datetime import datetime

def sefup_derectories(config):
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    print(f'[setup] Directories Created')



def get_logger(config):
    import logging
    log_file = os.path.join(config.log_dir, 'trainig.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s  - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()

        ]
    )
    return logging.getLogger(__name__)


def set_seed(seed=42):
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print(f'[setup] Random seed set to {seed}')