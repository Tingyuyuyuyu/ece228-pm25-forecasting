"""Small helpers: seeding, device pick, logging."""
import logging
import os
import random
import sys

import numpy as np

try:
    import torch
except ImportError:
    torch = None


def set_seed(seed):
    """Seed Python, NumPy, and PyTorch RNGs together."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def get_device():
    """Pick the best device available: CUDA > MPS (Apple Silicon) > CPU."""
    if torch is None:
        return "cpu"
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_logger(name, log_file=None):
    """Return a logger that writes to stdout (and optionally to a file).
    If the named logger already has handlers we just return it -- this
    keeps re-creation safe across the main script and submodules."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if log_file is not None:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    logger.propagate = False
    return logger
