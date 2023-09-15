import platform

import torch.cuda


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


def get_os() -> str:
    return platform.system()
