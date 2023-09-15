import platform

import torch.cuda
from torch.nn import L1Loss


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


def get_os() -> str:
    return platform.system()


def get_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    return (param_size + buffer_size) / 1024 ** 2


def RMSELoss(yhat, y):
    return torch.sqrt(torch.mean((yhat - y) ** 2))


class MAE(L1Loss):
    def __init__(self):
        super().__init__()
        pass
