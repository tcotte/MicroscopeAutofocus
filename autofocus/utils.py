import torch.cuda


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"