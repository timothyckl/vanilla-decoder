import torch


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_autocast_context(device):
    """Return torch.autocast kwargs for the given device, or a nullcontext if none apply."""
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if device.type == "mps":
        return torch.autocast(device_type="mps", dtype=torch.bfloat16)
    from contextlib import nullcontext

    return nullcontext()
