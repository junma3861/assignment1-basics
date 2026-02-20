import torch
import torch.nn as nn
from collections.abc import Iterable

def clip_gradients(parameters: Iterable[torch.nn.Parameter], max_norm: float):
    """
    Clips the gradients of the given parameters to have a maximum norm of `max_norm`.

    Args:
        parameters (Iterable[torch.nn.Parameter]): The parameters whose gradients need to be clipped.
        max_norm (float): The maximum allowed norm for the gradients.

    Returns:
        None
    """
    total_norm = 0.0
    for param in parameters:
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5

    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for param in parameters:
            if param.grad is not None:
                param.grad.data.mul_(clip_coef)