import torch
import torch.nn as nn

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Compute the softmax of the input tensor `x` along the specified dimension `dim`.

    Args:
        x (torch.Tensor): The input tensor.
        dim (int): The dimension along which to compute the softmax.

    Returns:
        torch.Tensor: The softmax of the input tensor along dimension `dim`.
    """
    # Subtract the maximum value along dimension dim for numerical stability
    max_x = torch.max(x, dim=dim, keepdim=True).values
    x_exp = torch.exp(x - max_x)
    
    # Sum of exponentials along dimension dim
    sum_x_exp = torch.sum(x_exp, dim=dim, keepdim=True)
    
    # Compute softmax
    softmax_x = x_exp / sum_x_exp
    
    return softmax_x