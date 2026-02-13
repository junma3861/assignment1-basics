import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super(RMSNorm, self).__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        # Compute RMS: sqrt(mean(x^2) + eps)
        rms = torch.sqrt((x ** 2).mean(dim=-1, keepdim=True) + self.eps)
        return (x / rms * self.weight).to(in_dtype)