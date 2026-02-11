import torch
import torch.nn as nn

class RotaryPositionEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super(RotaryPositionEmbedding, self).__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        position_ids = token_positions[:, :seq_len]
        freqs = self.theta ** (torch.arange(0, self.d_k, 2, device=self.device) / self.d_k)
        freqs = position_ids[:, :, None] * freqs[None, None, :]
        cos_emb = torch.cos(freqs)
        sin_emb = torch.sin(freqs)
        emb = torch.zeros_like(x)
        emb[:, :, ::2] = cos_emb
        emb[:, :, 1::2] = sin_emb
        return x * emb

    