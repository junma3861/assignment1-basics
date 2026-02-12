import torch
import torch.nn as nn

class RotaryPositionEmbedding(nn.Module):
    def __init__(self, d_k: int, theta: float, max_seq_len: int, device=None):
        super(RotaryPositionEmbedding, self).__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # x shape: (..., seq_len, d_k)
        seq_len = x.size(-2)
        
        # Handle both 1D and 2D token_positions
        if token_positions.dim() == 1:
            # token_positions shape: (seq_len,)
            position_ids = token_positions[:seq_len]
        else:
            # token_positions shape: (batch, seq_len)
            position_ids = token_positions[..., :seq_len]
        
        # Compute frequencies: theta^(2i/d_k) for i in [0, d_k/2)
        # This gives us 1/frequencies that decrease exponentially
        freqs = 1.0 / (self.theta ** (torch.arange(0, self.d_k, 2, device=self.device, dtype=torch.float32) / self.d_k))
        
        # Compute position * frequency for each position and frequency
        # position_ids: (..., seq_len), freqs: (d_k/2,) -> (..., seq_len, d_k/2)
        position_freqs = position_ids.unsqueeze(-1).float() * freqs.unsqueeze(0)
        
        # Compute cos and sin
        cos_emb = torch.cos(position_freqs)
        sin_emb = torch.sin(position_freqs)
        
        # Apply RoPE: rotate pairs of dimensions
        # Split x into even and odd indices
        x_even = x[..., ::2]  # Shape: (..., seq_len, d_k/2)
        x_odd = x[..., 1::2]   # Shape: (..., seq_len, d_k/2)
        
        # Apply rotation matrix:
        # [cos -sin] [x_even]
        # [sin  cos] [x_odd ]
        x_rotated_even = x_even * cos_emb - x_odd * sin_emb
        x_rotated_odd = x_even * sin_emb + x_odd * cos_emb
        
        # Interleave the rotated even and odd components
        x_out = torch.zeros_like(x)
        x_out[..., ::2] = x_rotated_even
        x_out[..., 1::2] = x_rotated_odd
        
        return x_out

    