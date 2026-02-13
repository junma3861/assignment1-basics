import torch
import torch.nn as nn

from cs336_basics.multihead_self_attention import MultiHeadSelfAttention
from cs336_basics.rmsnorm import RMSNorm
from cs336_basics.positionwise_feedforward import PositionwiseFeedForward

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, use_rope=True, theta=10000.0, max_seq_len=2048):
        super(TransformerBlock, self).__init__()
        self.self_attention = MultiHeadSelfAttention(d_model, num_heads, use_rope=use_rope, theta=theta, max_seq_len=max_seq_len)
        self.layer_norm1 = RMSNorm(d_model)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff)
        self.layer_norm2 = RMSNorm(d_model)

    def forward(self, x, token_positions=None):
        # Pre-norm architecture: apply layer norm before the operation
        # Self-attention with residual connection
        attn_output = self.self_attention(self.layer_norm1(x), token_positions=token_positions)
        x = x + attn_output  # Residual connection
        
        # Feed-forward network with residual connection
        ff_output = self.feed_forward(self.layer_norm2(x))
        x = x + ff_output  # Residual connection
        
        return x