import torch
import torch.nn as nn

from cs336_basics.scaled_dot_product_attention import scaled_dot_product_attention
from cs336_basics.rope import RotaryPositionEmbedding

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, use_rope=False, theta=10000.0, max_seq_len=2048):
        super(MultiHeadSelfAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.use_rope = use_rope
        self.theta = theta
        self.max_seq_len = max_seq_len
        
        assert self.head_dim * num_heads == d_model, "d_model must be divisible by num_heads"
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        
        if self.use_rope:
            self.rope = RotaryPositionEmbedding(self.head_dim, theta=theta, max_seq_len=max_seq_len)

    def forward(self, x, token_positions=None):
        batch_size, seq_length, _ = x.size()
        
        # Linear projections
        Q = self.q_linear(x)  # (batch_size, seq_length, d_model)
        K = self.k_linear(x)  # (batch_size, seq_length, d_model)
        V = self.v_linear(x)  # (batch_size, seq_length, d_model)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, seq_length, head_dim)
        K = K.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, seq_length, head_dim)
        V = V.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, seq_length, head_dim)
        
        # Apply rotary positional embeddings
        if self.use_rope:
            if token_positions is None:
                token_positions = torch.arange(seq_length, device=x.device).unsqueeze(0).expand(batch_size, -1)  # (batch_size, seq_length)
            Q = self.rope(Q, token_positions)  # (batch_size, num_heads, seq_length, head_dim)
            K = self.rope(K, token_positions)  # (batch_size, num_heads, seq_length, head_dim)
        
        # Apply causal mask to prevent attending to future tokens
        # Lower triangular mask: 1 for positions to keep, 0 for positions to mask
        mask = torch.tril(torch.ones(seq_length, seq_length, device=x.device)).bool()  # (seq_length, seq_length)

        # Scaled dot-product attention
        attn_weights = scaled_dot_product_attention(Q, K, V, mask=mask)  # (batch_size, num_heads, seq_length, head_dim)
        
        # Concatenate heads and pass through final linear layer
        attn_output = attn_weights.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)  # (batch_size, seq_length, d_model)

        output = self.out_linear(attn_output)  # (batch_size, seq_length, d_model)
        return output
    
