import torch
import torch.nn as nn

from cs336_basics.softmax import softmax

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """
    Compute the scaled dot-product attention.

    Args:
        Q: Query tensor of shape (batch_size, num_heads, seq_length_q, head_dim)
        K: Key tensor of shape (batch_size, num_heads, seq_length_k, head_dim)
        V: Value tensor of shape (batch_size, num_heads, seq_length_v, head_dim)
        mask: Optional mask tensor of shape (batch_size, num_heads, seq_length_q, seq_length_k)

    Returns:
        Output tensor of shape (batch_size, num_heads, seq_length_q, head_dim)
    """
    # Step 1: Compute the dot product between Q and K^T
    d_k = Q.size(-1)  # head_dim
    scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))

    # Step 2: Apply the mask (if provided)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    # Step 3: Apply softmax to get the attention weights
    attn_weights = softmax(scores, dim=-1)

    # Step 4: Compute the output by multiplying the attention weights with V
    output = torch.matmul(attn_weights, V)

    return output