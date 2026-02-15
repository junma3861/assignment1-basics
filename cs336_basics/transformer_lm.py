import torch
import torch.nn as nn

from cs336_basics.transformer_block import TransformerBlock
from cs336_basics.rmsnorm import RMSNorm


class TransformerLM(nn.Module):
    def __init__(self, vocab_size, context_length, d_model, num_heads, d_ff, num_layers, rope_theta=10000.0):
        super(TransformerLM, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, use_rope=True, theta=rope_theta, max_seq_len=context_length)
            for _ in range(num_layers)
        ])
        self.layer_norm_final = RMSNorm(d_model)
        self.output_linear = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids):
        x = self.token_embedding(input_ids)  # (batch_size, seq_length, d_model)
        
        for layer in self.layers:
            x = layer(x)  # (batch_size, seq_length, d_model)

        # Apply final layer normalization
        x = self.layer_norm_final(x)
        
        logits = self.output_linear(x)  # (batch_size, seq_length, vocab_size)
        return logits