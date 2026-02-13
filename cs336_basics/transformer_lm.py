import torch
import torch.nn as nn

from cs336_basics.transformer_block import TransformerBlock


class TransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, use_rope=True, theta=10000.0, max_seq_len=2048):
        super(TransformerLM, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, use_rope=use_rope, theta=theta, max_seq_len=max_seq_len)
            for _ in range(num_layers)
        ])
        self.output_linear = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids):
        x = self.token_embedding(input_ids)  # (batch_size, seq_length, d_model)
        
        for layer in self.layers:
            x = layer(x)  # (batch_size, seq_length, d_model)
        
        logits = self.output_linear(x)  # (batch_size, seq_length, vocab_size)
        return logits