import torch
import torch.nn as nn

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super(Embedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.Tensor(num_embeddings, embedding_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight[x]