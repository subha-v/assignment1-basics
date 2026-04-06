import torch
import torch.nn as nn
from einops import rearrange, einsum

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        tensor = torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        torch.nn.init.trunc_normal_(tensor, mean=0.0, std=1.0, a=-3.0, b=3.0)
        self.embedding_weights = nn.Parameter(tensor)

    # Look up the embedding vectors for the given token IDs
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids shape is batch_size, sequence_lenght
        # self.weight shape is vocab_size, d_model

        # since token_ids is a tensor full of integers it realizes its indices
        result = self.embedding_weights[token_ids]
        return result
