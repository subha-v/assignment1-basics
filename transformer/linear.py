import torch
import torch.nn as nn
import math
from einops import rearrange, einsum


class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        # initializes the superclass constructor, helps us keep track of gradients etc
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        std = math.sqrt(2.0 / (in_features + out_features))
        tensor = torch.empty((out_features, in_features), device=device, dtype=dtype)
        torch.nn.init.trunc_normal_(tensor, mean=0.0, std=std, a=-3 * std, b=3 * std)
        self.weights = nn.Parameter(tensor)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # the ... ensures that we keep everything back in the output
        result = einsum(self.weights, x, "out_features in_features, ... in_features -> ... out_features")
        return result