import torch
import torch.nn as nn
import math
from typing import Optional
from einops import rearrange, einsum
from transformer.linear import Linear

# Implementing section 3.4

# 3.4.1 - RMS Layer Normalization
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.g = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    # process an input tensor of shape (batch_size, sequence_length, d_model)
    # return a tensor of the same shape
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        result = (x/rms) * self.g # element wise
        return result.to(in_dtype)

# 3.4.2 position Wise FFN
# Note that the nn module implements call which internally calls self.forward(x) so its the same

class FFN(nn.Module):
    # you should set d_ff to approximately 8/3 * d_model
    def __init__(self, d_model, d_ff, device=None, dtype=None):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device =device, dtype = dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x):
        w1Output = self.w1(x)
        siluActivation = w1Output * torch.sigmoid(w1Output)
        w3Output = self.w3(x)
        gatedOutput = siluActivation * w3Output
        return self.w2(gatedOutput)

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len:int, device=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device
        frequencies = 1.0 / (theta ** (torch.arange(0, d_k, 2, device=device).float() / d_k))
        positions = torch.arange(max_seq_len, device=device).float()
        # with angles, row i corresponds to position i and column k corrseponds to pair k
        angles = positions.unsqueeze(1) * frequencies.unsqueeze(0)
        self.register_buffer('cos', torch.cos(angles), persistent = False)
        self.register_buffer('sin', torch.sin(angles), persistent = True)

    """
    Process an input tensor of shape (..., seq_len, d_k) and return a tensor of the same shape.
    We should tolerate x with an arbitrary number of batch dimensions
    Assume the token positions are a tensor of shape (..., seq_len)
    """
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # Basically we want to rotate each token of x
        # Loop through everything before seq_len
        # Then group through pairs based on d_k
        # Rotate each pair based on angles[i, pair] where pair+=1 after each loop
        # this tells us that the last dimension is the product of p1 and p2
        # This tells us that the second dimension p2 must be 2
        # Essentially we are telling them the last dimension is a tensor of pairs so that's why we're explicitly
        # saying '2' here and then we tell them what d_k_half is in terms of d_k
        x_split = rearrange(x, "... (d_k_half two) -> ... d_k_half two", d_k_half=self.d_k // 2, two=2)
        cosine = self.cos[token_positions]
        sine = self.sin[token_positions]
        x1 = x_split[..., 0]
        x2 = x_split[..., 1]

        out1 = x1 * cosine - x2 * sine
        out2 = x1 * sine + x2 * cosine
        stacked = torch.stack([out1, out2], dim=-1)
        # by putting the parentheses around we tell them that those are merged together
        result = rearrange(stacked, "... d_k_half two -> ... (d_k_half two)", two=2)
        return result
    
def softmax(x, dim=-1):
    max_val = torch.amax(x, dim=dim, keepdim = True)
    total_sum = torch.sum(torch.exp(x - max_val), dim=dim, keepdim = True)
    return torch.exp(x - max_val) / total_sum

# 3.4.4 Scaled Dot Prod Attention
def scaled_dot_product_attention(q: torch.Tensor, k: torch.tensor, v: torch.tensor, mask: Optional[torch.Tensor]=None) -> torch.Tensor:
    qk = einsum(q, k, "... seq_q d_k, ... seq_k d_k -> ... seq_q seq_k")
    d_k = q.size(-1)
    inner = qk / math.sqrt(d_k)
    # The -floatinf already applies to the actual values here
    masked_inner = torch.where(mask, inner, -float('inf'))
    attention_weights = softmax(masked_inner, dim=-1)
    final_prod = einsum(attention_weights, v, "... seq_q seq_k, ... seq_k d_v -> ... seq_q d_v")
    return final_prod

# 3.4.5 Causal Multi Head Self Attention

class CausalMHA(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_seq_len=None, theta=None, device=None, dtype=None):
        super().__init__()

        self.d_model = d_model
    
        self.d_k = d_model // num_heads

        self.num_heads = num_heads
        # Input to the linear projection is a vector of size d_model
        # Output of the linear projection is a vector of size d_k
        self.Wq = Linear(self.d_model, num_heads * self.d_k, device=device, dtype=dtype)
        self.Wk = Linear(self.d_model, num_heads * self.d_k, device=device, dtype=dtype)
        self.Wv = Linear(self.d_model, num_heads * self.d_k, device=device, dtype=dtype)
        self.Wo = Linear(num_heads * self.d_k, self.d_model, device=device, dtype=dtype)
        
        # RoPE
        if max_seq_len is not None and theta is not None:
            self.rope = RotaryPositionalEmbedding(theta = theta, d_k = self.d_k, max_seq_len = max_seq_len, device=device)
        else:
            self.rope=None


    # tensor x is passing through the model! :D
    def forward(self, x: torch.Tensor, token_pos=None):
        seq =x.shape[-2]
        # Forward pass thru
        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)

        # Then we want to split this up by head
        # Note that we swap num_heads to be before so it can be a batch dimension since the last two dimensions for 
        # our attention computation should always be seq, d_k and num_heads should be a batch dim
        Q = rearrange(Q, "... seq (num_heads d_k) -> ... num_heads seq d_k", num_heads = self.num_heads)
        V = rearrange(V, "... seq (num_heads d_k) -> ... num_heads seq d_k", num_heads = self.num_heads)
        K = rearrange(K, "... seq (num_heads d_k) -> ... num_heads seq d_k", num_heads = self.num_heads)

        # Now we have to create a mask which is going to be a diagonal one
        mask = ~torch.triu(torch.ones(seq, seq, dtype=torch.bool, device=x.device), diagonal=1)

        # Apply RoPE
        # Token position just 0 to seq_len -1
        
        if self.rope is not None:
            if token_pos is None:
                token_pos = torch.arange(seq, device=x.device)
            Q = self.rope(Q, token_pos)
            K = self.rope(K, token_pos)

        output = scaled_dot_product_attention(Q, K, V, mask)

        # Merge all the heads back together
        output = rearrange(output, "... num_heads seq d_k -> ... seq (num_heads d_k)")

        output = self.Wo(output)

        return output
        



