import torch
import torch.nn as nn
import math
from typing import Optional
from collections.abc import Iterable
from einops import rearrange, einsum
from transformer.linear import Linear
from transformer.embedding import Embedding

# Cross Entropy Loss 4.1

def cross_entropy_loss(predicted_logits: torch.Tensor, targets: torch.Tensor):
    max_logits = torch.max(predicted_logits, dim=-1, keepdim=True).values
    s = predicted_logits - max_logits
    log_sum_exp = torch.log(torch.sum(torch.exp(s), dim=-1)) + max_logits.squeeze(-1)
    target_logits = torch.gather(predicted_logits, -1, targets.unsqueeze(-1)).squeeze(-1)
    return (log_sum_exp - target_logits).mean()


# 4.3 AdamW

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)


    def step(self, closure=None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad.data
                state = self.state[p]
                if len(state) == 0:
                    state["t"] = 0
                    state["m"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)
                
                t = state["t"]
                m = state["m"]
                v = state["v"]
                t = t+1
                a_t = lr * math.sqrt(1 - beta2**t)/(1-beta1**t)
                m = beta1 * m + (1- beta1) * g
                v = beta2 * v + (1-beta2) * g**2
                p.data -= lr * weight_decay * p.data

                p.data -= a_t * m / (torch.sqrt(v) + eps)

                state["t"] = t
                state["m"] = m
                state["v"] = v

        return loss
            

# 4.4 Cosine Learning Rate Scheduling
def cosine_lr_scheduling(t, a_max, a_min, T_w, T_c):
    if(t < T_w):
        a_t = (t* a_max) /T_w 
    elif(t >= T_w and t <= T_c):
        a_t = a_min + 0.5 * (1 + math.cos((t-T_w) * math.pi / (T_c - T_w) )) * (a_max - a_min)
    else:
        a_t = a_min
    return a_t

# Gradient Clipping 
def gradient_clip(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, eps = 1e-6):
    parameters = list(parameters)
    total_sqrt = 0.0
    for p in parameters:
        if p.grad is None:
            continue
        total_sqrt += p.grad.data.pow(2).sum()
    
    total_norm = torch.sqrt(total_sqrt)

    if(total_norm >= max_l2_norm):
        for p in parameters:
            if p.grad is None:
                continue
            p.grad.data *= (max_l2_norm / (total_norm + eps))
    
    return