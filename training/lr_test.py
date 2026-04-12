import torch
import math
from collections.abc import Callable
from typing import Optional

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] 
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p] 
                t = state.get("t", 0) 
                grad = p.grad.data 
                p.data -= lr / math.sqrt(t + 1) * grad 
                state["t"] = t + 1 
        return loss

# Initialize some random weights to optimize
weights = torch.nn.Parameter(5 * torch.randn((10, 10)))

# The assignment asks you to try: 1e1, 1e2, and 1e3
current_lr = 1e3
opt = SGD([weights], lr=current_lr)

print(f"--- Running SGD with learning rate: {current_lr} ---")

# Run for 10 iterations
for t in range(10):
    opt.zero_grad()               # Reset the gradients
    loss = (weights**2).mean()    # Compute a scalar loss value
    
    # Print the loss so you can see if it decays or diverges!
    print(f"Iteration {t}: Loss = {loss.item():.4f}")
    
    loss.backward()               # Run backward pass (computes gradients)
    opt.step()                    # Run optimizer step