import numpy as np
import torch
import random

def data_loader(x, batch_size, context_length, device_string) -> tuple[torch.Tensor, torch.Tensor]:
    max_index = len(x) - context_length
    random_indices = np.random.randint(0, max_index, size = batch_size)
    inputs = np.stack([x[s: s + context_length] for s in random_indices])
    outputs = np.stack([x[s + 1: s + 1 + context_length] for s in random_indices])
    inputs = torch.from_numpy(inputs).long().to(device_string)
    outputs = torch.from_numpy(outputs).long().to(device_string)
    return (inputs, outputs)
    

# 5.2 Checkpointing
def save_checkpoint(model, optimizer, iteration, out):
    # Returns a dict like mapping of param names to tensors
    model_dict = model.state_dict() 
    optimizer_dict = optimizer.state_dict()
    entire_dict = {"model": model_dict, "optimizer": optimizer_dict, "iteration": iteration}
    torch.save(entire_dict, out)

def load_checkpoint(src, model, optimizer):
    entire_dict = torch.load(src)
    model.load_state_dict(entire_dict["model"])
    optimizer.load_state_dict(entire_dict["optimizer"])
    iteration = entire_dict["iteration"]
    return iteration


# 5.3 Training Together
def training_loop()