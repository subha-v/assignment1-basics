import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
import wandb
import yaml

from transformer.transformer import EntireTransformer
from training.loss import (
    AdamW,
    cosine_lr_scheduling,
    cross_entropy_loss,
    gradient_clip,
)

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
def training_loop(cfg):
    # Basically we want to publish to wandb now
    m_cfg = cfg["model"]
    o_cfg = cfg["optimizer"]
    t_cfg = cfg["training"]
    d_cfg = cfg["data"]
    l_cfg = cfg["logging"]
    s_cfg = cfg["system"]

    torch.manual_seed(s_cfg["seed"])
    np.random.seed(s_cfg["seed"])
    random.seed(s_cfg["seed"])

    train_data = np.memmap(d_cfg["train_path"], dtype=d_cfg["dtype"], mode="r")
    val_data = np.memmap(d_cfg["val_path"], dtype=d_cfg["dtype"], mode="r")

    Path(l_cfg["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)

    dtypeMap = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
    device = s_cfg["device"]
    dtype = dtypeMap[s_cfg["dtype"]]

    model = EntireTransformer(
        vocab_size=m_cfg["vocab_size"],
        context_length=m_cfg["context_length"],
        num_layers=m_cfg["num_layers"],
        d_model=m_cfg["d_model"],
        num_heads=m_cfg["num_heads"],
        d_ff=m_cfg["d_ff"],
        theta=m_cfg["rope_theta"],
        use_rms_norm = m_cfg.get("use_rms_norm", True),
        use_swiglu = m_cfg.get("use_swiglu", True),
        pre_norm=m_cfg.get("pre_norm", True),
        use_rope=m_cfg.get("use_rope", True),
        device=device,
        dtype=dtype,
    )

    optimizer = AdamW(
        model.parameters(),
        lr=o_cfg["lr_max"],
        betas=tuple(o_cfg["betas"]),
        eps=o_cfg["eps"],
        weight_decay=o_cfg["weight_decay"],
    )

    startIter = 0
    if s_cfg.get("resume_from") is not None:
        startIter = load_checkpoint(s_cfg["resume_from"], model, optimizer)

    wandb.init(
        project=l_cfg["wandb_project"],
        entity=l_cfg["wandb_entity"],
        name=l_cfg["wandb_run_name"],
        config=cfg,
    )

    batch_size = t_cfg["batch_size"]
    context_length = m_cfg["context_length"]
    total_iters = t_cfg["total_iters"]
    log_interval = t_cfg["log_interval"]
    eval_interval = t_cfg["eval_interval"]
    eval_iters = t_cfg["eval_iters"]
    checkpoint_interval = t_cfg["checkpoint_interval"]
    checkpoint_dir = Path(l_cfg["checkpoint_dir"])

    lr_max = o_cfg["lr_max"]
    lr_min = o_cfg["lr_min"]
    warmup_iters = o_cfg["warmup_iters"]
    cosine_cycle_iters = o_cfg["cosine_cycle_iters"]
    grad_clip = o_cfg["grad_clip"]

    startTime = time.time()

    model.train()
    for iteration in range(startIter, total_iters):
        lr = cosine_lr_scheduling(iteration, lr_max, lr_min, warmup_iters, cosine_cycle_iters)
        for g in optimizer.param_groups:
            g["lr"] = lr

        inputs, targets = data_loader(train_data, batch_size, context_length, device)
        logits = model(inputs)
        loss = cross_entropy_loss(logits, targets)

        optimizer.zero_grad()
        loss.backward()
        gradient_clip(model.parameters(), grad_clip)
        optimizer.step()

        if iteration % log_interval == 0:
            wandb.log(
                {"train/loss": loss.item(), "train/lr": lr, "wall_time": time.time() - startTime},
                step=iteration,
            )

        if iteration > 0 and iteration % eval_interval == 0:
            model.eval()
            valLossSum = 0.0
            with torch.no_grad():
                for _ in range(eval_iters):
                    valInputs, valTargets = data_loader(val_data, batch_size, context_length, device)
                    valLogits = model(valInputs)
                    valLossSum += cross_entropy_loss(valLogits, valTargets).item()
            avgValLoss = valLossSum / eval_iters
            wandb.log({"val/loss": avgValLoss, "wall_time": time.time() - startTime}, step=iteration)
            model.train()

        if iteration > 0 and iteration % checkpoint_interval == 0:
            save_checkpoint(model, optimizer, iteration, checkpoint_dir / f"ckpt_{iteration}.pt")

    save_checkpoint(model, optimizer, total_iters, checkpoint_dir / "ckpt_final.pt")
    wandb.finish()


def load_config():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("overrides", nargs="*",
                   help="dotted overrides, e.g. training.batch_size=128")
    args = p.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    for ov in args.overrides:
        key, val = ov.split("=", 1)
        d = cfg
        *parts, last = key.split(".")
        for k in parts:
            d = d[k]
        d[last] = yaml.safe_load(val)
    return cfg


if __name__ == "__main__":
    training_loop(load_config())

