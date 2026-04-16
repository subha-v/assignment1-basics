import yaml

from cs336_basics.modal_utils import VOLUME_MOUNTS, app, build_image, secrets
from training.train import training_loop


@app.function(
    image=build_image(),
    gpu="H100",
    volumes=VOLUME_MOUNTS,
    secrets=secrets(),
    timeout=60 * 60 * 4,
    max_containers=3,
)
def train_remote(cfg: dict):
    training_loop(cfg)


@app.local_entrypoint()
def modal_main(
    config: str = "training/configs/baseline.yaml",
    lr: float = 3e-4,
):
    cfg_list = []

    with open(config) as f:
        baseline_cfg = yaml.safe_load(f)
    baseline_cfg["optimizer"]["lr_max"] = lr
    baseline_cfg["optimizer"]["lr_min"] = lr / 10
    baseline_cfg["logging"]["wandb_run_name"] = "swiglu_baseline"
    baseline_cfg["logging"]["checkpoint_dir"] = "data/checkpoints/swiglu_baseline"
    cfg_list.append(baseline_cfg)

    with open(config) as f:
        silu_cfg = yaml.safe_load(f)
    silu_cfg["model"]["use_swiglu"] = False
    silu_cfg["model"]["d_ff"] = 2048
    silu_cfg["optimizer"]["lr_max"] = lr
    silu_cfg["optimizer"]["lr_min"] = lr / 10
    silu_cfg["logging"]["wandb_run_name"] = "silu_ablation"
    silu_cfg["logging"]["checkpoint_dir"] = "data/checkpoints/silu_ablation"
    cfg_list.append(silu_cfg)

    for _ in train_remote.map(cfg_list):
        pass
