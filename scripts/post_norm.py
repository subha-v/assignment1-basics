import yaml

from cs336_basics.modal_utils import VOLUME_MOUNTS, app, build_image, secrets
from training.train import training_loop


@app.function(
    image=build_image(),
    gpu="H100",
    volumes=VOLUME_MOUNTS,
    secrets=secrets(),
    timeout=60 * 60 * 4,
)
def train_remote(cfg: dict):
    training_loop(cfg)

# jst run the post norm
@app.local_entrypoint()
def modal_main(
    config: str = "training/configs/baseline.yaml",
    lr: float = 3e-3,
):
    with open(config) as f:
        cfg = yaml.safe_load(f)

    cfg["model"]["pre_norm"] = False
    cfg["optimizer"]["lr_max"] = lr
    cfg["optimizer"]["lr_min"] = lr / 10

    run_name = f"post_norm_lr_{lr:.0e}"
    cfg["logging"]["wandb_run_name"] = run_name
    cfg["logging"]["checkpoint_dir"] = f"data/checkpoints/{run_name}"

    train_remote.remote(cfg)
