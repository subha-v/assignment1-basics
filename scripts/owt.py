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
    lrs: str = "1e-4,3e-4,1e-3",
):
    lr_values = [float(x) for x in lrs.split(",")]

    cfg_list = []
    for lr in lr_values:
        with open(config) as f:
            cfg = yaml.safe_load(f)

        cfg["model"]["vocab_size"] = 32000
        cfg["data"]["train_path"] = "data/owt_train.bin"
        cfg["data"]["val_path"] = "data/owt_val.bin"
        cfg["optimizer"]["lr_max"] = lr
        cfg["optimizer"]["lr_min"] = lr / 10
        run_name = f"owt_lr_{lr:.0e}"
        cfg["logging"]["wandb_run_name"] = run_name
        cfg["logging"]["checkpoint_dir"] = f"data/checkpoints/{run_name}"
        cfg_list.append(cfg)

    for _ in train_remote.map(cfg_list):
        pass
