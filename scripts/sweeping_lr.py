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
def modal_main(config: str = "training/configs/baseline.yaml", lrs: str = "1e-4,3e-4,1e-3,3e-3,1e-2"):
    lrValues = [float(x) for x in lrs.split(",")]

    cfgList = []
    for lr in lrValues:
        with open(config) as f:
            cfg = yaml.safe_load(f)

        cfg["optimizer"]["lr_max"] = lr
        cfg["optimizer"]["lr_min"] = lr / 10
        cfg["logging"]["wandb_run_name"] = f"lr_{lr:.0e}"
        cfgList.append(cfg)

    for _ in train_remote.map(cfgList):
        pass
