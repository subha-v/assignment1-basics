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
    batches: str = "8,32,64,128,256,512,1024",
):
    batch_values = [int(x) for x in batches.split(",")]

    with open(config) as f:
        base_cfg = yaml.safe_load(f)

    ref_batch = base_cfg["training"]["batch_size"]
    ref_iters = base_cfg["training"]["total_iters"]
    ref_warmup = base_cfg["optimizer"]["warmup_iters"]
    ref_cosine = base_cfg["optimizer"]["cosine_cycle_iters"]

    cfg_list = []
    for bs in batch_values:
        with open(config) as f:
            cfg = yaml.safe_load(f)

        scale = ref_batch / bs
        total_iters = int(ref_iters * scale)
        warmup_iters = int(ref_warmup * scale)
        cosine_iters = int(ref_cosine * scale)

        cfg["training"]["batch_size"] = bs
        cfg["training"]["total_iters"] = total_iters
        cfg["training"]["checkpoint_interval"] = total_iters
        cfg["training"]["eval_interval"] = max(1, total_iters // 20)
        cfg["optimizer"]["warmup_iters"] = warmup_iters
        cfg["optimizer"]["cosine_cycle_iters"] = cosine_iters

        cfg["logging"]["wandb_run_name"] = f"bs_{bs}"
        cfg["logging"]["checkpoint_dir"] = f"data/checkpoints/bs_{bs}"
        cfg_list.append(cfg)

    for _ in train_remote.map(cfg_list):
        pass
