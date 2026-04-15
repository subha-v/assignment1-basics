from cs336_basics.modal_utils import VOLUME_MOUNTS, app, build_image, secrets
from tokenizer.tokenize_data import tokenize_data_parallel


@app.function(
    image=build_image(),
    cpu=16,
    volumes=VOLUME_MOUNTS,
    secrets=secrets(),
    timeout=60 * 60 * 24,
)
def tokenize_remote(file_to_tokenize, vocab_path, merges_path, output_path, num_workers=16):
    tokenize_data_parallel(file_to_tokenize, vocab_path, merges_path, output_path, num_workers)


@app.local_entrypoint()
def modal_main(dataset: str = "tinystories"):
    if dataset == "owt":
        train_input = "data/owt_train.txt"
        val_input = "data/owt_valid.txt"
        vocab_path = "data/owt_vocab.json"
        merges_path = "data/owt_merges.txt"
        train_output = "data/owt_train.bin"
        val_output = "data/owt_val.bin"
    else:
        train_input = "data/TinyStoriesV2-GPT4-train.txt"
        val_input = "data/TinyStoriesV2-GPT4-valid.txt"
        vocab_path = "data/ts_vocab.json"
        merges_path = "data/ts_merges.txt"
        train_output = "data/tinystories_train.bin"
        val_output = "data/tinystories_val.bin"

    # train first then val!
    # tokenize_remote.remote(train_input, vocab_path, merges_path, train_output)
    tokenize_remote.remote(val_input, vocab_path, merges_path, val_output)
