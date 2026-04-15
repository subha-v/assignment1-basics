from tokenizer.tokenizer import Tokenizer
from cs336_basics.pretokenization_example import find_chunk_boundaries
from multiprocessing import Pool, cpu_count
import numpy as np

# Each worker process needs its own tokenizer 
_worker_tokenizer = None


def init_worker(vocab_path, merges_path, special_tokens):
    global _worker_tokenizer
    _worker_tokenizer = Tokenizer.from_files(vocab_path, merges_path, special_tokens)


def encode_chunk(args):
    file_path, start_byte, end_byte = args
    with open(file_path, "rb") as f:
        f.seek(start_byte)
        raw_bytes = f.read(end_byte - start_byte)
    text = raw_bytes.decode("utf-8", errors="replace")
    return _worker_tokenizer.encode(text)


def tokenize_data_parallel(file_to_tokenize, vocab_path, merges_path, output_path, num_workers=None, special_tokens=None):
    if special_tokens is None:
        special_tokens = ["<|endoftext|>"]
    if num_workers is None:
        num_workers = cpu_count()

    # We split into more chunks than workers 
    num_chunks = num_workers * 4

 
    with open(file_to_tokenize, "rb") as f:
        # This is from the pretokenization example, finding boundaries
        boundaries = find_chunk_boundaries(f, num_chunks, special_tokens[0].encode("utf-8"))

    chunk_args = []
    for i in range(len(boundaries) - 1):
        chunk_args.append((file_to_tokenize, boundaries[i], boundaries[i + 1]))

    print(f"Tokenizing {file_to_tokenize} with {num_workers} workers across {len(chunk_args)} chunks", flush=True)

    total_tokens = 0
    chunks_done = 0
    with open(output_path, "wb") as f_out:
        with Pool(num_workers, initializer=init_worker, initargs=(vocab_path, merges_path, special_tokens)) as pool:
            # imap preserves order
            for token_ids in pool.imap(encode_chunk, chunk_args):
                np.asarray(token_ids, dtype=np.uint16).tofile(f_out)
                chunks_done += 1
                total_tokens += len(token_ids)
                print(f"Chunk {chunks_done}/{len(chunk_args)} done, total tokens so far: {total_tokens}", flush=True)

    print(f"Done! Wrote {total_tokens} tokens to {output_path}", flush=True)
    return total_tokens


if __name__ == "__main__":
    input_path = "data/TinyStoriesV2-GPT4-train.txt"
    vocab_path = "data/ts_vocab.json"
    merges_path = "data/ts_merges.txt"
    output_path = "data/tinystories_train.bin"
    tokenize_data_parallel(input_path, vocab_path, merges_path, output_path)