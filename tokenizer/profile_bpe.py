from tokenizer.bpe import bpe_tokenize

bpe_tokenize("tests/fixtures/tinystories_sample_5M.txt", 10000, ["<|endoftext|>"])
