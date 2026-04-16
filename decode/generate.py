import torch
import yaml

from decode.decode import decode
from tokenizer.tokenizer import Tokenizer
from transformer.transformer import EntireTransformer


config_path = "training/configs/baseline.yaml"
checkpoint_path = "checkpoints/bs_256/ckpt_final.pt"
vocab_path = "data/ts_vocab.json"
merges_path = "data/ts_merges.txt"

prompt = "Kaitlyn was playing clash royale and then"
max_new_tokens = 256
temperature = 0.8
top_p = 0.9
device = "cpu"
seed = 0

end_of_doc = "<|endoftext|>"


with open(config_path) as f:
    cfg = yaml.safe_load(f)

m_cfg = cfg["model"]

torch.manual_seed(seed)

tokenizer = Tokenizer.from_files(vocab_path, merges_path, special_tokens=[end_of_doc])
eot_id = tokenizer.invertedVocab[end_of_doc.encode("utf-8")]

model = EntireTransformer(
    vocab_size=m_cfg["vocab_size"],
    context_length=m_cfg["context_length"],
    num_layers=m_cfg["num_layers"],
    d_model=m_cfg["d_model"],
    num_heads=m_cfg["num_heads"],
    d_ff=m_cfg["d_ff"],
    theta=m_cfg["rope_theta"],
    device=device,
    dtype=torch.float32,
)

ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
model.load_state_dict(ckpt["model"])
model.context_length = m_cfg["context_length"]

generated = decode(
    prompt=prompt,
    model=model,
    tokenizer=tokenizer,
    eot_id=eot_id,
    max_new_tokens=max_new_tokens,
    temperature=temperature,
    top_p=top_p,
    device=device,
)

print(prompt + generated)
