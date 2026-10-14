# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "transformers>=5.2"]
#
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
#
# [tool.uv.sources]
# torch = { index = "pytorch-cpu" }
# ///
import json
import re
from pathlib import Path

import torch
import torch.nn.functional as F
import transformers
from transformers import Qwen3Config, Qwen3ForCausalLM

SEED = 20261013
COMMAND = "uv run scripts/gen_qwen3_fixture.py"
OUT = Path(__file__).resolve().parent.parent / "test" / "fixtures" / "torch" / "qwen3.json"

DIMS = {"V": 5, "d_model": 4, "d_head": 4, "d_ff": 6, "num_heads": 2, "num_kv_heads": 1, "num_layers": 2}
ROPE_THETA = 10.0
SEQUENCE = [2, 4, 3, 0]


def float64_everywhere():
    to = torch.Tensor.to

    def to64(self, *args, **kwargs):
        args = tuple(torch.float64 if a is torch.float32 else a for a in args)
        if kwargs.get("dtype") is torch.float32:
            kwargs["dtype"] = torch.float64
        return to(self, *args, **kwargs)

    softmax = F.softmax

    def softmax64(*args, **kwargs):
        if kwargs.get("dtype") is torch.float32:
            kwargs["dtype"] = torch.float64
        return softmax(*args, **kwargs)

    torch.Tensor.to = to64
    torch.Tensor.float = torch.Tensor.double
    F.softmax = softmax64
    torch.nn.functional.softmax = softmax64


def config():
    return Qwen3Config(
        vocab_size=DIMS["V"],
        hidden_size=DIMS["d_model"],
        intermediate_size=DIMS["d_ff"],
        num_hidden_layers=DIMS["num_layers"],
        num_attention_heads=DIMS["num_heads"],
        num_key_value_heads=DIMS["num_kv_heads"],
        head_dim=DIMS["d_head"],
        rope_parameters={"rope_type": "default", "rope_theta": ROPE_THETA},
        rms_norm_eps=1e-6,
        tie_word_embeddings=False,
        attn_implementation="eager",
        dtype=torch.float64,
    )


def page_path(key):
    return re.sub(r"\.(\d+)\.", r"[\1].", key)


def fill(params, generator):
    with torch.no_grad():
        for path, p in params.items():
            if path.endswith("norm.weight"):
                p.copy_(1 + torch.randn(p.shape, generator=generator) * 0.1)
            else:
                p.copy_(torch.randn(p.shape, generator=generator) * 0.6)


def encode(t):
    return {"shape": list(t.shape), "values": [float(v) for v in t.detach().reshape(-1)]}


def main():
    torch.set_default_dtype(torch.float64)
    float64_everywhere()
    generator = torch.Generator().manual_seed(SEED)
    model = Qwen3ForCausalLM(config()).double().eval()
    params = {page_path(key): p for key, p in model.named_parameters()}
    fill(params, generator)
    tokens = torch.tensor([SEQUENCE])
    output = model(input_ids=tokens, labels=tokens)
    output.loss.backward()
    loss = output.loss
    logits = output.logits[0, :-1]
    payload = {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "command": COMMAND,
        "seed": SEED,
        "sequence": SEQUENCE,
        "weights": {path: encode(p) for path, p in params.items()},
        "loss": loss.item(),
        "logits": encode(logits),
        "grads": {path: encode(p.grad) for path, p in params.items()},
    }
    OUT.write_text(json.dumps(payload, indent=1) + "\n")
    print(f"{OUT.name}: loss {loss.item():.6f}")


if __name__ == "__main__":
    main()
