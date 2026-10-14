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
from transformers import DeepseekV3Config, DeepseekV3ForCausalLM

SEED = 20261014
COMMAND = "uv run scripts/gen_deepseek_v3_fixture.py"
OUT = Path(__file__).resolve().parent.parent / "test" / "fixtures" / "torch" / "deepseek_v3.json"

DIMS = {
    "V": 5, "d_model": 4, "d_ff": 6, "d_moe": 2, "num_heads": 2, "q_lora_rank": 3, "kv_lora_rank": 3,
    "qk_nope_head_dim": 2, "qk_rope_head_dim": 4, "v_head_dim": 2, "n_routed_experts": 6, "n_group": 2,
    "topk_group": 1, "num_experts_per_tok": 2, "n_shared_experts": 1, "num_layers": 2, "first_k_dense_replace": 1,
}
ROPE_THETA = 10.0
ROUTED_SCALING_FACTOR = 2.5
SEQUENCE = [2, 4, 3, 0]


def float64_everywhere():
    to = torch.Tensor.to
    as_type = torch.Tensor.type

    def to64(self, *args, **kwargs):
        args = tuple(torch.float64 if a is torch.float32 else a for a in args)
        if kwargs.get("dtype") is torch.float32:
            kwargs["dtype"] = torch.float64
        return to(self, *args, **kwargs)

    def type64(self, *args, **kwargs):
        args = tuple(torch.float64 if a is torch.float32 else a for a in args)
        return as_type(self, *args, **kwargs)

    softmax = F.softmax

    def softmax64(*args, **kwargs):
        if kwargs.get("dtype") is torch.float32:
            kwargs["dtype"] = torch.float64
        return softmax(*args, **kwargs)

    torch.Tensor.to = to64
    torch.Tensor.type = type64
    torch.Tensor.float = torch.Tensor.double
    F.softmax = softmax64
    torch.nn.functional.softmax = softmax64


def config():
    return DeepseekV3Config(
        vocab_size=DIMS["V"],
        hidden_size=DIMS["d_model"],
        intermediate_size=DIMS["d_ff"],
        moe_intermediate_size=DIMS["d_moe"],
        num_hidden_layers=DIMS["num_layers"],
        first_k_dense_replace=DIMS["first_k_dense_replace"],
        num_attention_heads=DIMS["num_heads"],
        num_key_value_heads=DIMS["num_heads"],
        n_shared_experts=DIMS["n_shared_experts"],
        n_routed_experts=DIMS["n_routed_experts"],
        num_experts_per_tok=DIMS["num_experts_per_tok"],
        n_group=DIMS["n_group"],
        topk_group=DIMS["topk_group"],
        routed_scaling_factor=ROUTED_SCALING_FACTOR,
        norm_topk_prob=True,
        q_lora_rank=DIMS["q_lora_rank"],
        kv_lora_rank=DIMS["kv_lora_rank"],
        qk_nope_head_dim=DIMS["qk_nope_head_dim"],
        qk_rope_head_dim=DIMS["qk_rope_head_dim"],
        v_head_dim=DIMS["v_head_dim"],
        rope_parameters={"rope_type": "default", "rope_theta": ROPE_THETA},
        rope_interleave=True,
        rms_norm_eps=1e-6,
        tie_word_embeddings=False,
        attn_implementation="eager",
        experts_implementation="eager",
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


def routing_order(top_k_index, num_experts):
    expert_mask = F.one_hot(top_k_index, num_classes=num_experts + 1).permute(2, 1, 0)
    expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
    order = {"topk_indices": top_k_index.tolist(), "experts": {}}
    for expert_idx in expert_hit:
        expert_idx = expert_idx[0]
        if expert_idx == num_experts:
            continue
        _, token_idx = torch.where(expert_mask[expert_idx])
        order["experts"][int(expert_idx)] = [int(i) for i in token_idx]
    return order


def encode(t):
    return {"shape": list(t.shape), "values": [float(v) for v in t.detach().reshape(-1)]}


def main():
    torch.set_default_dtype(torch.float64)
    float64_everywhere()
    generator = torch.Generator().manual_seed(SEED)
    model = DeepseekV3ForCausalLM(config()).double().eval()
    params = {page_path(key): p for key, p in model.named_parameters()}
    fill(params, generator)
    routing = {}
    moe = model.model.layers[DIMS["first_k_dense_replace"]].mlp
    moe.gate.register_forward_hook(lambda _m, _i, out: routing.update(routing_order(out[2][:-1], DIMS["n_routed_experts"])))
    tokens = torch.tensor([SEQUENCE])
    output = model(input_ids=tokens, labels=tokens)
    output.loss.backward()
    payload = {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "command": COMMAND,
        "seed": SEED,
        "sequence": SEQUENCE,
        "weights": {path: encode(p) for path, p in params.items()},
        "loss": output.loss.item(),
        "logits": encode(output.logits[0, :-1]),
        "grads": {path: encode(p.grad if p.grad is not None else torch.zeros_like(p)) for path, p in params.items()},
        "routing": routing,
    }
    OUT.write_text(json.dumps(payload, indent=1) + "\n")
    print(f"{OUT.name}: loss {output.loss.item():.6f}")


if __name__ == "__main__":
    main()
