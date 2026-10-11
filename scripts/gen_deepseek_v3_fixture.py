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
from pathlib import Path

import torch
import torch.nn.functional as F
import transformers
from transformers.models.deepseek_v3 import modeling_deepseek_v3 as hf
from transformers.models.deepseek_v3.configuration_deepseek_v3 import DeepseekV3Config

SEED = 20261011
COMMAND = "uv run scripts/gen_deepseek_v3_fixture.py"
OUT = Path(__file__).resolve().parent.parent / "test" / "fixtures" / "torch" / "deepseek_v3.json"

DIMS = {"T": 4, "d": 4, "f": 6, "V": 5, "H": 2, "dn": 2, "dr": 2, "dv": 2, "rq": 3, "rkv": 3, "Ne": 6, "G": 2, "TG": 1, "K": 2, "mf": 2, "Sh": 1}
SEQUENCE = [2, 4, 3, 0, 1]
ROUTER_BIAS = [0, 0.1, -0.02, 0, 0.02, -0.03]


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
        hidden_size=DIMS["d"],
        intermediate_size=DIMS["f"],
        moe_intermediate_size=DIMS["mf"],
        num_hidden_layers=2,
        first_k_dense_replace=1,
        num_attention_heads=DIMS["H"],
        num_key_value_heads=DIMS["H"],
        n_shared_experts=DIMS["Sh"],
        n_routed_experts=DIMS["Ne"],
        num_experts_per_tok=DIMS["K"],
        n_group=DIMS["G"],
        topk_group=DIMS["TG"],
        routed_scaling_factor=2.5,
        norm_topk_prob=True,
        q_lora_rank=DIMS["rq"],
        kv_lora_rank=DIMS["rkv"],
        qk_nope_head_dim=DIMS["dn"],
        qk_rope_head_dim=DIMS["dr"],
        v_head_dim=DIMS["dv"],
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
        rope_interleave=True,
        rms_norm_eps=1e-6,
        tie_word_embeddings=False,
        attn_implementation="eager",
        experts_implementation="eager",
        dtype=torch.float64,
    )


def attention_paths(prefix, attn):
    return {
        f"{prefix}.self_attn.Wqa": attn.q_a_proj.weight,
        f"{prefix}.self_attn.q_a_layernorm.weight": attn.q_a_layernorm.weight,
        f"{prefix}.self_attn.Wqb": attn.q_b_proj.weight,
        f"{prefix}.self_attn.Wkva": attn.kv_a_proj_with_mqa.weight,
        f"{prefix}.self_attn.kv_a_layernorm.weight": attn.kv_a_layernorm.weight,
        f"{prefix}.self_attn.Wkvb": attn.kv_b_proj.weight,
        f"{prefix}.self_attn.Wo": attn.o_proj.weight,
    }


def mlp_paths(prefix, mlp):
    return {f"{prefix}.Wg": mlp.gate_proj.weight, f"{prefix}.Wu": mlp.up_proj.weight, f"{prefix}.Wd": mlp.down_proj.weight}


def page_paths(model, lm_head, take):
    dense, moe = model.layers
    paths = {"E": model.embed_tokens.weight, "W": lm_head.weight, "norm.weight": model.norm.weight}
    for name, layer in [("dense_layer", dense), ("moe_layer", moe)]:
        paths[f"{name}.input_layernorm.weight"] = layer.input_layernorm.weight
        paths[f"{name}.post_attention_layernorm.weight"] = layer.post_attention_layernorm.weight
        paths.update(attention_paths(name, layer.self_attn))
    paths.update(mlp_paths("dense_layer.mlp", dense.mlp))
    paths.update(mlp_paths("moe_layer.mlp.shared_experts", moe.mlp.shared_experts))
    paths["moe_layer.mlp.Wr"] = moe.mlp.gate.weight
    paths = {path: take(p) for path, p in paths.items()}
    experts = moe.mlp.experts
    mf = DIMS["mf"]
    for e in range(DIMS["Ne"]):
        gate_up = take(experts.gate_up_proj)[e]
        paths[f"moe_layer.mlp.expert[{e + 1}].Wg"] = gate_up[:mf]
        paths[f"moe_layer.mlp.expert[{e + 1}].Wu"] = gate_up[mf:]
        paths[f"moe_layer.mlp.expert[{e + 1}].Wd"] = take(experts.down_proj)[e]
    return paths


def fill(model, lm_head, generator):
    with torch.no_grad():
        for name, p in [*model.named_parameters(), ("lm_head.weight", lm_head.weight)]:
            if "norm" in name:
                p.copy_(1 + torch.randn(p.shape, generator=generator) * 0.1)
            else:
                p.copy_(torch.randn(p.shape, generator=generator) * 0.6)
        model.layers[1].mlp.gate.e_score_correction_bias.copy_(torch.tensor(ROUTER_BIAS))


def encode(t):
    return {"shape": list(t.shape), "values": [float(v) for v in t.detach().reshape(-1)]}


def main():
    torch.set_default_dtype(torch.float64)
    float64_everywhere()
    generator = torch.Generator().manual_seed(SEED)
    model = hf.DeepseekV3Model(config()).double().eval()
    lm_head = torch.nn.Linear(DIMS["d"], DIMS["V"], bias=False).double()
    fill(model, lm_head, generator)
    ids = torch.tensor([SEQUENCE[:-1]])
    labels = torch.tensor(SEQUENCE[1:])
    captured = {}
    for i, name in enumerate(["x1", "x2"]):
        model.layers[i].register_forward_hook(lambda _m, _i, out, name=name: captured.__setitem__(name, out[0] if isinstance(out, tuple) else out))
    hidden = model(input_ids=ids).last_hidden_state
    logp = F.log_softmax(lm_head(hidden).double(), -1)
    loss = F.nll_loss(logp.view(-1, DIMS["V"]), labels)
    loss.backward()
    weights = page_paths(model, lm_head, lambda p: p)
    grads = page_paths(model, lm_head, lambda p: p.grad if p.grad is not None else torch.zeros_like(p))
    payload = {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "command": COMMAND,
        "seed": SEED,
        "sequence": SEQUENCE,
        "router_bias": ROUTER_BIAS,
        "weights": {path: encode(t) for path, t in weights.items()},
        "loss": loss.item(),
        "nodes": {"x1": encode(captured["x1"][0]), "x2": encode(captured["x2"][0]), "h": encode(hidden[0])},
        "grads": {path: encode(t) for path, t in grads.items()},
    }
    OUT.write_text(json.dumps(payload, indent=1) + "\n")
    print(f"{OUT.name}: loss {loss.item():.6f}")


if __name__ == "__main__":
    main()
