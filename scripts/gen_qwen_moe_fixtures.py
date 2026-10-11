# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "transformers>=5.19"]
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
from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as hf35
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig
from transformers.models.qwen3_moe import modeling_qwen3_moe as hf3
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig

SEED = 20261011
COMMAND = "uv run scripts/gen_qwen_moe_fixtures.py"
OUT_DIR = Path(__file__).resolve().parent.parent / "test" / "fixtures" / "torch"
SEQUENCE = [2, 4, 3, 0]

QWEN3_MOE = {"T": 3, "d": 4, "V": 5, "H": 2, "Hkv": 1, "dh": 4, "Ne": 4, "Nk": 2, "mf": 2}
QWEN3_5_MOE = {"T": 3, "d": 4, "V": 5, "H": 2, "Hkv": 1, "dh": 8, "rd": 2, "Hk": 1, "Hv": 2, "dk": 2, "dv": 2, "K": 4, "Ne": 4, "Nk": 2, "mf": 2, "sf": 2}


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


def qwen3_moe_config(dims):
    return Qwen3MoeConfig(
        vocab_size=dims["V"],
        hidden_size=dims["d"],
        moe_intermediate_size=dims["mf"],
        num_hidden_layers=1,
        num_attention_heads=dims["H"],
        num_key_value_heads=dims["Hkv"],
        head_dim=dims["dh"],
        num_experts=dims["Ne"],
        num_experts_per_tok=dims["Nk"],
        norm_topk_prob=True,
        decoder_sparse_step=1,
        mlp_only_layers=[],
        rope_parameters={"rope_type": "default", "rope_theta": 1e6},
        rms_norm_eps=1e-6,
        tie_word_embeddings=False,
        attn_implementation="eager",
        experts_implementation="eager",
        dtype=torch.float64,
    )


def qwen3_5_moe_config(dims):
    return Qwen3_5MoeTextConfig(
        vocab_size=dims["V"],
        hidden_size=dims["d"],
        moe_intermediate_size=dims["mf"],
        shared_expert_intermediate_size=dims["sf"],
        num_experts=dims["Ne"],
        num_experts_per_tok=dims["Nk"],
        num_hidden_layers=2,
        layer_types=["linear_attention", "full_attention"],
        num_attention_heads=dims["H"],
        num_key_value_heads=dims["Hkv"],
        head_dim=dims["dh"],
        linear_num_key_heads=dims["Hk"],
        linear_num_value_heads=dims["Hv"],
        linear_key_head_dim=dims["dk"],
        linear_value_head_dim=dims["dv"],
        linear_conv_kernel_dim=dims["K"],
        rope_parameters={"rope_type": "default", "rope_theta": 1e7, "partial_rotary_factor": dims["rd"] / dims["dh"], "mrope_section": [1, 0, 0], "mrope_interleaved": True},
        rms_norm_eps=1e-6,
        attn_output_gate=True,
        tie_word_embeddings=False,
        attn_implementation="eager",
        experts_implementation="eager",
        dtype=torch.float64,
    )


def expert_paths(prefix, experts, dims, take):
    paths = {}
    mf = dims["mf"]
    for e in range(dims["Ne"]):
        gate_up = take(experts.gate_up_proj)[e]
        paths[f"{prefix}.expert[{e + 1}].Wg"] = gate_up[:mf]
        paths[f"{prefix}.expert[{e + 1}].Wu"] = gate_up[mf:]
        paths[f"{prefix}.expert[{e + 1}].Wd"] = take(experts.down_proj)[e]
    return paths


def qwen3_moe_paths(model, lm_head, dims, take):
    layer = model.layers[0]
    sa = layer.self_attn
    paths = {
        "E": model.embed_tokens.weight,
        "W": lm_head.weight,
        "norm.weight": model.norm.weight,
        "layer.input_layernorm.weight": layer.input_layernorm.weight,
        "layer.post_attention_layernorm.weight": layer.post_attention_layernorm.weight,
        "layer.self_attn.Wq": sa.q_proj.weight,
        "layer.self_attn.Wk": sa.k_proj.weight,
        "layer.self_attn.Wv": sa.v_proj.weight,
        "layer.self_attn.Wo": sa.o_proj.weight,
        "layer.self_attn.q_norm.weight": sa.q_norm.weight,
        "layer.self_attn.k_norm.weight": sa.k_norm.weight,
        "layer.mlp.Wr": layer.mlp.gate.weight,
    }
    paths = {path: take(p) for path, p in paths.items()}
    paths.update(expert_paths("layer.mlp", layer.mlp.experts, dims, take))
    return paths


def qwen3_5_moe_paths(model, lm_head, dims, take):
    lin, full = model.layers
    la, sa = lin.linear_attn, full.self_attn
    paths = {"E": model.embed_tokens.weight, "W": lm_head.weight, "norm.weight": model.norm.weight}
    for name, layer in [("linear_layer", lin), ("full_layer", full)]:
        paths[f"{name}.input_layernorm.weight"] = layer.input_layernorm.weight
        paths[f"{name}.post_attention_layernorm.weight"] = layer.post_attention_layernorm.weight
        paths[f"{name}.mlp.routed.Wr"] = layer.mlp.gate.weight
        paths[f"{name}.mlp.Wsg"] = layer.mlp.shared_expert_gate.weight
        paths[f"{name}.mlp.shared_expert.Wg"] = layer.mlp.shared_expert.gate_proj.weight
        paths[f"{name}.mlp.shared_expert.Wu"] = layer.mlp.shared_expert.up_proj.weight
        paths[f"{name}.mlp.shared_expert.Wd"] = layer.mlp.shared_expert.down_proj.weight
    paths.update({
        "linear_layer.linear_attn.Wqkv": la.in_proj_qkv.weight,
        "linear_layer.linear_attn.Wz": la.in_proj_z.weight,
        "linear_layer.linear_attn.Wb": la.in_proj_b.weight,
        "linear_layer.linear_attn.Wa": la.in_proj_a.weight,
        "linear_layer.linear_attn.Wconv": la.conv1d.weight,
        "linear_layer.linear_attn.dt_bias": la.dt_bias,
        "linear_layer.linear_attn.A_log": la.A_log,
        "linear_layer.linear_attn.norm.weight": la.norm.weight,
        "linear_layer.linear_attn.Wout": la.out_proj.weight,
        "full_layer.self_attn.Wq": sa.q_proj.weight,
        "full_layer.self_attn.Wk": sa.k_proj.weight,
        "full_layer.self_attn.Wv": sa.v_proj.weight,
        "full_layer.self_attn.Wo": sa.o_proj.weight,
        "full_layer.self_attn.q_norm.weight": sa.q_norm.weight,
        "full_layer.self_attn.k_norm.weight": sa.k_norm.weight,
    })
    paths = {path: (take(p)[:, 0, :] if path.endswith("Wconv") else take(p)) for path, p in paths.items()}
    for name, layer in [("linear_layer", lin), ("full_layer", full)]:
        paths.update(expert_paths(f"{name}.mlp.routed", layer.mlp.experts, dims, take))
    return paths


def fill(model, lm_head, generator, zero_centered):
    with torch.no_grad():
        for name, p in [*model.named_parameters(), ("lm_head.weight", lm_head.weight)]:
            if name.endswith("A_log"):
                p.copy_(torch.log(torch.rand(p.shape, generator=generator) * 1.5 + 0.2))
            elif name.endswith("dt_bias"):
                p.copy_(torch.rand(p.shape, generator=generator) - 0.5)
            elif "norm" in name and zero_centered and not name.endswith("linear_attn.norm.weight"):
                p.copy_(torch.randn(p.shape, generator=generator) * 0.1)
            elif "norm" in name:
                p.copy_(1 + torch.randn(p.shape, generator=generator) * 0.1)
            else:
                p.copy_(torch.randn(p.shape, generator=generator) * 0.6)


def encode(t):
    return {"shape": list(t.shape), "values": [float(v) for v in t.detach().reshape(-1)]}


def run(name, model, dims, paths_of, node_names, zero_centered):
    generator = torch.Generator().manual_seed(SEED)
    lm_head = torch.nn.Linear(dims["d"], dims["V"], bias=False).double()
    fill(model, lm_head, generator, zero_centered)
    ids = torch.tensor([SEQUENCE[:-1]])
    labels = torch.tensor(SEQUENCE[1:])
    captured = {}
    for i, node in enumerate(node_names):
        model.layers[i].register_forward_hook(lambda _m, _i, out, node=node: captured.__setitem__(node, out[0] if isinstance(out, tuple) else out))
    hidden = model(input_ids=ids).last_hidden_state
    logp = F.log_softmax(lm_head(hidden).double(), -1)
    loss = F.nll_loss(logp.view(-1, dims["V"]), labels)
    loss.backward()
    weights = paths_of(model, lm_head, dims, lambda p: p)
    grads = paths_of(model, lm_head, dims, lambda p: p.grad if p.grad is not None else torch.zeros_like(p))
    payload = {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "command": COMMAND,
        "seed": SEED,
        "sequence": SEQUENCE,
        "weights": {path: encode(t) for path, t in weights.items()},
        "loss": loss.item(),
        "nodes": {**{node: encode(captured[node][0]) for node in node_names}, "h": encode(hidden[0])},
        "grads": {path: encode(t) for path, t in grads.items()},
    }
    out = OUT_DIR / f"{name}.json"
    out.write_text(json.dumps(payload, indent=1) + "\n")
    print(f"{out.name}: loss {loss.item():.6f}")


def main():
    torch.set_default_dtype(torch.float64)
    float64_everywhere()
    run("qwen3_moe", hf3.Qwen3MoeModel(qwen3_moe_config(QWEN3_MOE)).double().eval(), QWEN3_MOE, qwen3_moe_paths, ["x"], False)
    run("qwen3_5_moe", hf35.Qwen3_5MoeTextModel(qwen3_5_moe_config(QWEN3_5_MOE)).double().eval(), QWEN3_5_MOE, qwen3_5_moe_paths, ["x1", "x"], True)


if __name__ == "__main__":
    main()
