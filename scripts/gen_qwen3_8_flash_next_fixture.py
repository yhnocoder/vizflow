# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "transformers==5.19.0"]
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
from transformers.models.qwen4_exp import modeling_qwen4_exp as hf
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig

SEED = 20261011
COMMAND = "uv run scripts/gen_qwen3_8_flash_next_fixture.py"
OUT = Path(__file__).resolve().parent.parent / "test" / "fixtures" / "torch" / "qwen3_8_flash_next.json"
SEQUENCE = [2, 4, 3, 0]

CONFIG = dict(
    vocab_size=5,
    hidden_size=4,
    num_hidden_layers=2,
    layer_types=["linear_attention", "full_attention"],
    num_attention_heads=2,
    num_key_value_heads=1,
    head_dim=8,
    linear_num_key_heads=1,
    linear_num_value_heads=2,
    linear_key_head_dim=2,
    linear_value_head_dim=2,
    linear_conv_kernel_dim=4,
    moe_intermediate_size=2,
    shared_expert_intermediate_size=2,
    num_experts=4,
    num_experts_per_tok=2,
    norm_topk_prob=True,
    hc_count=2,
    hc_lowrank=2,
    ple_layer_ids=[1],
    ple_embed_dim=8,
    ple_conv_kernel_size=4,
    ngram_size=3,
    heads_per_ngram=2,
    ngram_vocab_size_base=12,
    make_ngram_vocab_size_divisible_by=1,
    split_ngram_parts=1,
    indexer_n_heads=2,
    indexer_kv_heads=1,
    indexer_head_dim=4,
    indexer_budget=2,
    indexer_compress_ratio=1,
    output_gate_type="sigmoid",
    eos_token_id=5,
    rope_parameters={"rope_type": "default", "rope_theta": 1e7, "partial_rotary_factor": 0.25, "mrope_section": [1, 0, 0], "mrope_interleaved": True},
    rms_norm_eps=1e-6,
    tie_word_embeddings=False,
    attn_implementation="eager",
    experts_implementation="eager",
    dtype=torch.float64,
)


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


def fill(model, lm_head, generator):
    with torch.no_grad():
        for name, p in [*model.named_parameters(), ("lm_head.weight", lm_head.weight)]:
            if name.endswith("A_log"):
                p.copy_(torch.log(torch.rand(p.shape, generator=generator) * 1.5 + 0.2))
            elif name.endswith("dt_bias"):
                p.copy_(torch.rand(p.shape, generator=generator) - 0.5)
            elif name.endswith("linear_attn.norm.weight"):
                p.copy_(1 + torch.randn(p.shape, generator=generator) * 0.1)
            elif "norm" in name:
                p.copy_(torch.randn(p.shape, generator=generator) * 0.1)
            else:
                p.copy_(torch.randn(p.shape, generator=generator) * 0.6)


def hyper_paths(prefix, hc, combine=True):
    paths = {
        f"{prefix}.hc_norm.weight": hc.hc_norm.weight,
        f"{prefix}.Wdown": hc.input_mix_weight_down.weight,
        f"{prefix}.Wup": hc.input_mix_weight_up.weight,
    }
    if not combine:
        return paths
    paths = {path.replace(prefix, f"{prefix}.mix", 1): p for path, p in paths.items()}
    paths[f"{prefix}.Wbeta"] = hc.block_inject_weight.weight
    return paths


def moe_paths(prefix, mlp, take):
    paths = {
        f"{prefix}.routed.Wr": take(mlp.gate.weight),
        f"{prefix}.Wsg": take(mlp.shared_expert_gate.weight),
        f"{prefix}.shared_expert.Wg": take(mlp.shared_expert.gate_proj.weight),
        f"{prefix}.shared_expert.Wu": take(mlp.shared_expert.up_proj.weight),
        f"{prefix}.shared_expert.Wd": take(mlp.shared_expert.down_proj.weight),
        f"{prefix}.routed.experts.Wgu": take(mlp.experts.gate_up_proj),
        f"{prefix}.routed.experts.Wd": take(mlp.experts.down_proj),
    }
    return paths


def page_paths(model, lm_head, take):
    lin, full = model.layers
    ple, la, sa = lin.ple, lin.linear_attn, full.self_attn
    paths = {"E": model.embed_tokens.weight, "W": lm_head.weight}
    for name, layer in [("linear_layer", lin), ("full_layer", full)]:
        paths.update(hyper_paths(f"{name}.attn_hyper_connection", layer.attn_hyper_connection))
        paths.update(hyper_paths(f"{name}.mlp_hyper_connection", layer.mlp_hyper_connection))
    paths.update(hyper_paths("hyper_connection_mixer", model.hyper_connection_mixer, combine=False))
    paths.update({
        "linear_layer.ple.ple_embedding.Wng": ple.ple_embedding.ngram_embedding.weight,
        "linear_layer.ple.Wkey": ple.key_proj.weight,
        "linear_layer.ple.Wval": ple.value_proj.weight,
        "linear_layer.ple.norm_key.weight": ple.norm_key.weight,
        "linear_layer.ple.norm_query.weight": ple.norm_query.weight,
        "linear_layer.ple.norm_conv.weight": ple.norm_conv.weight,
        "linear_layer.ple.Wconv": ple.conv1d.weight,
        "linear_layer.linear_attn.Wqkv": la.in_proj_qkv.weight,
        "linear_layer.linear_attn.Wz": la.in_proj_z.weight,
        "linear_layer.linear_attn.Wb": la.in_proj_b.weight,
        "linear_layer.linear_attn.Wa": la.in_proj_a.weight,
        "linear_layer.linear_attn.Wconv": la.conv1d.weight,
        "linear_layer.linear_attn.dt_bias": la.dt_bias,
        "linear_layer.linear_attn.A_log": la.A_log,
        "linear_layer.linear_attn.norm.weight": la.norm.weight,
        "linear_layer.linear_attn.Wout": la.out_proj.weight,
        "full_layer.self_attn.attention.Wq": sa.q_proj.weight,
        "full_layer.self_attn.attention.Wk": sa.k_proj.weight,
        "full_layer.self_attn.attention.Wv": sa.v_proj.weight,
        "full_layer.self_attn.attention.Wo": sa.o_proj.weight,
        "full_layer.self_attn.attention.q_norm.weight": sa.q_norm.weight,
        "full_layer.self_attn.attention.k_norm.weight": sa.k_norm.weight,
        "full_layer.self_attn.indexer.Wiqk": sa.indexer.index_qk_proj.weight,
        "full_layer.self_attn.indexer.q_layernorm.weight": sa.indexer.q_layernorm.weight,
        "full_layer.self_attn.indexer.k_layernorm.weight": sa.indexer.k_layernorm.weight,
    })
    paths = {path: (take(p)[:, 0, :] if path.endswith("Wconv") else take(p)) for path, p in paths.items()}
    paths.update(moe_paths("linear_layer.mlp", lin.mlp, take))
    paths.update(moe_paths("full_layer.mlp", full.mlp, take))
    return paths


def encode(t):
    return {"shape": list(t.shape), "values": [float(v) for v in t.detach().reshape(-1)]}


def encode_int64(t):
    return {"shape": list(t.shape), "values": [str(int(v)) for v in t.reshape(-1)]}


def main():
    torch.set_default_dtype(torch.float64)
    float64_everywhere()
    config = Qwen4ExpTextConfig(**CONFIG)
    model = hf.Qwen4ExpTextModel(config).double().eval()
    lm_head = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False).double()
    fill(model, lm_head, torch.Generator().manual_seed(SEED))
    lin, full = model.layers
    captured = {}
    lin.register_forward_hook(lambda _m, _i, out: captured.__setitem__("linear_layer", out))
    full.register_forward_hook(lambda _m, _i, out: captured.__setitem__("full_layer", out))
    ngram = lin.ple.ple_embedding
    ngram.ngram_embedding.register_forward_pre_hook(lambda _m, args: captured.__setitem__("ngram_ids", args[0]))
    full.self_attn.indexer.register_forward_hook(lambda _m, _i, out: captured.__setitem__("indexer", out))
    ids = torch.tensor([SEQUENCE[:-1]])
    labels = torch.tensor(SEQUENCE[1:])
    hidden = model(input_ids=ids).last_hidden_state
    logp = F.log_softmax(lm_head(hidden).double(), -1)
    loss = F.nll_loss(logp.view(-1, config.vocab_size), labels)
    loss.backward()
    weights = page_paths(model, lm_head, lambda p: p)
    grads = page_paths(model, lm_head, lambda p: p.grad if p.grad is not None else torch.zeros_like(p))
    no_grad = sorted(name for name, p in model.named_parameters() if p.grad is None)
    selected = (captured["indexer"][0, 0] == 0).double()
    payload = {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "command": COMMAND,
        "seed": SEED,
        "sequence": SEQUENCE,
        "weights": {path: encode(t) for path, t in weights.items()},
        "loss": loss.item(),
        "nodes": {
            "H2": encode(captured["linear_layer"][0]),
            "H48": encode(captured["full_layer"][0]),
            "h": encode(hidden[0]),
            "full_layer.self_attn.indexer.sel_tok": encode(selected),
        },
        "int64": {
            "linear_layer.ple.ple_embedding.m": encode_int64(ngram.layer_multipliers),
            "linear_layer.ple.ple_embedding.sizes": encode_int64(ngram.ngram_heads_vocab_sizes),
            "linear_layer.ple.ple_embedding.offsets": encode_int64(ngram.ngram_heads_offsets),
            "linear_layer.ple.ple_embedding.ngram_ids": encode_int64(captured["ngram_ids"][0]),
        },
        "no_grad": no_grad,
        "grads": {path: encode(t) for path, t in grads.items()},
    }
    OUT.write_text(json.dumps(payload, indent=1) + "\n")
    print(f"{OUT.name}: loss {loss.item():.6f}, params without grad {no_grad}")


if __name__ == "__main__":
    main()
