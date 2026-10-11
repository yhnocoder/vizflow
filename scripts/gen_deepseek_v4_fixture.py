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
from transformers.models.deepseek_v4 import modeling_deepseek_v4 as hf
from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config

SEED = 20261011
COMMAND = "uv run scripts/gen_deepseek_v4_fixture.py"
OUT = Path(__file__).resolve().parent.parent / "test" / "fixtures" / "torch" / "deepseek_v4.json"

DIMS = {"T": 4, "d": 4, "V": 5, "H": 2, "D": 6, "rd": 4, "rq": 3, "g": 2, "dg": 2, "W": 2, "m": 2, "mh": 3, "HI": 2, "cI": 5, "kI": 1, "N": 2, "E": 4, "K": 2, "f": 2}
SINKHORN_ITERS = 3
SWIGLU_LIMIT = 2.0
SEQUENCE = [2, 4, 3, 0, 1]
TID2EID = [[0, 1], [1, 2], [2, 3], [3, 0], [0, 2]]
ROUTER_BIAS = [0.0, 0.1, -0.02, 0.03]
LAYERS = ["csa_layer", "hca_layer"]


def float64_everywhere():
    to, as_type, tensor_softmax, softmax = torch.Tensor.to, torch.Tensor.type, torch.Tensor.softmax, F.softmax

    def to64(self, *args, **kwargs):
        args = tuple(torch.float64 if a is torch.float32 else a for a in args)
        if kwargs.get("dtype") is torch.float32:
            kwargs["dtype"] = torch.float64
        return to(self, *args, **kwargs)

    def type64(self, *args, **kwargs):
        args = tuple(torch.float64 if a is torch.float32 else a for a in args)
        return as_type(self, *args, **kwargs)

    def tensor_softmax64(self, *args, **kwargs):
        if kwargs.get("dtype") is torch.float32:
            kwargs["dtype"] = torch.float64
        return tensor_softmax(self, *args, **kwargs)

    def softmax64(*args, **kwargs):
        if kwargs.get("dtype") is torch.float32:
            kwargs["dtype"] = torch.float64
        return softmax(*args, **kwargs)

    torch.Tensor.to = to64
    torch.Tensor.type = type64
    torch.Tensor.float = torch.Tensor.double
    torch.Tensor.softmax = tensor_softmax64
    F.softmax = softmax64
    torch.nn.functional.softmax = softmax64


def config():
    return DeepseekV4Config(
        vocab_size=DIMS["V"],
        hidden_size=DIMS["d"],
        moe_intermediate_size=DIMS["f"],
        num_hidden_layers=2,
        num_attention_heads=DIMS["H"],
        num_key_value_heads=1,
        head_dim=DIMS["D"],
        partial_rotary_factor=DIMS["rd"] / DIMS["D"],
        q_lora_rank=DIMS["rq"],
        o_groups=DIMS["g"],
        o_lora_rank=DIMS["dg"],
        sliding_window=DIMS["W"],
        layer_types=["compressed_sparse_attention", "heavily_compressed_attention"],
        compress_rates={"compressed_sparse_attention": DIMS["m"], "heavily_compressed_attention": DIMS["mh"]},
        mlp_layer_types=["hash_moe", "moe"],
        rope_theta=10000.0,
        compress_rope_theta=160000.0,
        index_n_heads=DIMS["HI"],
        index_head_dim=DIMS["cI"],
        index_topk=DIMS["kI"],
        hc_mult=DIMS["N"],
        hc_sinkhorn_iters=SINKHORN_ITERS,
        hc_eps=1e-6,
        n_routed_experts=DIMS["E"],
        num_experts_per_tok=DIMS["K"],
        n_shared_experts=1,
        scoring_func="sqrtsoftplus",
        routed_scaling_factor=1.5,
        swiglu_limit=SWIGLU_LIMIT,
        rms_norm_eps=1e-6,
        tie_word_embeddings=False,
        attn_implementation="eager",
        experts_implementation="eager",
        dtype=torch.float64,
    )


def compressor_paths(prefix, c):
    return {
        f"{prefix}.Wkv": c.kv_proj.weight,
        f"{prefix}.Wgate": c.gate_proj.weight,
        f"{prefix}.position_bias": c.position_bias,
        f"{prefix}.kv_norm.weight": c.kv_norm.weight,
    }


def hc_paths(prefix, hc):
    return {f"{prefix}.fn": hc.fn, f"{prefix}.base": hc.base, f"{prefix}.scale": hc.scale}


def mlp_paths(prefix, mlp):
    return {f"{prefix}.Wg": mlp.gate_proj.weight, f"{prefix}.Wu": mlp.up_proj.weight, f"{prefix}.Wd": mlp.down_proj.weight}


def layer_paths(name, layer):
    attn = layer.self_attn
    paths = {
        f"{name}.input_layernorm.weight": layer.input_layernorm.weight,
        f"{name}.post_attention_layernorm.weight": layer.post_attention_layernorm.weight,
        f"{name}.self_attn.Wqa": attn.q_a_proj.weight,
        f"{name}.self_attn.q_a_norm.weight": attn.q_a_norm.weight,
        f"{name}.self_attn.Wqb": attn.q_b_proj.weight,
        f"{name}.self_attn.Wkv": attn.kv_proj.weight,
        f"{name}.self_attn.kv_norm.weight": attn.kv_norm.weight,
        f"{name}.self_attn.Woa": attn.o_a_proj.weight,
        f"{name}.self_attn.Wob": attn.o_b_proj.weight,
        f"{name}.self_attn.sinks": attn.sinks,
        f"{name}.mlp.gate.Wr": layer.mlp.gate.weight,
    }
    paths.update(hc_paths(f"{name}.attn_hc", layer.attn_hc))
    paths.update(hc_paths(f"{name}.ffn_hc", layer.ffn_hc))
    paths.update(compressor_paths(f"{name}.self_attn.compressor", attn.compressor))
    if isinstance(attn.compressor, hf.DeepseekV4CSACompressor):
        indexer = attn.compressor.indexer
        paths.update(compressor_paths(f"{name}.self_attn.indexer.compressor", indexer))
        paths[f"{name}.self_attn.indexer.Wqb"] = indexer.q_b_proj.weight
        paths[f"{name}.self_attn.indexer.Ww"] = indexer.scorer.weights_proj.weight
    paths.update(mlp_paths(f"{name}.mlp.shared_experts", layer.mlp.shared_experts))
    return paths


def page_paths(model, lm_head, take):
    paths = {"E": model.embed_tokens.weight, "W": lm_head.weight, "norm.weight": model.norm.weight}
    paths.update({"hc_head.fn": model.hc_head.hc_fn, "hc_head.base": model.hc_head.hc_base, "hc_head.scale": model.hc_head.hc_scale})
    for name, layer in zip(LAYERS, model.layers):
        paths.update(layer_paths(name, layer))
    paths = {path: take(p) for path, p in paths.items()}
    f = DIMS["f"]
    for name, layer in zip(LAYERS, model.layers):
        experts = layer.mlp.experts
        for e in range(DIMS["E"]):
            gate_up = take(experts.gate_up_proj)[e]
            paths[f"{name}.mlp.expert[{e + 1}].Wg"] = gate_up[:f]
            paths[f"{name}.mlp.expert[{e + 1}].Wu"] = gate_up[f:]
            paths[f"{name}.mlp.expert[{e + 1}].Wd"] = take(experts.down_proj)[e]
    return paths


def fill(model, lm_head, generator):
    with torch.no_grad():
        for name, p in [*model.named_parameters(), ("lm_head.weight", lm_head.weight)]:
            if "norm" in name:
                p.copy_(1 + torch.randn(p.shape, generator=generator) * 0.1)
            else:
                p.copy_(torch.randn(p.shape, generator=generator) * 0.6)
        model.layers[0].mlp.gate.tid2eid.copy_(torch.tensor(TID2EID))
        model.layers[1].mlp.gate.e_score_correction_bias.copy_(torch.tensor(ROUTER_BIAS))


def encode(t):
    return {"shape": list(t.shape), "values": [float(v) for v in t.detach().reshape(-1)]}


def hash_mask():
    mask = torch.zeros(DIMS["V"], DIMS["E"])
    for v, experts in enumerate(TID2EID):
        mask[v, experts] = 1.0
    return mask


def main():
    torch.set_default_dtype(torch.float64)
    float64_everywhere()
    generator = torch.Generator().manual_seed(SEED)
    model = hf.DeepseekV4Model(config()).double().eval()
    lm_head = torch.nn.Linear(DIMS["d"], DIMS["V"], bias=False).double()
    fill(model, lm_head, generator)
    ids = torch.tensor([SEQUENCE[:-1]])
    labels = torch.tensor(SEQUENCE[1:])
    captured = {}
    for i, name in enumerate(["X1", "X2"]):
        model.layers[i].register_forward_hook(lambda _m, _i, out, name=name: captured.__setitem__(name, out))
    model.layers[0].self_attn.compressor.indexer.register_forward_hook(lambda _m, _i, out: captured.__setitem__("selected", out))
    hidden = model(input_ids=ids, use_cache=False).last_hidden_state
    logp = F.log_softmax(lm_head(hidden).double(), -1)
    loss = F.nll_loss(logp.view(-1, DIMS["V"]), labels)
    loss.backward()
    weights = page_paths(model, lm_head, lambda p: p)
    grads = page_paths(model, lm_head, lambda p: p.grad if p.grad is not None else torch.zeros_like(p))
    indexer_without_grad = sorted(
        name for name, p in model.layers[0].self_attn.compressor.indexer.named_parameters() if p.grad is None
    )
    payload = {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "command": COMMAND,
        "seed": SEED,
        "dims": DIMS,
        "sinkhorn_iters": SINKHORN_ITERS,
        "swiglu_limit": SWIGLU_LIMIT,
        "sequence": SEQUENCE,
        "tid2eid": TID2EID,
        "hash_mask": hash_mask().tolist(),
        "router_bias": ROUTER_BIAS,
        "indexer_params_without_grad": indexer_without_grad,
        "weights": {path: encode(t) for path, t in weights.items()},
        "loss": loss.item(),
        "nodes": {"X1": encode(captured["X1"][0]), "X2": encode(captured["X2"][0]), "h": encode(hidden[0])},
        "selected": captured["selected"][0].tolist(),
        "grads": {path: encode(t) for path, t in grads.items()},
    }
    OUT.write_text(json.dumps(payload, indent=1) + "\n")
    print(f"{OUT.name}: loss {loss.item():.6f}, selected {payload['selected']}, indexer params without grad {indexer_without_grad}")


if __name__ == "__main__":
    main()
