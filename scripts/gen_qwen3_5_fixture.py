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
from transformers.models.qwen3_5 import modeling_qwen3_5 as hf
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

SEED = 20261011
COMMAND = "uv run scripts/gen_qwen3_5_fixture.py"
OUT = Path(__file__).resolve().parent.parent / "test" / "fixtures" / "torch" / "qwen3_5.json"

DIMS = {"T": 4, "d": 4, "f": 6, "V": 5, "H": 2, "Hkv": 1, "dh": 8, "rd": 2, "Hk": 1, "Hv": 2, "dk": 2, "dv": 2, "K": 3}
SEQUENCE = [2, 4, 3, 0, 1]


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
    return Qwen3_5TextConfig(
        vocab_size=DIMS["V"],
        hidden_size=DIMS["d"],
        intermediate_size=DIMS["f"],
        num_hidden_layers=2,
        layer_types=["linear_attention", "full_attention"],
        num_attention_heads=DIMS["H"],
        num_key_value_heads=DIMS["Hkv"],
        head_dim=DIMS["dh"],
        linear_num_key_heads=DIMS["Hk"],
        linear_num_value_heads=DIMS["Hv"],
        linear_key_head_dim=DIMS["dk"],
        linear_value_head_dim=DIMS["dv"],
        linear_conv_kernel_dim=DIMS["K"],
        rope_parameters={"rope_type": "default", "rope_theta": 1e7, "partial_rotary_factor": DIMS["rd"] / DIMS["dh"], "mrope_section": [1, 0, 0], "mrope_interleaved": True},
        rms_norm_eps=1e-6,
        attn_output_gate=True,
        tie_word_embeddings=False,
        attn_implementation="eager",
        dtype=torch.float64,
    )


def page_paths(model, lm_head):
    lin, full = model.layers
    la, sa = lin.linear_attn, full.self_attn
    params = {
        "E": model.embed_tokens.weight,
        "W": lm_head.weight,
        "norm.weight": model.norm.weight,
    }
    for name, layer in [("linear_layer", lin), ("full_layer", full)]:
        params[f"{name}.input_layernorm.weight"] = layer.input_layernorm.weight
        params[f"{name}.post_attention_layernorm.weight"] = layer.post_attention_layernorm.weight
        params[f"{name}.mlp.Wg"] = layer.mlp.gate_proj.weight
        params[f"{name}.mlp.Wu"] = layer.mlp.up_proj.weight
        params[f"{name}.mlp.Wd"] = layer.mlp.down_proj.weight
    params.update({
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
    return params


def fill(params, generator):
    with torch.no_grad():
        for path, p in params.items():
            if path.endswith("A_log"):
                p.copy_(torch.log(torch.rand(p.shape, generator=generator) * 1.5 + 0.2))
            elif path.endswith("dt_bias"):
                p.copy_(torch.rand(p.shape, generator=generator) - 0.5)
            elif "layernorm" in path or path.startswith("norm") or "_norm" in path:
                p.copy_(torch.randn(p.shape, generator=generator) * 0.1)
            elif path.endswith("norm.weight"):
                p.copy_(1 + torch.randn(p.shape, generator=generator) * 0.1)
            else:
                p.copy_(torch.randn(p.shape, generator=generator) * 0.6)


def encode(t):
    return {"shape": list(t.shape), "values": [float(v) for v in t.detach().reshape(-1)]}


def page_layout(path, t):
    if path.endswith("Wconv"):
        return t[:, 0, :]
    return t


def main():
    torch.set_default_dtype(torch.float64)
    float64_everywhere()
    generator = torch.Generator().manual_seed(SEED)
    cfg = config()
    model = hf.Qwen3_5TextModel(cfg).double().eval()
    lm_head = torch.nn.Linear(DIMS["d"], DIMS["V"], bias=False).double()
    params = page_paths(model, lm_head)
    fill(params, generator)
    ids = torch.tensor([SEQUENCE[:-1]])
    labels = torch.tensor(SEQUENCE[1:])
    captured = {}
    model.layers[0].register_forward_hook(lambda _m, _i, out: captured.__setitem__("x1", out[0] if isinstance(out, tuple) else out))
    model.layers[1].register_forward_hook(lambda _m, _i, out: captured.__setitem__("x", out[0] if isinstance(out, tuple) else out))
    hidden = model(input_ids=ids).last_hidden_state
    logp = F.log_softmax(lm_head(hidden).double(), -1)
    loss = F.nll_loss(logp.view(-1, DIMS["V"]), labels)
    loss.backward()
    payload = {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "command": COMMAND,
        "seed": SEED,
        "sequence": SEQUENCE,
        "weights": {path: encode(page_layout(path, p)) for path, p in params.items()},
        "loss": float(loss),
        "nodes": {"x1": encode(captured["x1"][0]), "x": encode(captured["x"][0]), "h": encode(hidden[0])},
        "grads": {path: encode(page_layout(path, p.grad)) for path, p in params.items()},
    }
    OUT.write_text(json.dumps(payload, indent=1) + "\n")
    print(f"{OUT.name}: loss {float(loss):.6f}")


if __name__ == "__main__":
    main()
