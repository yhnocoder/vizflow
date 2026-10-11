# /// script
# requires-python = ">=3.10"
# dependencies = ["torch"]
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
import math
import sys
import tempfile
import types
import urllib.request
from pathlib import Path

import torch

REVISION = "60d8d70770c6776ff598c94bb586a859a38244f1"
SOURCE = f"https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/raw/{REVISION}/inference/model.py"
COMMAND = "uv run scripts/gen_deepseek_v4_kv_fixture.py"
OUT = Path(__file__).resolve().parent.parent / "examples" / "kv" / "deepseek-v4-fixture.json"
SEED = 20261011
PREFILL = 5
LAST = 19
LAYERS = {"swa": 0, "csa": 4, "hca": 8}
TOY = dict(
    max_batch_size=1,
    max_seq_len=32,
    dtype="bf16",
    scale_fmt=None,
    dim=8,
    n_heads=2,
    q_lora_rank=4,
    head_dim=8,
    rope_head_dim=2,
    o_groups=2,
    o_lora_rank=3,
    window_size=4,
    index_n_heads=2,
    index_head_dim=4,
    index_topk=2,
    compress_rope_theta=160000.0,
    original_seq_len=65536,
    rope_theta=10000.0,
    rope_factor=16,
    beta_fast=32,
    beta_slow=1,
)

calls = []


def sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale):
    idx = topk_idxs.long()
    valid = idx != -1
    rows = kv[0][idx.clamp(min=0)] * valid.unsqueeze(-1)
    scores = torch.einsum("bshd,bskd->bshk", q, rows) * softmax_scale
    scores = scores.masked_fill(~valid.unsqueeze(2), float("-inf"))
    sink = attn_sink.view(1, 1, -1, 1).expand(*scores.shape[:3], 1)
    probs = torch.cat([scores, sink], dim=-1).softmax(dim=-1)[..., :-1]
    o = torch.einsum("bshk,bskd->bshd", probs, rows)
    calls.append({"q": q.clone(), "kv": kv.clone(), "topk": topk_idxs.clone(), "o": o.clone()})
    return o


def hadamard(n):
    h = torch.ones(1, 1)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h


def load_official():
    folder = Path(tempfile.mkdtemp())
    (folder / "model.py").write_bytes(urllib.request.urlopen(SOURCE).read())
    kernel = types.ModuleType("kernel")
    kernel.act_quant = lambda x, *args, **kwargs: x
    kernel.fp4_act_quant = lambda x, *args, **kwargs: x
    kernel.sparse_attn = sparse_attn
    kernel.fp8_gemm = kernel.fp4_gemm = kernel.hc_split_sinkhorn = None
    sys.modules["kernel"] = kernel
    sys.path.insert(0, str(folder))
    import model

    model.rotate_activation = lambda x: x @ hadamard(x.size(-1)).to(x.dtype) * x.size(-1) ** -0.5
    return model


def values(t):
    return [values(x) for x in t] if t.dim() > 0 else float(t.detach())


def init(module, rng):
    for name, p in sorted(module.named_parameters()):
        if p.dtype.is_complex:
            continue
        if name.endswith("norm.weight"):
            data = 1 + 0.3 * (torch.rand(p.shape, generator=rng, dtype=torch.float64) * 2 - 1)
        else:
            data = torch.rand(p.shape, generator=rng, dtype=torch.float64) * 2 - 1
        p.data = (torch.round(data * 10) / 10).to(torch.float32 if p.dtype == torch.float32 else torch.float64)


def state(attn):
    out = {"kv_cache": attn.kv_cache[0]}
    if attn.compress_ratio:
        out["kv_state"] = attn.compressor.kv_state[0]
        out["score_state"] = attn.compressor.score_state[0]
        if attn.indexer is not None:
            out["index_cache"] = attn.indexer.kv_cache[0]
            out["index_kv_state"] = attn.indexer.compressor.kv_state[0]
            out["index_score_state"] = attn.indexer.compressor.score_state[0]
    return {k: values(v.clone().to(torch.float64)) for k, v in out.items()}


def encode(obj):
    if isinstance(obj, float):
        if math.isinf(obj):
            return "inf" if obj > 0 else "-inf"
        return round(obj, 10)
    if isinstance(obj, list):
        return [encode(x) for x in obj]
    if isinstance(obj, dict):
        return {k: encode(v) for k, v in obj.items()}
    return obj


def main():
    torch.set_default_dtype(torch.float64)
    model = load_official()
    model.default_dtype = torch.float64
    rng = torch.Generator().manual_seed(SEED)
    xs = torch.round((torch.rand(1, LAST + 1, TOY["dim"], generator=rng, dtype=torch.float64) * 2 - 1) * 10) / 10
    layers = {}
    for kind, ratio in LAYERS.items():
        args = model.ModelArgs(**TOY, compress_ratios=(ratio,))
        attn = model.Attention(0, args)
        init(attn, rng)
        weights = {name: values(p.data.to(torch.float64)) for name, p in sorted(attn.named_parameters())}
        freqs = attn.freqs_cis[: LAST + 1]
        calls.clear()
        attn(xs[:, :PREFILL].clone(), 0)
        initial = state(attn)
        steps = []
        for t in range(PREFILL, LAST + 1):
            calls.clear()
            y = attn(xs[:, t : t + 1].clone(), t)
            call = calls[-1]
            steps.append({
                "t": t,
                "after": state(attn),
                "topk_idxs": values(call["topk"][0, 0].to(torch.float64)),
                "o": values(call["o"][0, 0].to(torch.float64)),
                "y": values(y[0, 0].to(torch.float64)),
            })
        layers[kind] = {
            "compress_ratio": ratio,
            "weights": weights,
            "cos": values(freqs.real.to(torch.float64)),
            "sin": values(freqs.imag.to(torch.float64)),
            "initial": initial,
            "steps": steps,
        }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fixture = {"torch": torch.__version__, "command": COMMAND, "source": SOURCE, "seed": SEED, "config": TOY, "prefill": PREFILL, "x": values(xs[0]), "layers": layers}
    OUT.write_text(json.dumps(encode(fixture)) + "\n")


if __name__ == "__main__":
    main()
