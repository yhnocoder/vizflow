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
from pathlib import Path

import torch
import torch.nn.functional as F

SEED = 20261011
COMMAND = "uv run scripts/gen_quant_fixture.py"
OUT_PATH = Path(__file__).resolve().parent.parent / "test" / "fixtures" / "torch" / "quant.json"
OPS = ["block_scale", "quantize", "dequantize"]

E4M3_MAX = 448.0
AMAX_FLOOR = 1e-4
E2M1_VALUES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
BLOCK_SOURCE = "按 DeepSeek V3 官方 act_quant 与 weight_dequant 的公式在 CPU 上重写，官方实现是需要 GPU 的 Triton 内核"
E2M1_SOURCE = "torch 不能把张量转换成 float4_e2m1fn_x2，这里枚举 E2M1 的 8 个非负可表示值，取最近的一个，平局取编码为偶数的那个"

LINEAR_X = [
    [0.5, -1.25, 6.0, 1.45],
    [-0.3, 0.8, 1.1, -0.45],
    [0.9, 0.2, -0.6, 1.3],
]
LINEAR_Y = [1, 3, 0]
LINEAR_B = 2

generator = torch.Generator().manual_seed(SEED)


def plain(v):
    return float(v) + 0.0


def encode(t):
    return {"shape": list(t.shape), "values": [plain(v) for v in t.detach().reshape(-1).tolist()]}


def fp32_values(values):
    return torch.tensor(values, dtype=torch.float64).to(torch.float32).to(torch.float64)


def rounding_samples(fmt_max, subnormal_min, ties):
    fixed = [0.0, subnormal_min / 2, subnormal_min, subnormal_min * 1.5, subnormal_min * 2.5, *ties, fmt_max]
    magnitude = torch.exp(torch.empty(40, dtype=torch.float64).uniform_(math.log(subnormal_min), math.log(fmt_max), generator=generator))
    values = fixed + magnitude.tolist()
    values = values + [-v for v in values[1:]]
    return fp32_values(values)


def float8_rounding(dtype, fmt_max, subnormal_min, ties):
    x = rounding_samples(fmt_max, subnormal_min, ties)
    y = x.to(torch.float32).to(dtype).to(torch.float64)
    return {"x": [plain(v) for v in x.tolist()], "y": [plain(v) for v in y.tolist()]}


def e2m1_round(v):
    a = abs(v)
    best = min(range(len(E2M1_VALUES)), key=lambda i: (abs(E2M1_VALUES[i] - a), i % 2))
    return math.copysign(E2M1_VALUES[best], v) + 0.0


def e2m1_rounding():
    x = rounding_samples(6.0, 0.5, [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 5.5])
    return {"x": [plain(v) for v in x.tolist()], "y": [e2m1_round(v) for v in x.tolist()], "source": E2M1_SOURCE}


def block_amax(x, br, bc):
    rows, cols = x.shape
    out = torch.zeros(math.ceil(rows / br), math.ceil(cols / bc), dtype=x.dtype)
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            out[i, j] = x[i * br:(i + 1) * br, j * bc:(j + 1) * bc].abs().max()
    return out


def expand(s, br, bc, shape):
    return s.repeat_interleave(br, 0).repeat_interleave(bc, 1)[: shape[0], : shape[1]]


def deepseek_block_quant_cpu_rewrite(x, br, bc, ue8m0):
    x = x.to(torch.float32)
    amax = block_amax(x, br, bc).clamp(min=AMAX_FLOOR)
    s = amax / E4M3_MAX
    if ue8m0:
        s = torch.pow(2.0, torch.ceil(torch.log2(s)))
    y = (x / expand(s, br, bc, x.shape)).to(torch.float8_e4m3fn)
    return s.to(torch.float64), y.to(torch.float64)


def dyadic(rows, cols, denominator):
    return torch.round(torch.randn(rows, cols, generator=generator, dtype=torch.float64) * denominator) / denominator


def block_case(title, x, br, bc, rule):
    s, q = deepseek_block_quant_cpu_rewrite(x, br, bc, rule == "ue8m0")
    return {"title": title, "fmt": "fp8_e4m3", "rule": rule, "br": br, "bc": bc, "x": encode(x), "s": encode(s), "q": encode(q), "xq": encode(q * expand(s, br, bc, q.shape))}


def block_cases():
    act = dyadic(3, 300, 256) * 2
    act[0, 5] = 37.5
    act[2, 290] = -0.001953125
    act[1, 128:256] = 0.0
    tall = dyadic(130, 3, 1024) * 0.05
    tall[3, 1] = 1.75
    tall[128:, :] = 0.0
    wide = dyadic(2, 130, 1024) * 0.05
    wide[1, 129] = -0.75
    small = dyadic(5, 7, 64)
    small[4, 6] = -9.0
    return [
        block_case("activation (3, 300), 1 x 128 blocks, outlier and zero block", act, 1, 128, "amax"),
        block_case("activation (3, 300), 1 x 128 blocks, ue8m0 scales", act, 1, 128, "ue8m0"),
        block_case("weight (130, 3), 128 x 128 blocks, zero block", tall, 128, 128, "amax"),
        block_case("weight (2, 130), 128 x 128 blocks, ue8m0 scales", wide, 128, 128, "ue8m0"),
        block_case("small (5, 7), 2 x 3 blocks", small, 2, 3, "amax"),
    ]


def bf16(t):
    return t.to(torch.bfloat16).to(torch.float64)


def fp32(t):
    return t.to(torch.float32).to(torch.float64)


def amax_scale(x, br, bc):
    amax = block_amax(x, br, bc).to(torch.float32).clamp(min=AMAX_FLOOR)
    return (amax / E4M3_MAX).to(torch.float64)


def fake_quant(x, br, bc):
    s = amax_scale(x, br, bc)
    full = expand(s, br, bc, x.shape)
    scaled = (x / full).to(torch.float32)
    q = scaled.to(torch.float8_e4m3fn).to(torch.float64)
    keep = (scaled.abs() <= E4M3_MAX).to(torch.float64)
    return s, full, q, keep, bf16(q * full)


def straight_through(grad, full, keep):
    return grad * full / full * keep


def linear_case():
    x = bf16(torch.tensor(LINEAR_X, dtype=torch.float64))
    W_given = torch.rand(4, 4, generator=generator, dtype=torch.float64) * 2 - 1
    W = bf16(W_given)
    y = torch.tensor(LINEAR_Y)
    rows = torch.arange(len(LINEAR_Y))
    sx, fx, qx, kx, xq = fake_quant(x, 1, LINEAR_B)
    sw, fw, qw, kw, Wq = fake_quant(W, LINEAR_B, LINEAR_B)
    z = bf16(xq @ Wq.T)
    zr = bf16(x @ W.T)
    logp = fp32(F.log_softmax(z, -1))
    loss = fp32(-logp[rows, y].mean())
    g_logp = torch.zeros_like(logp)
    g_logp[rows, y] = -1.0 / len(LINEAR_Y)
    g_z = g_logp - torch.exp(logp) * g_logp.sum(-1, keepdim=True)
    g_x = straight_through(g_z @ Wq, fx, kx)
    g_W = straight_through(g_z.T @ xq, fw, kw)
    return {
        "b": LINEAR_B,
        "rounding": "each op is computed in float64 and its output rounded to the node dtype (bf16 through fp32, fp32), inputs and weights too; fp32 scales and fp8 codes are already representable; gradients are not rounded",
        "x": encode(torch.tensor(LINEAR_X, dtype=torch.float64)),
        "W": encode(W_given),
        "y": LINEAR_Y,
        "sx": encode(sx),
        "qx": encode(qx),
        "xq": encode(xq),
        "sw": encode(sw),
        "qw": encode(qw),
        "Wq": encode(Wq),
        "z": encode(z),
        "zr": encode(zr),
        "logp": encode(logp),
        "loss": loss.item(),
        "grads": {"x": encode(g_x), "W": encode(g_W)},
    }


def main():
    rounding = {
        "fp8_e4m3": float8_rounding(torch.float8_e4m3fn, 448.0, 2.0**-9, [1.0625, 1.1875, 2.0**-8 * 1.5, 13.0, 460.0]),
        "fp8_e5m2": float8_rounding(torch.float8_e5m2, 57344.0, 2.0**-16, [1.125, 1.375, 2.0**-15 * 1.5, 0.3, 59000.0]),
        "fp4_e2m1": e2m1_rounding(),
    }
    payload = {
        "torch": torch.__version__,
        "command": COMMAND,
        "seed": SEED,
        "ops": OPS,
        "rounding": rounding,
        "block_source": BLOCK_SOURCE,
        "cases": block_cases(),
        "linear": linear_case(),
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=1) + "\n")
    print(f"{OUT_PATH.name}: {len(payload['cases'])} block cases")


if __name__ == "__main__":
    main()
