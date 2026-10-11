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
import random
from pathlib import Path

import torch

SEED = 20261011
COMMAND = "uv run scripts/gen_rounding_fixture.py"
OUT = Path(__file__).resolve().parent.parent / "test" / "fixtures" / "torch" / "rounding.json"
DTYPES = {
    "fp32": torch.float32,
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp8_e4m3": torch.float8_e4m3fn,
    "fp8_e5m2": torch.float8_e5m2,
}
EDGES = [0.0, -0.0, 464.0, 465.0, 480.0, 1000.0, -1000.0, 2.0**-9, 1.5 * 2.0**-10, 2.0**-10, 65519.0, 65520.0, 57344.0, 60000.0, 61440.0, 1e-30, 1e40, math.inf, -math.inf, math.nan]


def encode(v):
    if math.isnan(v):
        return "nan"
    if math.isinf(v):
        return "inf" if v > 0 else "-inf"
    return v


def main():
    rng = random.Random(SEED)
    values = EDGES + [rng.uniform(-3, 3) * 10.0 ** rng.randint(-9, 5) for _ in range(800)]
    x = torch.tensor(values, dtype=torch.float64)
    cases = [{"dtype": name, "x": [encode(v) for v in values], "y": [encode(v) for v in x.to(dt).to(torch.float64).tolist()]} for name, dt in DTYPES.items()]
    OUT.write_text(json.dumps({"torch": torch.__version__, "command": COMMAND, "seed": SEED, "cases": cases}) + "\n")


if __name__ == "__main__":
    main()
