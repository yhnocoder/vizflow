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
from transformers import GenerationConfig
from transformers.generation.logits_process import TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper

OUT = Path(__file__).resolve().parent.parent / "test" / "fixtures" / "torch" / "sampling.json"
SETTINGS = [
    {"temperature": 0.7},
    {"temperature": 1.2, "top_k": 4},
    {"temperature": 1.0, "top_p": 0.8},
    {"temperature": 1.2, "top_k": 4, "top_p": 0.9},
    {"temperature": 0.5, "top_k": 2, "top_p": 0.3},
]


def main() -> None:
    torch.manual_seed(20261011)
    rows = torch.log_softmax(torch.randn(6, 7, dtype=torch.float64) * 1.5, dim=-1)
    cases = []
    for setting in SETTINGS:
        config = GenerationConfig(do_sample=True, **setting)
        warpers = [TemperatureLogitsWarper(config.temperature)]
        if "top_k" in setting:
            warpers.append(TopKLogitsWarper(setting["top_k"]))
        if "top_p" in setting:
            warpers.append(TopPLogitsWarper(setting["top_p"]))
        scores = rows.clone()
        for warper in warpers:
            scores = warper(None, scores)
        probs = torch.softmax(scores, dim=-1)
        cases.append({"setting": setting, "q": probs.tolist()})
    OUT.write_text(json.dumps({"logp": rows.tolist(), "cases": cases}))


if __name__ == "__main__":
    main()
