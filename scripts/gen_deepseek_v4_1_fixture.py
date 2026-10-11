# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "numpy", "sympy", "pillow"]
#
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
#
# [tool.uv.sources]
# torch = { index = "pytorch-cpu" }
# ///
import hashlib
import importlib
import io
import json
import sys
import tempfile
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

REPO = "deepseek-ai/DeepSeek-V4.1-Flash"
REVISION = "2cba9e42aa026125f3ed06c6d98c1db82f7ca027"
FILES = {
    "model.py": "4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65",
    "vision.py": "5d49edc196a4ef22384abe76d35a40098cbe1e74b586c8f66a2edff4f076b26c",
    "engram.py": "11f35ecbead8150c35aa002b3d180ef290b05a25afe883a11884f94d476d3897",
    "image_processor.py": "482759e3bcc4e9bb5ee582b244cc563f5d0e163d8b48dda91ebb7106e62f9272",
}
WIDEN = [
    (".float()", ".double()"),
    ("torch.float32", "torch.float64"),
    ("torch.bfloat16", "torch.float64"),
    ("np.float32", "np.float64"),
]
KERNEL = '''
import torch


def act_quant(x, block_size=128, scale_fmt=None, scale_dtype=torch.float32, inplace=False):
    assert inplace
    return x


def fp4_act_quant(x, block_size=32, inplace=False, scale_dtype=None):
    assert inplace
    return x


def fp8_gemm(*args, **kwargs):
    raise NotImplementedError


def fp4_gemm(*args, **kwargs):
    raise NotImplementedError


def sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale):
    b, m, h, d = q.shape
    valid = topk_idxs >= 0
    idx = topk_idxs.clamp_min(0).long()
    gathered = torch.gather(kv.unsqueeze(1).expand(b, m, kv.size(1), d), 2, idx.unsqueeze(-1).expand(b, m, idx.size(-1), d))
    scores = torch.einsum("bmhd,bmkd->bmhk", q, gathered) * softmax_scale
    scores = scores.masked_fill(~valid.unsqueeze(2), float("-inf"))
    sink = attn_sink.view(1, 1, h, 1).expand(b, m, h, 1).to(scores.dtype)
    probs = torch.softmax(torch.cat([scores, sink], dim=-1), dim=-1)[..., :-1]
    return torch.einsum("bmhk,bmkd->bmhd", probs, gathered)


def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult=4, sinkhorn_iters=20, eps=1e-6):
    hc = hc_mult
    pre = torch.sigmoid(mixes[..., :hc] * hc_scale[0] + hc_base[:hc]) + eps
    post = 2 * torch.sigmoid(mixes[..., hc : 2 * hc] * hc_scale[1] + hc_base[hc : 2 * hc])
    comb = (mixes[..., 2 * hc :] * hc_scale[2] + hc_base[2 * hc :]).unflatten(-1, (hc, hc))
    comb = comb.softmax(-1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return pre, post, comb
'''

SEED = 20261011
COMMAND = "uv run scripts/gen_deepseek_v4_1_fixture.py"
OUT = Path(__file__).resolve().parent.parent / "test" / "fixtures" / "torch" / "deepseek_v4_1.json"

VOCAB = ["<bos>", "<eos>", "<pad>", "The", "the", "cat", "sat", "on", "mat", "<image>"]
TOKEN_MAP = [0, 1, 2, 3, 3, 4, 5, 6, 7, 8]
IMAGE_ID = 9
TEXT_BEFORE = [0, 3, 5]
TEXT_AFTER = [4, 5, 6, 7, 8]
IMAGE_HW = (8, 4)
LAYERS = [0, 1, 2, 20, 24, 25]
SINKHORN_ITERS = 3
SWIGLU_LIMIT = 1.0
ROUTER_BIAS = [0.0, 0.1, -0.05, 0.02]
ROUTER_BIAS_VL = [0.3, -0.2, 0.0, 0.1]
DIMS = {
    "V": len(VOCAB), "d": 4, "H": 2, "D": 6, "rd": 4, "rq": 3, "g": 2, "dg": 2, "W": 4,
    "HI": 2, "cI": 6, "kI": 2, "B": 2, "bs": 2, "N": 2, "E": 4, "K": 2, "f": 2,
    "nmax": 3, "Hg": 2, "dgr": 2, "Ve": 5,
    "dv": 4, "Hv": 1, "fv": 4, "p": 2, "r": 3,
}


def fetch(target):
    for name, digest in FILES.items():
        url = f"https://huggingface.co/{REPO}/resolve/{REVISION}/inference/{name}"
        data = urllib.request.urlopen(url, timeout=60).read()
        if hashlib.sha256(data).hexdigest() != digest:
            raise SystemExit(f"{name} at {REVISION} does not match the recorded sha256")
        text = data.decode()
        for old, new in WIDEN:
            text = text.replace(old, new)
        (target / name).write_text(text)
    (target / "kernel.py").write_text(KERNEL)


def toy_args(M):
    return M.ModelArgs(
        max_batch_size=1, max_seq_len=16, dtype="bf16", expert_dtype=None,
        vocab_size=DIMS["V"], dim=DIMS["d"], moe_inter_dim=DIMS["f"], n_layers=len(LAYERS), n_mtp_layers=0,
        n_heads=DIMS["H"], n_routed_experts=DIMS["E"], n_activated_experts=DIMS["K"], route_scale=1.5,
        swiglu_limit=SWIGLU_LIMIT, q_lora_rank=DIMS["rq"], head_dim=DIMS["D"], rope_head_dim=DIMS["rd"],
        o_groups=DIMS["g"], o_lora_rank=DIMS["dg"], window_size=DIMS["W"],
        compress_ratios=(0, 0, 2, 1, 1, 1), kv_source_layers=(2, 3), index_source_layers=(2, 3, 4),
        compress_rope_theta=160000.0, original_seq_len=65536, rope_theta=10000.0, rope_factor=16, beta_fast=32, beta_slow=1,
        index_n_heads=DIMS["HI"], index_head_dim=DIMS["cI"], index_topk=DIMS["kI"],
        candidate_source_layer=3, candidate_topk_blocks=DIMS["B"], candidate_block_size=DIMS["bs"],
        hc_mult=DIMS["N"], hc_sinkhorn_iters=SINKHORN_ITERS, hc_eps=1e-6,
        engram_layer_ids=(1,), engram_num_embeddings=(0,), engram_max_ngram_size=DIMS["nmax"],
        engram_vocab_size=DIMS["Ve"], engram_n_heads=DIMS["Hg"], engram_head_dim=DIMS["dgr"], engram_pad_id=2,
        engram_compressed_vocab_size=max(TOKEN_MAP) + 1,
        vision_n_layers=1, vision_dim=DIMS["dv"], vision_n_heads=DIMS["Hv"], vision_inter_dim=DIMS["fv"],
        vision_patch_size=DIMS["p"], vision_downsample_ratio=DIMS["r"], vision_max_n_token=64, vision_min_pixels=0,
        image_token_id=IMAGE_ID, dspark_block_size=0,
    )


def plain_engram_lookup(self, indices):
    return F.embedding(indices, self.weight)


def fill(model, generator):
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.numel() == 0:
                continue
            if name.endswith("norm.weight") or "norm1" in name or "norm2" in name or name.endswith("_weight"):
                p.copy_(1 + torch.randn(p.shape, generator=generator) * 0.1)
            else:
                p.copy_(torch.randn(p.shape, generator=generator) * 0.6)
        for layer in model.layers:
            layer.ffn.gate.bias.copy_(torch.tensor(ROUTER_BIAS))
            layer.ffn.gate.bias_vl.copy_(torch.tensor(ROUTER_BIAS_VL))


def page_paths(model):
    paths = {
        "E": model.embed.weight,
        "W": model.head.weight,
        "norm.weight": model.norm.weight,
        "image_start": model.image_start,
        "image_end": model.image_end,
        "image_newline": model.image_newline,
        "vision.Wpe": model.vision.patch_embed.proj.weight,
        "vision.bpe": model.vision.patch_embed.proj.bias,
        "vision.norm.weight": model.vision.norm.weight,
        "aligner.W1": model.aligner.w1.weight,
        "aligner.b1": model.aligner.w1.bias,
        "aligner.W2": model.aligner.w2.weight,
        "aligner.b2": model.aligner.w2.bias,
    }
    block = model.vision.blocks[0]
    paths.update({
        "vision.block.norm1.weight": block.norm1.weight,
        "vision.block.attn.Wqkv": block.attn.wqkv.weight,
        "vision.block.attn.bqkv": block.attn.wqkv.bias,
        "vision.block.attn.Wo": block.attn.wo.weight,
        "vision.block.attn.bo": block.attn.wo.bias,
        "vision.block.norm2.weight": block.norm2.weight,
        "vision.block.mlp.W1": block.mlp.w1.weight,
        "vision.block.mlp.W2": block.mlp.w2.weight,
    })
    for real, layer in zip(LAYERS, model.layers):
        name = f"layer{real}"
        attn, ffn = layer.attn, layer.ffn
        paths.update({
            f"{name}.hc_attn.fn": layer.hc_attn_fn,
            f"{name}.hc_attn.base": layer.hc_attn_base,
            f"{name}.hc_attn.scale": layer.hc_attn_scale,
            f"{name}.hc_ffn.fn": layer.hc_ffn_fn,
            f"{name}.hc_ffn.base": layer.hc_ffn_base,
            f"{name}.hc_ffn.scale": layer.hc_ffn_scale,
            f"{name}.attn_norm.weight": layer.attn_norm.weight,
            f"{name}.ffn_norm.weight": layer.ffn_norm.weight,
            f"{name}.attn.sink": attn.attn_sink,
            f"{name}.attn.Wqa": attn.wq_a.weight,
            f"{name}.attn.q_norm.weight": attn.q_norm.weight,
            f"{name}.attn.Wqb": attn.wq_b.weight,
            f"{name}.attn.Wkv": attn.wkv.weight,
            f"{name}.attn.kv_norm.weight": attn.kv_norm.weight,
            f"{name}.attn.Woa": attn.wo_a.weight,
            f"{name}.attn.Wob": attn.wo_b.weight,
            f"{name}.ffn.gate.Wr": ffn.gate.weight,
            f"{name}.ffn.gate.bias": ffn.gate.bias,
            f"{name}.ffn.gate.bias_vl": ffn.gate.bias_vl,
            f"{name}.ffn.experts.W1": torch.stack([e.w1.weight for e in ffn.experts]),
            f"{name}.ffn.experts.W3": torch.stack([e.w3.weight for e in ffn.experts]),
            f"{name}.ffn.experts.W2": torch.stack([e.w2.weight for e in ffn.experts]),
            f"{name}.ffn.shared_experts.Wg": ffn.shared_experts.w1.weight,
            f"{name}.ffn.shared_experts.Wu": ffn.shared_experts.w3.weight,
            f"{name}.ffn.shared_experts.Wd": ffn.shared_experts.w2.weight,
        })
        if attn.compressor is not None:
            paths[f"{name}.attn.compressor.Wkv"] = attn.compressor.wkv.weight
            paths[f"{name}.attn.compressor.kv_norm.weight"] = attn.compressor.norm.weight
            if attn.compressor.compress_ratio > 1:
                paths[f"{name}.attn.compressor.Wgate"] = attn.compressor.wgate.weight
        if attn.indexer is not None:
            paths[f"{name}.attn.indexer.Wqb"] = attn.indexer.wq_b.weight
            paths[f"{name}.attn.indexer.Ww"] = attn.indexer.weights_proj.weight
            if attn.indexer.owns_k:
                paths[f"{name}.attn.indexer.Wk"] = attn.indexer.wk.weight
                paths[f"{name}.attn.indexer.k_norm.weight"] = attn.indexer.k_norm.weight
        if layer.engram is not None:
            paths.update({
                f"{name}.engram.table": layer.engram.embed.weight,
                f"{name}.engram.Wkv": layer.engram.wkv.weight,
                f"{name}.engram.q_weight": layer.engram.q_weight,
                f"{name}.engram.k_weight": layer.engram.k_weight,
            })
    return paths


def stacked_grad(paths, name):
    return name.endswith(("experts.W1", "experts.W2", "experts.W3"))


def encode(t):
    return {"shape": list(t.shape), "values": [float(v) for v in t.detach().reshape(-1)]}


def encode_int(t):
    return {"shape": list(t.shape), "values": [str(int(v)) for v in t.reshape(-1)]}


def image_png(generator):
    h, w = IMAGE_HW
    pixels = torch.randint(0, 256, (h, w, 3), generator=generator, dtype=torch.uint8)
    buffer = io.BytesIO()
    Image.fromarray(pixels.numpy(), "RGB").save(buffer, format="PNG")
    return pixels, buffer.getvalue()


def run(source):
    sys.path.insert(0, str(source))
    engram = importlib.import_module("engram")
    engram.build_compressed_token_map = lambda tokenizer: (list(TOKEN_MAP), max(TOKEN_MAP) + 1)
    engram.NgramHashState.forward = engram.NgramHashState.forward.__wrapped__
    M = importlib.import_module("model")
    ip = importlib.import_module("image_processor")
    torch.set_default_dtype(torch.float64)
    args = toy_args(M)
    layout = engram.EngramLayout.from_args(args)
    args.engram_num_embeddings = (sum(p for per in layout.primes[0] for p in per),)
    M.ParallelEngramEmbedding.forward = plain_engram_lookup
    model = M.Transformer(args, tokenizer=object()).double()
    generator = torch.Generator().manual_seed(SEED)
    fill(model, generator)

    pixels, png = image_png(generator)
    patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w = ip.load_image({"data": png}, args)
    types = ip.image_token_types(n_llm_h, n_llm_w)
    span = types.numel()
    sequence = TEXT_BEFORE + [IMAGE_ID] * span + TEXT_AFTER
    ids = torch.tensor([sequence[:-1]])
    labels = torch.tensor(sequence[1:])
    token_types = torch.full_like(ids, ip.TEXT)
    token_types[0, len(TEXT_BEFORE) : len(TEXT_BEFORE) + span] = types
    image = ip.ImageInput(start=len(TEXT_BEFORE), patches=patches, n_vit_h=n_vit_h, n_vit_w=n_vit_w, types=types)

    captured = {}
    for real, layer in zip(LAYERS, model.layers):
        layer.register_forward_hook(lambda _m, _i, out, real=real: captured.__setitem__(f"layer{real}", out))
        if layer.attn.indexer is not None:
            layer.attn.indexer.register_forward_hook(lambda _m, _i, out, real=real: captured.__setitem__(f"top{real}", out))
        if layer.engram is not None:
            layer.engram.register_forward_hook(lambda _m, _i, out, real=real: captured.__setitem__(f"engram{real}", out))
    model.engram_hash.register_forward_hook(lambda _m, _i, out: captured.__setitem__("hashes", out))
    model.vision.register_forward_hook(lambda _m, _i, out: captured.__setitem__("vision", out))
    model.aligner.register_forward_hook(lambda _m, _i, out: captured.__setitem__("aligner", out))
    model.norm.register_forward_hook(lambda _m, _i, out: captured.__setitem__("final_norm", out))
    merge = model.merge_image_embeddings

    def merge_and_keep(images, h):
        merge(images, h)
        captured["merged"] = h.clone()

    model.merge_image_embeddings = merge_and_keep
    M.Transformer.encode_image = M.Transformer.encode_image.__wrapped__
    head = M.ParallelHead.forward
    M.ParallelHead.forward = lambda self, x, full_logits=False: head(self, x, full_logits=True)
    _, logits, _ = M.Transformer.forward.__wrapped__(model, ids, 0, images=[[image]], token_types=token_types)
    captured["candidates"] = M.shared_attn.candidates.clone()
    logp = F.log_softmax(logits[0], -1)
    loss = F.nll_loss(logp, labels)
    loss.backward()

    paths = page_paths(model)
    weights = {path: encode(p) for path, p in paths.items()}
    grads = {}
    for real, layer in zip(LAYERS, model.layers):
        for which in ("W1", "W2", "W3"):
            experts = layer.ffn.experts
            attr = which.lower()
            grads[f"layer{real}.ffn.experts.{which}"] = encode(torch.stack([
                getattr(e, attr).weight.grad if getattr(e, attr).weight.grad is not None else torch.zeros_like(getattr(e, attr).weight)
                for e in experts
            ]))
    for path, p in paths.items():
        if path in grads:
            continue
        grads[path] = encode(p.grad if p.grad is not None else torch.zeros_like(p))
    without_grad = sorted(path for path, p in paths.items() if not stacked_grad(paths, path) and p.grad is None)
    hash_state = model.engram_hash
    nodes = {
        "vision": encode(captured["vision"]),
        "aligner": encode(captured["aligner"]),
        "merged": encode(captured["merged"][0]),
        "engram1": encode(captured["engram1"][0]),
        "final_norm": encode(captured["final_norm"][0]),
        "logits": encode(logits[0]),
    }
    for real in LAYERS:
        X, pre = captured[f"layer{real}"]
        nodes[f"X{real}"] = encode(X[0])
        nodes[f"pre{real}"] = encode(pre[0])
    selected = {f"layer{real}": captured[f"top{real}"][0].tolist() for real in (2, 20, 24)}
    return {
        "torch": torch.__version__,
        "repo": REPO,
        "revision": REVISION,
        "command": COMMAND,
        "seed": SEED,
        "dims": DIMS,
        "layers": LAYERS,
        "sinkhorn_iters": SINKHORN_ITERS,
        "swiglu_limit": SWIGLU_LIMIT,
        "vocab": VOCAB,
        "token_map": TOKEN_MAP,
        "ids": ids[0].tolist(),
        "labels": labels.tolist(),
        "token_types": token_types[0].tolist(),
        "pixels": pixels.tolist(),
        "patches": encode(patches.reshape(patches.size(0), -1)),
        "grid": {"n_vit_h": n_vit_h, "n_vit_w": n_vit_w, "n_llm_h": n_llm_h, "n_llm_w": n_llm_w},
        "multipliers": encode_int(hash_state.multipliers[0]),
        "primes": encode_int(hash_state.primes[0]),
        "offsets": encode_int(hash_state.offsets[0]),
        "pad_id": int(hash_state.pad_id),
        "hashes": encode_int(captured["hashes"][0, :, 0, :]),
        "router_bias": ROUTER_BIAS,
        "router_bias_vl": ROUTER_BIAS_VL,
        "params_without_grad": without_grad,
        "weights": weights,
        "loss": loss.item(),
        "nodes": nodes,
        "selected": selected,
        "candidates": captured["candidates"][0].int().tolist(),
        "grads": grads,
    }


def main():
    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp)
        fetch(source)
        payload = run(source)
    OUT.write_text(json.dumps(payload, indent=1) + "\n")
    print(f"{OUT.name}: loss {payload['loss']:.6f}, without grad {payload['params_without_grad']}")
    print("selected", payload["selected"])


if __name__ == "__main__":
    main()
