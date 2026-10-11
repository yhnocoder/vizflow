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

SEED = 20261010
COMMAND = "uv run scripts/gen_torch_fixtures.py"
OUT_DIR = Path(__file__).resolve().parent.parent / "test" / "fixtures" / "torch"

LMHEAD_H = [0.8, -1.1, 0.5, 1.4]
LMHEAD_W = [
    [0.9, -0.6, 0.3, 0.6],
    [0.2, -0.8, 0.6, 0.2],
    [-0.4, 0.2, 0.9, 0.3],
    [0.5, 0.7, -0.6, -0.2],
    [-0.7, 0.4, -0.3, -0.5],
]
SOFTMAX_X = [
    [2.0, 1.0, 0.2, -0.6, -1.4],
    [-0.5, 1.8, 0.3, 0.9, -1.0],
    [0.4, -0.2, 1.5, 0.1, 0.6],
]
SOFTMAX_T = [0, 1, 2]

generator = torch.Generator().manual_seed(SEED)


def randn(*shape):
    return torch.randn(shape, generator=generator, dtype=torch.float64)


def positive(*shape):
    return torch.rand(shape, generator=generator, dtype=torch.float64) * 1.5 + 0.5


def randint(high, shape):
    return torch.randint(0, high, shape, generator=generator)


def encode_number(v):
    if math.isnan(v):
        return "nan"
    if math.isinf(v):
        return "inf" if v > 0 else "-inf"
    return v


def encode(t):
    return {
        "shape": list(t.shape),
        "values": [encode_number(float(v)) for v in t.detach().reshape(-1).tolist()],
    }


def tensor_arg(index):
    return {"tensor": index}


def op_case(op, title, fn, tensors, args, options=None):
    leaves = []
    for kind, value in tensors:
        if kind == "label":
            leaves.append(value.clone())
        else:
            leaves.append(value.clone().requires_grad_(True))
    out = fn(*leaves)
    upstream = randn(*out.shape)
    if out.requires_grad:
        out.backward(upstream)
    return {
        "op": op,
        "title": title,
        "tensors": [
            {"kind": kind, **encode(leaf)} for (kind, _), leaf in zip(tensors, leaves)
        ],
        "args": args,
        "options": options or {},
        "output": encode(out),
        "upstream": encode(upstream),
        "grads": [
            None if kind == "label" else encode(leaf.grad if leaf.grad is not None else torch.zeros_like(leaf))
            for (kind, _), leaf in zip(tensors, leaves)
        ],
    }


def distinct(*shape):
    n = math.prod(shape)
    values = torch.randperm(n, generator=generator).to(torch.float64) * 0.37 - n * 0.18
    return (values + randn(n) * 0.01).reshape(*shape)


def binary_cases(op, fn):
    a_args = [tensor_arg(0), tensor_arg(1)]
    right = positive if op == "div" else randn
    pairs = [
        ("same shape (2, 3)", randn(2, 3), right(2, 3)),
        ("broadcast (2, 3) with (3,)", randn(2, 3), right(3)),
        ("broadcast (3, 1) with (1, 4)", randn(3, 1), right(1, 4)),
        ("broadcast (2, 1, 3) with (4, 1)", randn(2, 1, 3), right(4, 1)),
        ("broadcast (4,) with scalar ()", randn(4), right()),
    ]
    return [
        op_case(op, title, fn, [("input", a), ("input", b)], a_args)
        for title, a, b in pairs
    ]


def unary_cases(op, fn):
    make = positive if op == "log" else randn
    return [
        op_case(op, f"shape {tuple(shape)}", fn, [("input", make(*shape))], [tensor_arg(0)])
        for shape in [(5,), (2, 3), (2, 2, 3), ()]
    ]


def matmul_cases():
    pairs = [
        ("2D x 2D", randn(2, 3), randn(3, 4)),
        ("1D x 2D", randn(3), randn(3, 4)),
        ("2D x 1D", randn(2, 3), randn(3)),
        ("2D x 2D square", randn(4, 4), randn(4, 4)),
    ]
    return [
        op_case("matmul", title, torch.matmul, [("input", a), ("input", b)], [tensor_arg(0), tensor_arg(1)])
        for title, a, b in pairs
    ]


def transpose_cases():
    specs = [((2, 3), 0, 1), ((2, 3), -1, -2), ((2, 3, 4), 0, 2), ((2, 3, 4), 1, -1), ((3, 3), 1, 1)]
    return [
        op_case(
            "transpose",
            f"shape {shape} dims ({d0}, {d1})",
            lambda x, d0=d0, d1=d1: torch.transpose(x, d0, d1),
            [("input", randn(*shape))],
            [tensor_arg(0), d0, d1],
        )
        for shape, d0, d1 in specs
    ]


def reduce_cases(op, fn, specs, make=randn):
    cases = []
    for shape, dim, keepdim in specs:
        args = [tensor_arg(0)] if dim is None else [tensor_arg(0), dim]
        options = {} if keepdim is None else {"keepdim": keepdim}

        def run(x, dim=dim, keepdim=keepdim):
            kwargs = {} if keepdim is None else {"keepdim": keepdim}
            if dim is None:
                return fn(x, **kwargs) if kwargs else fn(x)
            return fn(x, dim, **kwargs)

        title = f"shape {shape} dim {dim} keepdim {keepdim}"
        cases.append(op_case(op, title, run, [("input", make(*shape))], args, options))
    return cases


def sum_mean_cases(op, fn):
    def run_all(x, keepdim=None):
        return fn(x) if keepdim is None else fn(x, dim=list(range(x.dim())), keepdim=keepdim)

    specs = [
        ((2, 3), 0, None),
        ((2, 3), -1, True),
        ((2, 3, 4), [0, 2], False),
        ((2, 3, 4), [-1, 0], True),
        ((2, 3, 4), 1, False),
        ((4,), 0, True),
    ]
    cases = reduce_cases(op, fn, specs)
    for shape in [(2, 3), (5,)]:
        cases.append(op_case(op, f"shape {shape} all elements", run_all, [("input", randn(*shape))], [tensor_arg(0)]))
    cases.append(
        op_case(
            op,
            "shape (2, 3) all elements keepdim True",
            lambda x: run_all(x, True),
            [("input", randn(2, 3))],
            [tensor_arg(0)],
            {"keepdim": True},
        )
    )
    return cases


def max_cases():
    specs = [((2, 3), 0, None), ((2, 3), -1, True), ((2, 3, 4), 1, False), ((2, 3, 4), -3, True), ((5,), 0, False)]
    return reduce_cases("max", lambda x, d, **kw: torch.max(x, d, **kw).values, specs, distinct)


def logsumexp_cases():
    specs = [((2, 3), 0, None), ((2, 3), -1, True), ((2, 3, 4), [0, 2], False), ((2, 3, 4), 1, True), ((5,), 0, False)]
    return reduce_cases("logsumexp", torch.logsumexp, specs)


def softmax_cases(op, fn):
    specs = [((5,), 0), ((2, 3), -1), ((2, 3), 0), ((2, 3, 4), 1), ((2, 3, 4), -3)]
    return [
        op_case(op, f"shape {shape} dim {dim}", lambda x, dim=dim: fn(x, dim), [("input", randn(*shape) * 2)], [tensor_arg(0), dim])
        for shape, dim in specs
    ]


def gather_cases():
    specs = [
        ((5,), 0, (2,)),
        ((3, 4), 1, (3, 2)),
        ((3, 4), -1, (2, 3)),
        ((3, 4), 0, (5, 4)),
        ((2, 3, 4), 1, (2, 2, 3)),
        ((2, 3, 4), -3, (3, 3, 4)),
    ]
    cases = []
    for shape, dim, index_shape in specs:
        index = randint(shape[dim], index_shape)
        cases.append(
            op_case(
                "gather",
                f"input {shape} dim {dim} index {index_shape}",
                lambda x, i, dim=dim: torch.gather(x, dim, i),
                [("input", randn(*shape)), ("label", index)],
                [tensor_arg(0), dim, tensor_arg(1)],
            )
        )
    return cases


def nll_cases():
    cases = []
    batched = [
        ("no ignored rows", torch.tensor([1, 0, 3, 2])),
        ("some rows ignored", torch.tensor([1, -100, 3, -100])),
        ("all rows ignored", torch.tensor([-100, -100, -100, -100])),
    ]
    for reduction in ["mean", "sum", "none"]:
        for title, target in batched:
            cases.append(
                nll_case(f"(4, 5) {title} reduction {reduction}", randn(4, 5), target, reduction, -100)
            )
        cases.append(nll_case(f"(5,) with target () reduction {reduction}", randn(5), torch.tensor(3), reduction, -100))
        cases.append(nll_case(f"(5,) with ignored target () reduction {reduction}", randn(5), torch.tensor(2), reduction, 2))
    cases.append(nll_case("(3, 4) custom ignore_index 1 reduction mean", randn(3, 4), torch.tensor([1, 0, 1]), "mean", 1))
    cases.append(nll_case("(3, 4) default options", randn(3, 4), torch.tensor([3, 0, 2]), None, None))
    return cases


def nll_case(title, x, target, reduction, ignore_index):
    options = {}
    if reduction is not None:
        options["reduction"] = reduction
    if ignore_index is not None:
        options["ignore_index"] = ignore_index
    return op_case(
        "nll_loss",
        title,
        lambda inp, tgt: F.nll_loss(inp, tgt, **options),
        [("input", x), ("label", target)],
        [tensor_arg(0), tensor_arg(1)],
        options,
    )


def op_fixtures():
    cases = []
    cases += binary_cases("add", torch.add)
    cases += binary_cases("sub", torch.sub)
    cases += binary_cases("mul", torch.mul)
    cases += binary_cases("div", torch.div)
    cases += unary_cases("neg", torch.neg)
    cases += unary_cases("exp", torch.exp)
    cases += unary_cases("log", torch.log)
    cases += matmul_cases()
    cases += transpose_cases()
    cases += sum_mean_cases("sum", torch.sum)
    cases += sum_mean_cases("mean", torch.mean)
    cases += max_cases()
    cases += logsumexp_cases()
    cases += softmax_cases("softmax", torch.softmax)
    cases += softmax_cases("log_softmax", torch.log_softmax)
    cases += gather_cases()
    cases += nll_cases()
    return cases


def leaf(values, requires_grad=True):
    t = torch.tensor(values, dtype=torch.float64)
    return t.requires_grad_(True) if requires_grad else t


def tracked(nodes):
    for t in nodes.values():
        if t.requires_grad and not t.is_leaf:
            t.retain_grad()


def page_record(page, title, inputs, nodes, loss):
    tracked(nodes)
    loss.backward(torch.ones_like(loss))
    return {
        "page": page,
        "title": title,
        "inputs": {name: encode(value) for name, value in inputs.items()},
        "loss": encode(loss),
        "nodes": {
            name: {
                "value": encode(t),
                "grad": None if not t.requires_grad else encode(t.grad if t.grad is not None else torch.zeros_like(t)),
            }
            for name, t in nodes.items()
        },
    }


def lmhead_page(title, h_values, w_values, target):
    h = leaf([h_values])
    W = leaf(w_values)
    t = torch.tensor([target])
    WT = W.T
    z = h @ WT
    lse = torch.logsumexp(z, -1, keepdim=True)
    logp = z - lse
    loss = F.nll_loss(logp, t)
    nodes = {"h": h, "W": W, "t": t, "W.T": WT, "z": z, "LSE": lse, "log p": logp, "loss": loss}
    return page_record("lmhead", title, {"h": h, "W": W, "t": t.to(torch.float64)}, nodes, loss)


def softmax_single_page(title, x_values, target):
    x = leaf(x_values)
    t = torch.tensor([target])
    u = torch.exp(x)
    S = torch.sum(u, -1, keepdim=True)
    p = u / S
    y = torch.gather(p, -1, t)
    logy = torch.log(y)
    loss = torch.neg(logy)
    nodes = {"x": x, "t": t, "u": u, "S": S, "p": p, "y": y, "log y": logy, "l": loss}
    return page_record("softmax-single", title, {"x": x, "t": t.to(torch.float64)}, nodes, loss)


def softmax_batch_page(title, x_values, targets):
    X = leaf(x_values)
    t = torch.tensor([[v] for v in targets])
    U = torch.exp(X)
    S = torch.sum(U, -1, keepdim=True)
    P = U / S
    Y = torch.gather(P, -1, t)
    logY = torch.log(Y)
    lb = torch.neg(logY)
    loss = torch.mean(lb)
    nodes = {"X": X, "t": t, "U": U, "S": S, "P": P, "Y": Y, "log Y": logY, "l_b": lb, "l": loss}
    return page_record("softmax-batch", title, {"X": X, "t": t.to(torch.float64)}, nodes, loss)


def page_fixtures():
    cases = []
    for target in range(5):
        cases.append(lmhead_page(f"default values, target {target}", LMHEAD_H, LMHEAD_W, target))
    for d, V in [(4, 5), (3, 6), (6, 4)]:
        cases.append(
            lmhead_page(f"random d={d} V={V}", randn(1, d)[0].tolist(), randn(V, d).tolist(), int(randint(V, ()).item()))
        )
    for target in range(5):
        cases.append(softmax_single_page(f"default x, target {target}", SOFTMAX_X[0], target))
    for size in [5, 7]:
        cases.append(softmax_single_page(f"random size {size}", (randn(size) * 2).tolist(), int(randint(size, ()).item())))
    cases.append(softmax_batch_page("default values", SOFTMAX_X, SOFTMAX_T))
    for batch, classes in [(3, 5), (4, 6), (1, 5)]:
        cases.append(
            softmax_batch_page(
                f"random B={batch} C={classes}",
                (randn(batch, classes) * 2).tolist(),
                randint(classes, (batch,)).tolist(),
            )
        )
    return cases


STABILITY_INPUTS = [
    ("large logits", [1000.0, 1001.0, 999.0]),
    ("logits with -1000", [-1000.0, 0.0, 1.0]),
    ("rows of large and very negative logits", [[1000.0, 1001.0, 999.0], [-1000.0, 0.0, 1.0]]),
]


def finite(case):
    return {**case, "finite": True}


def stability_op_fixtures():
    cases = []
    for title, values in STABILITY_INPUTS:
        x = torch.tensor(values, dtype=torch.float64)
        cases.append(
            finite(
                op_case(
                    "logsumexp",
                    f"{title} dim -1 keepdim True",
                    lambda a: torch.logsumexp(a, -1, keepdim=True),
                    [("input", x)],
                    [tensor_arg(0), -1],
                    {"keepdim": True},
                )
            )
        )
        for op, fn in [("softmax", torch.softmax), ("log_softmax", torch.log_softmax)]:
            cases.append(finite(op_case(op, f"{title} dim -1", lambda a, fn=fn: fn(a, -1), [("input", x)], [tensor_arg(0), -1])))
    return cases


def stability_page_fixtures():
    cases = []
    for d, V in [(4, 5), (6, 8)]:
        cases.append(
            finite(
                lmhead_page(
                    f"large logits d={d} V={V}",
                    randn(1, d)[0].tolist(),
                    (randn(V, d) * 150).tolist(),
                    int(randint(V, ()).item()),
                )
            )
        )
    return cases


def scalar_operand_cases(op, fn):
    scalars = [0.5, -2.0] if op == "div" else [0.5, -1.5]
    return [
        op_case(op, f"(2, 3) with scalar {s}", lambda x, s=s: fn(x, s), [("input", randn(2, 3))], [tensor_arg(0), s])
        for s in scalars
    ]


def pow_cases():
    specs = [(2, randn), (3, randn), (-0.5, positive), (0.5, positive), (1.5, positive)]
    return [
        op_case("pow", f"shape (2, 3) exponent {e}", lambda x, e=e: torch.pow(x, e), [("input", make(2, 3))], [tensor_arg(0), e])
        for e, make in specs
    ]


def pointwise_cases(op, fn, make):
    return [
        op_case(op, f"shape {tuple(shape)}", fn, [("input", make(*shape))], [tensor_arg(0)])
        for shape in [(5,), (2, 3), ()]
    ]


def embedding_cases():
    specs = [
        ("ids (3,) weight (5, 4)", torch.tensor([2, 4, 3]), (5, 4)),
        ("ids () weight (5, 4)", torch.tensor(1), (5, 4)),
        ("repeated ids (2, 3) weight (6, 2)", torch.tensor([[0, 5, 0], [5, 5, 1]]), (6, 2)),
    ]
    return [
        op_case(
            "embedding",
            title,
            F.embedding,
            [("label", ids), ("input", randn(*weight))],
            [tensor_arg(0), tensor_arg(1)],
        )
        for title, ids, weight in specs
    ]


def narrow_cases():
    specs = [((2, 6), 1, 2, 3), ((5,), 0, 1, 4), ((2, 3, 4), -1, -3, 2), ((4, 3), 0, 0, 4), ((3, 4), 0, 1, 0)]
    return [
        op_case(
            "narrow",
            f"shape {shape} dim {dim} start {start} length {length}",
            lambda x, dim=dim, start=start, length=length: torch.narrow(x, dim, start, length),
            [("input", randn(*shape))],
            [tensor_arg(0), dim, start, length],
        )
        for shape, dim, start, length in specs
    ]


def cat_cases():
    specs = [
        ([(2, 3), (2, 2)], 1, None),
        ([(1, 3), (2, 3), (3, 3)], 0, None),
        ([(2, 2, 3)], -1, None),
        ([(4,), (2,)], None, None),
        ([(2, 1), (2, 3)], None, -1),
    ]
    cases = []
    for shapes, dim, keyword in specs:
        tensors = [("input", randn(*shape)) for shape in shapes]
        refs = [tensor_arg(i) for i in range(len(shapes))]
        args = [refs] if dim is None else [refs, dim]
        options = {} if keyword is None else {"dim": keyword}
        used = dim if dim is not None else (keyword if keyword is not None else 0)
        title = f"shapes {shapes} dim {used}" + (" as option" if keyword is not None else "")
        cases.append(op_case("cat", title, lambda *xs, used=used: torch.cat(xs, used), tensors, args, options))
    return cases


def transformer_op_fixtures():
    cases = []
    cases += scalar_operand_cases("add", torch.add)
    cases += scalar_operand_cases("sub", torch.sub)
    cases += scalar_operand_cases("mul", torch.mul)
    cases += scalar_operand_cases("div", torch.div)
    cases += pow_cases()
    cases += pointwise_cases("rsqrt", torch.rsqrt, positive)
    cases += pointwise_cases("sqrt", torch.sqrt, positive)
    cases += pointwise_cases("sigmoid", torch.sigmoid, lambda *shape: randn(*shape) * 3)
    cases += pointwise_cases("silu", F.silu, lambda *shape: randn(*shape) * 3)
    for op, fn in [("sigmoid", torch.sigmoid), ("silu", F.silu)]:
        x = torch.tensor([-800.0, -30.0, 0.0, 30.0, 800.0], dtype=torch.float64)
        cases.append(finite(op_case(op, "large magnitudes", fn, [("input", x)], [tensor_arg(0)])))
    cases += embedding_cases()
    cases += narrow_cases()
    cases += cat_cases()
    cases += pointwise_cases("relu", F.relu, randn)
    cases += pointwise_cases("gelu", F.gelu, lambda *shape: randn(*shape) * 3)
    cases += pointwise_cases("identity", lambda x: x.clone(), randn)
    cases += rotate_half_cases()
    cases += head_cases()
    cases += linear_attention_cases()
    cases += routing_cases()
    return cases


def topk_mask(x, k, dim):
    return torch.zeros_like(x).scatter(dim, x.topk(k, dim)[1], 1.0)


def routing_cases():
    cases = []
    for shape, k, dim in [((3, 6), 2, -1), ((3, 2, 3), 2, -1), ((4, 3), 1, 0)]:
        cases.append(op_case("topk_mask", f"{shape} k {k} dim {dim}", lambda x, k=k, dim=dim: topk_mask(x, k, dim), [("label", distinct(*shape))], [tensor_arg(0), k, dim]))
    for condition, shape, other in [((3, 4), (3, 4), -2.5), ((3, 1), (3, 4), 0.0), ((4,), (2, 4), -1e9)]:
        keep = (randn(*condition) > 0).to(torch.float64)
        cases.append(
            op_case("where", f"condition {condition} input {shape} other {other}", lambda c, x, other=other: torch.where(c != 0, x, other), [("label", keep), ("input", randn(*shape))], [tensor_arg(0), tensor_arg(1), other])
        )
    return cases


def route_pairs(sel):
    return [(t, e) for e in range(sel.shape[1]) for t in range(sel.shape[0]) if sel[t, e] == 1]


def grouped_mm(x, w, group):
    return torch.stack([w[int(group[r])] @ x[r] for r in range(x.shape[0])])


def sorted_moe_cases():
    cases = []
    for shape, k in [((4, 4), 2), ((3, 6), 3), ((5, 3), 1)]:
        sel = topk_mask(distinct(*shape), k, -1)
        rows = lambda s: torch.tensor([float(t) for t, _ in route_pairs(s)])
        experts = lambda s: torch.tensor([float(e) for _, e in route_pairs(s)])
        cases.append(op_case("route_rows", f"{shape} k {k}", rows, [("label", sel)], [tensor_arg(0), k]))
        cases.append(op_case("route_experts", f"{shape} k {k}", experts, [("label", sel)], [tensor_arg(0), k]))
    for (R, d), (E, n), group in [((5, 3), (3, 2), [0, 0, 2, 2, 2]), ((4, 4), (4, 2), [1, 3, 3, 0]), ((2, 2), (1, 3), [0, 0])]:
        cases.append(
            op_case("grouped_mm", f"x ({R}, {d}) w ({E}, {n}, {d}) group {group}", grouped_mm, [("input", randn(R, d)), ("input", randn(E, n, d)), ("label", torch.tensor(group, dtype=torch.float64))], [tensor_arg(0), tensor_arg(1), tensor_arg(2)])
        )
    for (R, d), size, index in [((5, 2), 4, [1, 0, 1, 3, 1]), ((8, 3), 4, [0, 1, 1, 2, 0, 3, 2, 3]), ((3, 1), 2, [1, 1, 1])]:
        cases.append(
            op_case("index_add", f"src ({R}, {d}) size {size} index {index}", lambda src, idx, size=size: torch.zeros(size, src.shape[1]).index_add(0, idx.long(), src), [("input", randn(R, d)), ("label", torch.tensor(index, dtype=torch.float64))], [tensor_arg(0), tensor_arg(1), size])
        )
    return cases


def scaled_mm(a, sfa, b, sfb, block_k, block_n):
    M, K = a.shape
    N = b.shape[0]
    out = torch.zeros(M, N)
    for j in range(sfa.shape[1]):
        ks = slice(j * block_k, min(K, (j + 1) * block_k))
        scale = sfa[:, j : j + 1] * sfb[:, j].repeat_interleave(block_n)[:N].unsqueeze(0)
        out = out + scale * (a[:, ks] @ b[:, ks].T)
    return out


def scaled_mm_cases():
    cases = []
    for (M, N, K), (block_k, block_n) in [((2, 3, 4), (2, 2)), ((3, 5, 7), (3, 2)), ((2, 4, 6), (128, 128))]:
        k_blocks = -(-K // block_k)
        n_blocks = -(-N // block_n)
        tensors = [("input", randn(M, K)), ("input", positive(M, k_blocks)), ("input", randn(N, K)), ("input", positive(n_blocks, k_blocks))]
        fn = lambda a, sfa, b, sfb, block_k=block_k, block_n=block_n: scaled_mm(a, sfa, b, sfb, block_k, block_n)
        refs = [tensor_arg(i) for i in range(4)]
        if (block_k, block_n) == (128, 128):
            cases.append(op_case("scaled_mm", f"a ({M}, {K}) b ({N}, {K}) default blocks fp32", fn, tensors, refs))
        else:
            cases.append(op_case("scaled_mm", f"a ({M}, {K}) b ({N}, {K}) block_k {block_k} block_n {block_n} fp32", fn, tensors, refs + [block_k, block_n, "fp32"]))
    return cases


def causal_conv1d(x, w):
    T, C = x.shape
    K = w.shape[1]
    return F.conv1d(x.T.unsqueeze(0), w.unsqueeze(1), padding=K - 1, groups=C)[0, :, :T].T


def linear_attention_cases():
    cases = pointwise_cases("softplus", F.softplus, lambda *shape: randn(*shape) * 3)
    x = torch.tensor([-800.0, -30.0, 0.0, 19.0, 21.0, 800.0], dtype=torch.float64)
    cases.append(finite(op_case("softplus", "large magnitudes", F.softplus, [("input", x)], [tensor_arg(0)])))
    for shape, repeats, dim in [((3, 2, 4), 2, 1), ((3, 4), 3, 0), ((2, 1, 3), 3, -2)]:
        cases.append(
            op_case("repeat_interleave", f"{shape} repeats {repeats} dim {dim}", lambda x, repeats=repeats, dim=dim: torch.repeat_interleave(x, repeats, dim), [("input", randn(*shape))], [tensor_arg(0), repeats, dim])
        )
    for T, C, K in [(3, 8, 4), (5, 2, 3), (2, 3, 1)]:
        cases.append(op_case("causal_conv1d", f"T {T} channels {C} kernel {K}", causal_conv1d, [("input", randn(T, C)), ("input", randn(C, K))], [tensor_arg(0), tensor_arg(1)]))
    return cases


def select_matmul(a, b, index):
    return a[:, index, :] @ b


def head_cases():
    cases = []
    for shape, view in [((3, 8), (3, 2, 4)), ((3, 2, 4), (3, -1)), ((6,), (2, 3))]:
        cases.append(op_case("view", f"{shape} as {view}", lambda x, view=view: x.view(*view), [("input", randn(*shape))], [tensor_arg(0), *view]))
    for shape, dim, index in [((3, 2, 4), 1, 1), ((3, 2, 4), -1, -1), ((4,), 0, 2)]:
        cases.append(op_case("select", f"{shape} dim {dim} index {index}", lambda x, dim=dim, index=index: torch.select(x, dim, index), [("input", randn(*shape))], [tensor_arg(0), dim, index]))
    for a, b, index in [((3, 2, 4), (4, 3), 1), ((3, 2, 4), (4, 3), 0), ((2, 3, 2), (2, 4), 2)]:
        cases.append(
            op_case("select_matmul", f"{a} head {index} by {b}", lambda x, y, index=index: select_matmul(x, y, index), [("input", randn(*a)), ("input", randn(*b))], [tensor_arg(0), tensor_arg(1), index])
        )
    return cases


def rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def rotate_half_cases():
    return [
        op_case("rotate_half", f"shape {shape}", rotate_half, [("input", randn(*shape))], [tensor_arg(0)])
        for shape in [(4,), (3, 4), (2, 2, 6), (3, 2, 4)]
    ]


def write(name, cases):
    payload = {"torch": torch.__version__, "command": COMMAND, "seed": SEED, "cases": cases}
    path = OUT_DIR / name
    path.write_text(json.dumps(payload, indent=1) + "\n")
    print(f"{path.relative_to(OUT_DIR.parent.parent.parent)}: {len(cases)} cases")


def main():
    torch.set_default_dtype(torch.float64)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ops = op_fixtures()
    pages = page_fixtures()
    stable_ops = stability_op_fixtures()
    stable_pages = stability_page_fixtures()
    transformer_ops = transformer_op_fixtures()
    sorted_moe_ops = sorted_moe_cases()
    scaled_mm_ops = scaled_mm_cases()
    write("ops.json", ops + stable_ops + transformer_ops + sorted_moe_ops + scaled_mm_ops)
    write("pages.json", pages + stable_pages)


if __name__ == "__main__":
    main()
