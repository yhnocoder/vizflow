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
    cases += pointwise_cases("alias", lambda x: x.clone(), randn)
    cases += rotate_half_cases()
    cases += head_cases()
    cases += linear_attention_cases()
    cases += routing_cases()
    cases += chunk_delta_rule_cases()
    cases += hash_and_mask_cases()
    cases += pointwise_cases("reciprocal", torch.reciprocal, positive)
    cases += pointwise_cases("erf", torch.erf, lambda *shape: randn(*shape) * 2)
    cases += group_mask_cases()
    return cases


def group_mask_cases():
    cases = []
    for shape, dim, index in [((3, 4), 1, [[2, 0], [1, 3], [3, 2]]), ((4, 3), 0, [[1, 0, 3]]), ((2, 3, 2), -1, [[[1], [0], [1]], [[0], [0], [1]]])]:
        cases.append(op_case("scatter", f"{shape} dim {dim} index {index} value 1", lambda x, i, dim=dim: x.scatter(dim, i, 1.0), [("input", randn(*shape)), ("label", torch.tensor(index))], [tensor_arg(0), dim, tensor_arg(1), 1.0]))
    for shape, sizes in [((3, 1), (-1, 4)), ((2, 1, 3), (2, 2, 5, 3)), ((3, 2, 1), (-1, -1, 4))]:
        cases.append(op_case("expand", f"{shape} to {sizes}", lambda x, sizes=sizes: x.expand(*sizes), [("input", randn(*shape))], [tensor_arg(0), *sizes]))
    for shape, sizes in [((3, 2, 4), (-1, 8)), ((2, 6), (3, 2, 2))]:
        cases.append(op_case("reshape", f"{shape} to {sizes}", lambda x, sizes=sizes: x.reshape(*sizes), [("input", randn(*shape))], [tensor_arg(0), *sizes]))
    cases.append(op_case("logical_not", "bool (3, 4)", torch.logical_not, [("label", randint(2, (3, 4)).bool())], [tensor_arg(0)]))
    return cases


def hash_and_mask_cases():
    cases = []
    for a, b in [((3, 4), (3, 4)), ((3, 4), (4,))]:
        cases.append(op_case("remainder", f"float {a} by {b}", torch.remainder, [("input", randn(*a) * 4), ("input", positive(*b) * (randn(*b).sign()))], [tensor_arg(0), tensor_arg(1)]))
    ints = randint(1 << 20, (5, 1)) - (1 << 19)
    cases.append(op_case("remainder", "int64 (5, 1) by primes (1, 3)", torch.remainder, [("label", ints), ("label", torch.tensor([[13, 17, 19]]))], [tensor_arg(0), tensor_arg(1)]))
    cases.append(op_case("remainder", "int64 by a scalar", lambda x: torch.remainder(x, 7), [("label", ints)], [tensor_arg(0), 7]))
    left, right = randint(1 << 30, (4, 3)), randint(1 << 30, (4, 3))
    cases.append(op_case("bitwise_xor", "int64 (4, 3)", torch.bitwise_xor, [("label", left), ("label", right)], [tensor_arg(0), tensor_arg(1)]))
    cases.append(op_case("bitwise_xor", "int64 with a scalar", lambda x: torch.bitwise_xor(x, 5), [("label", left)], [tensor_arg(0), 5]))
    p, q = randint(2, (3, 4)).bool(), randint(2, (4,)).bool()
    cases.append(op_case("bitwise_and", "bool (3, 4) and (4,)", torch.bitwise_and, [("label", p), ("label", q)], [tensor_arg(0), tensor_arg(1)]))
    cases.append(op_case("bitwise_and", "int64 (4, 3)", torch.bitwise_and, [("label", left), ("label", right)], [tensor_arg(0), tensor_arg(1)]))
    small = randint(4, (3, 4))
    cases.append(op_case("eq", "int64 with a scalar", lambda x: torch.eq(x, 2), [("label", small)], [tensor_arg(0), 2]))
    cases.append(op_case("eq", "int64 (3, 4) and (4,)", torch.eq, [("label", small), ("label", randint(4, (4,)))], [tensor_arg(0), tensor_arg(1)]))
    cases.append(op_case("ge", "int64 with a scalar", lambda x: torch.ge(x, 2), [("label", small)], [tensor_arg(0), 2]))
    cases.append(op_case("ge", "float (3, 4) and (3, 1)", torch.ge, [("label", randn(3, 4)), ("label", randn(3, 1))], [tensor_arg(0), tensor_arg(1)]))
    cases += pointwise_cases("abs", torch.abs, randn)
    cases += pointwise_cases("sign", torch.sign, randn)
    x = torch.tensor([-2.0, -1e-6, 0.0, 1e-6, 2.5, 1e-7], dtype=torch.float64)
    cases.append(op_case("clamp_min", "around the bound 1e-6", lambda v: v.clamp_min(1e-6), [("input", x)], [tensor_arg(0), 1e-6]))
    cases.append(op_case("clamp_min", "(3, 4) at 0", lambda v: v.clamp_min(0.0), [("input", randn(3, 4))], [tensor_arg(0), 0.0]))
    for shape, sizes in [((3, 4), (1, 2)), ((2, 3), (2, 1, 2)), ((4,), (3,))]:
        cases.append(op_case("repeat", f"{shape} by {sizes}", lambda v, sizes=sizes: v.repeat(*sizes), [("input", randn(*shape))], [tensor_arg(0), *sizes]))
    for shape, dim in [((3, 5), 1), ((4, 2, 3), 0)]:
        cases.append(op_case("cummax", f"{shape} dim {dim}", lambda v, dim=dim: torch.cummax(v, dim).values, [("input", distinct(*shape))], [tensor_arg(0), dim]))
    tied = torch.tensor([[1.0, 3.0, 3.0, 2.0, 3.0]], dtype=torch.float64)
    cases.append(op_case("cummax", "ties pass the gradient to the latest maximum", lambda v: torch.cummax(v, 1).values, [("input", tied)], [tensor_arg(0), 1]))
    for T, C, K, d in [(7, 3, 4, 2), (5, 2, 3, 3), (3, 4, 4, 3)]:
        cases.append(op_case("causal_conv1d", f"T {T} channels {C} kernel {K} dilation {d}", lambda x, w, d=d: dilated_conv1d(x, w, d), [("input", randn(T, C)), ("input", randn(C, K))], [tensor_arg(0), tensor_arg(1)], {"dilation": d}))
    return cases


def dilated_conv1d(x, w, dilation):
    T, C = x.shape
    K = w.shape[1]
    pad = (K - 1) * dilation
    return F.conv1d(F.pad(x.T.unsqueeze(0), (pad, 0)), w.unsqueeze(1), groups=C, dilation=dilation)[0].T


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
    cases += expert_routing_cases()
    return cases


def grouped_linear(x, weight, group):
    return torch.stack([F.linear(x[r], weight[group[r]]) for r in range(x.shape[0])])


def expert_routing_cases():
    cases = []
    for shape, other in [((3, 4), 0.0), ((5,), 1.0)]:
        cases.append(op_case("greater", f"{shape} > {other}", lambda x, other=other: torch.greater(x, other), [("input", randn(*shape))], [tensor_arg(0), other]))
    cases.append(op_case("greater", "(2, 3) > (3,)", torch.greater, [("input", randn(2, 3)), ("input", randn(3))], [tensor_arg(0), tensor_arg(1)]))
    for shape, k, dim in [((3, 6), 2, -1), ((3, 2, 4), 3, -1), ((4, 3), 2, 0)]:
        cases.append(op_case("topk_indices", f"{shape} k {k} dim {dim}", lambda x, k=k, dim=dim: x.topk(k, dim)[1], [("label", distinct(*shape))], [tensor_arg(0), k, dim]))
    for shape, classes in [((3, 2), 5), ((4,), 4)]:
        cases.append(op_case("one_hot", f"{shape} into {classes} classes", lambda t, classes=classes: F.one_hot(t, classes), [("label", randint(classes, shape))], [tensor_arg(0), classes]))
    for shape, dims in [((3, 2, 5), (2, 1, 0)), ((2, 3), (1, 0)), ((2, 3, 4), (0, -1, 1))]:
        cases.append(op_case("permute", f"{shape} dims {dims}", lambda x, dims=dims: x.permute(*dims), [("input", randn(*shape))], [tensor_arg(0), *dims]))
    mask = torch.tensor([[[1, 0, 0], [0, 0, 1]], [[0, 1, 0], [1, 0, 0]], [[0, 0, 0], [0, 0, 0]], [[0, 1, 1], [0, 0, 0]]])
    for size, fill in [(6, None), (4, None), (9, None), (8, 7)]:
        options = {"size": size} if fill is None else {"size": size, "fill_value": fill}
        cases.append(op_case("nonzero_static", f"(4, 2, 3) size {size}" + ("" if fill is None else f" fill {fill}"), lambda m, options=options: torch.nonzero_static(m, **options), [("label", mask.clone())], [tensor_arg(0)], options))
    rows = torch.tensor([2, 0, 2, -3, 1])
    cols = torch.tensor([1, 0, 1, 2, -1])
    cases.append(op_case("index", "(4, 3)[rows]", lambda x, i: x[i], [("input", randn(4, 3)), ("label", rows.clone())], [tensor_arg(0), [tensor_arg(1)]]))
    cases.append(op_case("index", "(3, 3)[rows, cols]", lambda x, i, j: x[i, j], [("input", randn(3, 3)), ("label", rows.clone().clamp(min=-3, max=2)), ("label", cols.clone())], [tensor_arg(0), [tensor_arg(1), tensor_arg(2)]]))
    cases.append(op_case("index", "(3, 2, 4)[rows, cols]", lambda x, i, j: x[i, j], [("input", randn(3, 2, 4)), ("label", torch.tensor([0, 2, 2, 1])), ("label", torch.tensor([1, 0, 1, 1]))], [tensor_arg(0), [tensor_arg(1), tensor_arg(2)]]))
    for shape, dim, index, source in [((4, 3), 0, [2, 0, 2, 3, 2], (5, 3)), ((2, 4), 1, [3, 3, 0], (2, 3)), ((3,), 0, [1, 2, 1, 0], (4,))]:
        cases.append(
            op_case(
                "index_add",
                f"{shape} dim {dim} index {index}",
                lambda x, i, s, dim=dim: torch.index_add(x, dim, i, s),
                [("input", randn(*shape)), ("label", torch.tensor(index)), ("input", randn(*source))],
                [tensor_arg(0), dim, tensor_arg(1), tensor_arg(2)],
            )
        )
    for rows, groups, outs, width, group in [(6, 4, 4, 3, [0, 0, 1, 3, 3, 3]), (3, 2, 2, 2, [1, 0, 1])]:
        cases.append(
            op_case(
                "grouped_linear",
                f"{rows} rows over {groups} groups of ({outs}, {width})",
                grouped_linear,
                [("input", randn(rows, width)), ("input", randn(groups, outs, width)), ("label", torch.tensor(group))],
                [tensor_arg(0), tensor_arg(1), tensor_arg(2)],
            )
        )
    return cases


def lower_system(n, batch=()):
    return torch.tril(randn(*batch, n, n) * 0.4) + torch.diag_embed(positive(*batch, n))


def chunk_delta_rule_cases():
    cases = []
    for a, b in [((2, 3, 4), (2, 4, 5)), ((3, 4), (2, 4, 2)), ((2, 1, 3, 2), (3, 2, 2))]:
        cases.append(op_case("matmul", f"batched {a} x {b}", torch.matmul, [("input", randn(*a)), ("input", randn(*b))], [tensor_arg(0), tensor_arg(1)]))
    for shape, dim in [((4,), 0), ((3, 4), 1), ((2, 3, 4), -2)]:
        cases.append(op_case("cumsum", f"{shape} dim {dim}", lambda x, dim=dim: torch.cumsum(x, dim), [("input", randn(*shape))], [tensor_arg(0), dim]))
    for shape, dim in [((3, 4), -1), ((3, 4), 0), ((2, 3, 4), 2)]:
        cases.append(op_case("unsqueeze", f"{shape} dim {dim}", lambda x, dim=dim: torch.unsqueeze(x, dim), [("input", randn(*shape))], [tensor_arg(0), dim]))
    for shape, pad in [((3, 4), [0, 0, 0, 1]), ((2, 3), [1, 2]), ((2, 3, 2), [0, 1, 2, 0])]:
        cases.append(op_case("pad", f"{shape} pad {pad}", lambda x, pad=pad: F.pad(x, tuple(pad)), [("input", randn(*shape))], [tensor_arg(0), pad]))
    for shape, diagonal in [((4, 4), 1), ((3, 4), 0), ((2, 3, 3), -1)]:
        cases.append(op_case("triu", f"{shape} diagonal {diagonal}", lambda x, diagonal=diagonal: torch.triu(x, diagonal), [("input", randn(*shape))], [tensor_arg(0), diagonal]))
    for mask, shape, value in [((3, 4), (3, 4), -2.5), ((4, 4), (2, 4, 4), 0.0), ((4,), (3, 4), 1e9)]:
        hit = (randn(*mask) > 0).to(torch.float64)
        cases.append(
            op_case("masked_fill", f"mask {mask} input {shape} value {value}", lambda m, x, value=value: x.masked_fill(m != 0, value), [("label", hit), ("input", randn(*shape))], [tensor_arg(1), tensor_arg(0), value])
        )
    specs = [
        ("lower (3, 3) by (3, 2)", lower_system(3), randn(3, 2), False, False),
        ("upper (3, 3) by (3, 4)", lower_system(3).transpose(-1, -2).contiguous(), randn(3, 4), True, False),
        ("unitriangular lower (2, 4, 4) by (2, 4, 3)", lower_system(4, (2,)), randn(2, 4, 3), False, True),
    ]
    for title, a, b, upper, unitriangular in specs:
        cases.append(
            op_case(
                "solve_triangular",
                title,
                lambda a, b, upper=upper, unitriangular=unitriangular: torch.linalg.solve_triangular(a, b, upper=upper, unitriangular=unitriangular),
                [("input", a), ("input", b)],
                [tensor_arg(0), tensor_arg(1)],
                {"upper": upper, "unitriangular": unitriangular},
            )
        )
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


def clamp_cases():
    cases = []
    for shape, low, high in [((3, 4), -0.5, 0.5), ((2, 2, 3), None, 0.3), ((5,), -0.2, None)]:
        options = {key: value for key, value in [("min", low), ("max", high)] if value is not None}
        cases.append(op_case("clamp", f"{shape} min {low} max {high}", lambda x, options=options: torch.clamp(x, **options), [("input", randn(*shape))], [tensor_arg(0)], options))
    x = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=torch.float64)
    cases.append(op_case("clamp", "values on the bounds", lambda x: torch.clamp(x, -0.5, 0.5), [("input", x)], [tensor_arg(0)], {"min": -0.5, "max": 0.5}))
    return cases


def integer_and_scatter_cases():
    cases = []
    small = randint(4, (3, 4))
    cases.append(op_case("lt", "int64 with a scalar", lambda x: torch.lt(x, 2), [("label", small)], [tensor_arg(0), 2]))
    cases.append(op_case("lt", "float (3, 4) and (3, 1)", torch.lt, [("label", randn(3, 4)), ("label", randn(3, 1))], [tensor_arg(0), tensor_arg(1)]))
    p, q = randint(2, (3, 4)).bool(), randint(2, (4,)).bool()
    left, right = randint(1 << 30, (4, 3)), randint(1 << 30, (4, 3))
    cases.append(op_case("bitwise_or", "bool (3, 4) and (4,)", torch.bitwise_or, [("label", p), ("label", q)], [tensor_arg(0), tensor_arg(1)]))
    cases.append(op_case("bitwise_or", "int64 (4, 3)", torch.bitwise_or, [("label", left), ("label", right)], [tensor_arg(0), tensor_arg(1)]))
    cases.append(op_case("bitwise_or", "int64 with a scalar", lambda x: torch.bitwise_or(x, 5), [("label", left)], [tensor_arg(0), 5]))
    signed = randint(1 << 20, (5, 1)) - (1 << 19)
    cases.append(op_case("floor_divide", "negative and positive int64 by a scalar", lambda x: torch.floor_divide(x, 3), [("label", signed)], [tensor_arg(0), 3]))
    cases.append(op_case("floor_divide", "int64 (5, 1) by (1, 3) with a negative divisor", torch.floor_divide, [("label", signed), ("label", torch.tensor([[7, -4, 13]]))], [tensor_arg(0), tensor_arg(1)]))
    cases.append(op_case("logical_not", "int64 (3, 4)", torch.logical_not, [("label", small)], [tensor_arg(0)]))
    cases.append(op_case("logical_not", "bool (4,)", torch.logical_not, [("label", q)], [tensor_arg(0)]))
    a = torch.cat([randn(5), torch.zeros(1, dtype=torch.float64)]).reshape(2, 3)
    cases.append(op_case("copysign", "(2, 3) with (2, 3), including a zero input", torch.copysign, [("input", a), ("label", randn(2, 3))], [tensor_arg(0), tensor_arg(1)]))
    cases.append(op_case("copysign", "(2, 3) with (3,) including -0", torch.copysign, [("input", randn(2, 3)), ("label", torch.tensor([1.5, -0.0, -2.0], dtype=torch.float64))], [tensor_arg(0), tensor_arg(1)]))
    cases.append(op_case("copysign", "(4,) with a scalar", lambda x: torch.copysign(x, -1.0), [("input", randn(4))], [tensor_arg(0), -1.0]))
    for shape, dim in [((3, 5), -1), ((4, 3), 0)]:
        cases.append(op_case("sort", f"{shape} dim {dim}", lambda x, dim=dim: torch.sort(x, dim=dim, stable=True).values, [("input", distinct(*shape))], [tensor_arg(0), dim]))
    ties = torch.tensor([[3, 1, 3, 0, 1, 2], [5, 5, 4, 4, 5, 0]])
    cases.append(op_case("sort", "int64 (2, 6) with ties, default dim", lambda x: torch.sort(x, stable=True).values, [("label", ties)], [tensor_arg(0)]))
    for shape, dim, index, source in [((3, 5), 1, [[4, 0], [1, 2], [3, 1]], (3, 2)), ((4, 2), 0, [[2, 0], [0, 3]], (2, 2))]:
        cases.append(
            op_case(
                "scatter",
                f"{shape} dim {dim} index {index}",
                lambda x, i, s, dim=dim: torch.scatter(x, dim, i, s),
                [("input", randn(*shape)), ("label", torch.tensor(index)), ("input", randn(*source))],
                [tensor_arg(0), dim, tensor_arg(1), tensor_arg(2)],
            )
        )
    for shape, dim, index, value in [((3, 5), 1, [[4, 0], [1, 2], [3, 1]], 0.0), ((4, 2), 0, [[2, 0], [0, 3]], -1.5)]:
        cases.append(
            op_case(
                "scatter",
                f"{shape} dim {dim} index {index} value {value}",
                lambda x, i, dim=dim, value=value: torch.scatter(x, dim, i, value),
                [("input", randn(*shape)), ("label", torch.tensor(index))],
                [tensor_arg(0), dim, tensor_arg(1), value],
            )
        )
    mask = torch.tensor([True, False, True, True])
    grid = torch.tensor([[True, False, False], [False, True, True]])
    for title, shape, picked, values in [
        ("(4, 3) rows from (3, 3)", (4, 3), mask, (3, 3)),
        ("(4, 3) one row (3,) broadcast", (4, 3), mask, (3,)),
        ("(2, 3, 2) by a (2, 3) mask from (3, 2)", (2, 3, 2), grid, (3, 2)),
        ("(2, 3, 2) by a (2, 3) mask from (1, 2)", (2, 3, 2), grid, (1, 2)),
    ]:
        cases.append(
            op_case(
                "index_put",
                title,
                lambda x, m, v: x.index_put((m.bool(),), v),
                [("input", randn(*shape)), ("label", picked.clone()), ("input", randn(*values))],
                [tensor_arg(0), tensor_arg(1), tensor_arg(2)],
            )
        )
    keep = (randn(3, 1) > 0).to(torch.float64)
    cases.append(op_case("where", "condition (3, 1) input (3, 4) other tensor (4,)", lambda c, x, y: torch.where(c != 0, x, y), [("label", keep), ("input", randn(3, 4)), ("input", randn(4))], [tensor_arg(0), tensor_arg(1), tensor_arg(2)]))
    cases.append(op_case("pad", "(2, 3) pad [0, 1] value 2.5", lambda x: F.pad(x, (0, 1), value=2.5), [("input", randn(2, 3))], [tensor_arg(0), [0, 1]], {"value": 2.5}))
    cases.append(op_case("unsqueeze", "int64 (2, 3) dim 1", lambda x: torch.unsqueeze(x, 1), [("label", randint(1 << 40, (2, 3)))], [tensor_arg(0), 1]))
    return cases


def floor_divide_cases():
    positions = torch.arange(1, 10).unsqueeze(-1)
    ints = randint(1 << 20, (4, 3)) - (1 << 19)
    return [
        op_case("floor_divide", "int64 positions 1..9 by 4", lambda x: torch.floor_divide(x, 4), [("label", positions)], [tensor_arg(0), 4]),
        op_case("floor_divide", "int64 (4, 3) by a scalar", lambda x: torch.floor_divide(x, 7), [("label", ints)], [tensor_arg(0), 7]),
        op_case("floor_divide", "int64 (4, 3) by (3,)", torch.floor_divide, [("label", ints), ("label", torch.tensor([3, -5, 11]))], [tensor_arg(0), tensor_arg(1)]),
        op_case("floor_divide", "float (3, 4) by a scalar", lambda x: torch.floor_divide(x, 0.7), [("label", randn(3, 4) * 4)], [tensor_arg(0), 0.7]),
    ]


def attention_layout_cases():
    cases = []
    for shape, n_rep in [((2, 3, 4), 2), ((2, 2, 3, 2), 3)]:
        dim = len(shape) - 3
        cases.append(op_case("repeat_kv", f"{shape} n_rep {n_rep}", lambda x, n_rep=n_rep, dim=dim: torch.repeat_interleave(x, n_rep, dim), [("input", randn(*shape))], [tensor_arg(0), n_rep]))
    cases.append(op_case("contiguous", "(2, 3, 4)", lambda x: x.contiguous(), [("input", randn(2, 3, 4))], [tensor_arg(0)]))
    return cases


def trigonometric_cases():
    return unary_cases("cos", torch.cos) + unary_cases("sin", torch.sin)


def creation_and_reverse_cases():
    cases = binary_cases("rsub", torch.rsub) + scalar_operand_cases("rsub", torch.rsub)
    cases.append(op_case("pow", "scalar base 10.0 with exponent (2, 3)", lambda x: torch.pow(10.0, x), [("input", randn(2, 3))], [10.0, tensor_arg(0)]))
    cases.append(op_case("pow", "base (2, 3) with exponent (3,)", torch.pow, [("input", positive(2, 3)), ("input", randn(3))], [tensor_arg(0), tensor_arg(1)]))
    for args in [(5,), (2, 7), (1, 8, 3), (0.5, 2.5, 0.5), (6, 0, -2)]:
        cases.append(op_case("arange", f"arange{args}", lambda args=args: torch.arange(*args), [], list(args)))
    for size, dtype, name in [(4, torch.float32, "fp32"), (3, torch.bfloat16, "bf16")]:
        cases.append(op_case("causal_mask", f"size {size} {name}", lambda size=size, dtype=dtype: torch.full((size, size), torch.finfo(dtype).min, dtype=dtype).triu(1), [], [size, name]))
    cases.append(op_case("zeros", "(2, 3)", lambda: torch.zeros(2, 3), [], [2, 3]))
    cases.append(op_case("ones", "(3, 2)", lambda: torch.ones(3, 2), [], [3, 2]))
    cases.append(op_case("eye", "3", lambda: torch.eye(3), [], [3]))
    cases.append(op_case("zeros_like", "(2, 3)", torch.zeros_like, [("input", randn(2, 3))], [tensor_arg(0)]))
    for diagonal in [0, -1]:
        cases.append(op_case("tril", f"(2, 3, 3) diagonal {diagonal}", lambda x, diagonal=diagonal: torch.tril(x, diagonal), [("input", randn(2, 3, 3))], [tensor_arg(0), diagonal]))
    for shape, src, dim, index in [((2, 3, 2), (2, 2), 1, 2), ((3, 4), (3,), -1, 0)]:
        cases.append(op_case("select_scatter", f"{shape} dim {dim} index {index}", lambda x, v, dim=dim, index=index: torch.select_scatter(x, v, dim, index), [("input", randn(*shape)), ("input", randn(*src))], [tensor_arg(0), tensor_arg(1), dim, index]))
    for shape, src, dim, start, end in [((3, 4), (3, 2), -1, 1, 3), ((4, 2), (2, 2), 0, None, 2)]:
        cases.append(op_case("slice_scatter", f"{shape} dim {dim} {start}:{end}", lambda x, v, dim=dim, start=start, end=end: torch.slice_scatter(x, v, dim, start, end), [("input", randn(*shape)), ("input", randn(*src))], [tensor_arg(0), tensor_arg(1), dim, start, end]))
    return cases


DTYPE_NAMES = {torch.bool: "bool", torch.int32: "int32", torch.int64: "int64", torch.bfloat16: "bf16", torch.float32: "fp32"}


def dtype_case(op, fn, dtypes, args=()):
    inputs = [torch.tensor([1, 2]).to(dtype) for dtype in dtypes]
    return {"op": op, "inputs": [DTYPE_NAMES[d] for d in dtypes], "args": list(args), "output": DTYPE_NAMES[fn(*inputs, *args).dtype]}


def promotion_cases():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    floating = {
        "exp": torch.exp, "log": torch.log, "cos": torch.cos, "sin": torch.sin, "sqrt": torch.sqrt, "rsqrt": torch.rsqrt,
        "reciprocal": torch.reciprocal, "sigmoid": torch.sigmoid, "erf": torch.erf, "logsumexp": torch.logsumexp,
    }
    cases = []
    for op, fn in floating.items():
        args = (-1,) if op == "logsumexp" else ()
        for dtype in [torch.bool, torch.int32, torch.int64, torch.bfloat16]:
            cases.append(dtype_case(op, fn, [dtype], args))
    for pair in [(torch.int64, torch.int64), (torch.bool, torch.bool), (torch.int32, torch.int64), (torch.int64, torch.bfloat16)]:
        cases.append(dtype_case("div", torch.div, list(pair)))
    for op, fn in {"neg": torch.neg, "abs": torch.abs, "relu": torch.relu}.items():
        cases.append(dtype_case(op, fn, [torch.int64]))
    for op, fn in {"add": torch.add, "mul": torch.mul}.items():
        cases.append(dtype_case(op, fn, [torch.int32, torch.int64]))
    torch.set_default_dtype(previous)
    return cases


def write(name, cases):
    payload = {"torch": torch.__version__, "command": COMMAND, "seed": SEED, "cases": cases}
    path = OUT_DIR / name
    path.write_text(json.dumps(payload, indent=1) + "\n")
    print(f"{path.relative_to(OUT_DIR.parent.parent.parent)}: {len(cases)} cases")


def rrms_cases():
    cases = []
    for shape in [(4,), (2, 3), (2, 3, 4)]:
        x = torch.arange(math.prod(shape), dtype=torch.float64).reshape(shape) / 7 - 1
        cases.append(op_case("rrms", f"last axis {shape}", lambda x: torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6), [("input", x)], [tensor_arg(0), 1e-6]))
    return cases


def main():
    torch.set_default_dtype(torch.float64)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ops = op_fixtures()
    pages = page_fixtures()
    stable_ops = stability_op_fixtures()
    stable_pages = stability_page_fixtures()
    transformer_ops = transformer_op_fixtures()
    write("ops.json", ops + stable_ops + transformer_ops + clamp_cases() + floor_divide_cases() + integer_and_scatter_cases() + attention_layout_cases() + trigonometric_cases() + creation_and_reverse_cases())
    write("pages.json", pages + stable_pages)
    write("dtypes.json", promotion_cases())
    write("rrms.json", rrms_cases())


if __name__ == "__main__":
    main()
