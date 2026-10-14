import { describe, expect, it } from 'vitest';
import { type Node, Tensor, defineOp, graph } from '../../src/core';
import { createSteps } from '../../src/steps';

const close = (actual: Tensor | null, expected: number[], digits = 12) => {
  expect(actual).not.toBeNull();
  const data = Array.from((actual as Tensor).data);
  expect(data.length).toBe(expected.length);
  data.forEach((v, i) => expect(v).toBeCloseTo(expected[i], digits));
};

describe('elementwise ops with hand-computed values', () => {
  it('pow, rsqrt, sqrt, sigmoid and silu forward and backward', () => {
    const g = graph({ check: true });
    const x = g.input('x', [3], [0.25, 1, 4]);
    const sigmoid = (v: number) => 1 / (1 + Math.exp(-v));
    const cases: [Node, number[], number[]][] = [
      [g.pow(x, 2), [0.0625, 1, 16], [0.5, 2, 8]],
      [g.pow(x, -0.5), [2, 1, 0.5], [-4, -0.5, -0.0625]],
      [g.rsqrt(x), [2, 1, 0.5], [-4, -0.5, -0.0625]],
      [g.sqrt(x), [0.5, 1, 2], [1, 0.5, 0.25]],
      [g.sigmoid(x), [0.25, 1, 4].map(sigmoid), [0.25, 1, 4].map((v) => sigmoid(v) * (1 - sigmoid(v)))],
      [g.silu(x), [0.25, 1, 4].map((v) => v * sigmoid(v)), [0.25, 1, 4].map((v) => sigmoid(v) * (1 + v * (1 - sigmoid(v))))],
    ];
    for (const [node, value, grad] of cases) {
      close(node.value, value);
      close(g.jacobian(node, x), [grad[0], 0, 0, 0, grad[1], 0, 0, 0, grad[2]]);
    }
    expect(g.pow(x, 3).attrs).toEqual({ exponent: 3 });
  });

  it('sigmoid and silu stay finite for large magnitudes', () => {
    const g = graph();
    const x = g.input('x', [3], [-800, 0, 800]);
    close(g.sigmoid(x).value, [0, 0.5, 1]);
    close(g.silu(x).value, [0, 0, 800]);
  });
});

describe('scalar operands', () => {
  it('add, sub, mul and div take a number as the second argument', () => {
    const g = graph({ check: true });
    const x = g.input('x', [2], [1, -2]);
    const before = g.nodes.length;
    const sum = g.add(x, 1e-6);
    expect(g.nodes.length).toBe(before + 1);
    expect(sum.inputs).toEqual([x]);
    expect(sum.attrs).toEqual({ other: 1e-6 });
    close(sum.value, [1.000001, -1.999999]);
    close(g.sub(x, 0.5).value, [0.5, -2.5]);
    close(g.mul(x, 0.5).value, [0.5, -1]);
    close(g.div(x, 4).value, [0.25, -0.5]);
    close(g.jacobian(g.mul(x, -3), x), [-3, 0, 0, -3]);
    close(g.jacobian(g.div(x, 4), x), [0.25, 0, 0, 0.25]);
    close(g.jacobian(g.sub(x, 7), x), [1, 0, 0, 1]);
  });

  it('keeps the scalar out of the gradient and out of the node list', () => {
    const g = graph();
    const x = g.input('x', [3], [1, 2, 3]);
    g.sum(g.mul(x, 0.5));
    close(x.grad, [0.5, 0.5, 0.5]);
    expect(g.nodes.map((n) => n.name)).toEqual(['x', 'mul1', 'sum1']);
  });

  it('still needs a node as the first argument', () => {
    const g = graph();
    const x = g.input('x', [1], [1]);
    expect(() => g.add(2 as unknown as Node, x)).toThrow("add: argument 'input' must be a tensor");
    expect(() => g.matmul(x, 2 as unknown as Node)).toThrow("matmul: argument 'other' must be a tensor");
  });
});

describe('embedding', () => {
  it('picks rows of the weight and sends gradients back to them', () => {
    const g = graph({ check: true });
    const ids = g.label('ids', [2, 0, 2]);
    const W = g.param('W', [3, 2], [[1, 2], [3, 4], [5, 6]]);
    const e = g.embedding(ids, W);
    expect(e.shape).toEqual([3, 2]);
    expect(e.value.toNested()).toEqual([[5, 6], [1, 2], [5, 6]]);
    g.sum(e);
    expect(W.grad!.toNested()).toEqual([[1, 1], [0, 0], [2, 2]]);
    expect(ids.grad).toBeNull();
  });

  it('keeps the shape of the ids and rejects ids outside the table', () => {
    const g = graph();
    const W = g.param('W', [3, 2], [[1, 2], [3, 4], [5, 6]]);
    expect(g.embedding(g.label('one', 1), W).shape).toEqual([2]);
    expect(g.embedding(g.label('grid', [[0, 1], [2, 0]]), W).shape).toEqual([2, 2, 2]);
    expect(() => g.embedding(g.label('bad', [3]), W)).toThrow('embedding: index 3 at [0] is out of range for weight (3, 2)');
    expect(() => g.embedding(g.label('flat', [0]), g.param('v', [3], [1, 2, 3]))).toThrow('embedding: weight (3,) must be two-dimensional');
  });
});

describe('narrow, split and cat', () => {
  it('narrow copies a slice and accepts a negative start', () => {
    const g = graph({ check: true });
    const x = g.input('x', [2, 4], [[1, 2, 3, 4], [5, 6, 7, 8]]);
    expect(g.narrow(x, 1, 1, 2).value.toNested()).toEqual([[2, 3], [6, 7]]);
    expect(g.narrow(x, -1, -1, 1).value.toNested()).toEqual([[4], [8]]);
    expect(g.narrow(x, 0, 1, 1).value.toNested()).toEqual([[5, 6, 7, 8]]);
    expect(() => g.narrow(x, 1, 3, 2)).toThrow('narrow: start 3 and length 2 are out of range for input (2, 4) along dim 1');
    expect(g.narrow(x, 1, 1, 2, 2).value.toNested()).toEqual([[2, 4], [6, 8]]);
    const h = graph({ check: true });
    const y = h.input('y', [2, 4], [[1, 2, 3, 4], [5, 6, 7, 8]]);
    const odd = h.narrow(y, 1, 1, 2, 2);
    h.sum(h.mul(odd, odd));
    expect(y.grad!.toNested()).toEqual([[0, 4, 0, 8], [0, 12, 0, 16]]);
  });

  it('split returns narrow nodes named after the tensor, with a smaller last chunk', () => {
    const g = graph();
    const x = g.input('x', [2, 5], [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]);
    const parts = g.split(x, 2, 1);
    expect(parts.map((p) => p.name)).toEqual(['x_1', 'x_2', 'x_3']);
    expect(parts.map((p) => p.op)).toEqual(['narrow', 'narrow', 'narrow']);
    expect(parts.map((p) => p.shape)).toEqual([[2, 2], [2, 2], [2, 1]]);
    expect(parts[2].value.toNested()).toEqual([[5], [10]]);
    expect(parts.map((p) => p.attrs)).toEqual([
      { dim: 1, start: 0, length: 2, step: 1 },
      { dim: 1, start: 2, length: 2, step: 1 },
      { dim: 1, start: 4, length: 1, step: 1 },
    ]);
    const y = g.input('y', [2, 5], [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]);
    const rows = g.split(y, 1);
    expect(rows.map((p) => [p.name, p.shape])).toEqual([['y_1', [1, 5]], ['y_2', [1, 5]]]);
    expect(() => g.split(x, 2, 1)).toThrow("node 'x_1' already exists");
  });

  it('split takes names, tex and elem for each chunk', () => {
    const g = graph();
    const x = g.input('x', [4], [1, 2, 3, 4]);
    const [a, b] = g.split(x, 2, { names: ['a', 'b'], tex: ['\\mathbf{a}', '\\mathbf{b}'], elem: ['a', 'b'] });
    expect([a.name, a.tex, a.elem, b.name, b.tex, b.elem]).toEqual(['a', '\\mathbf{a}', 'a', 'b', '\\mathbf{b}', 'b']);
    expect(a.value.toNested()).toEqual([1, 2]);
    expect(() => g.split(x, 2, 0, { names: ['p'] })).toThrow('split: names has 1 entries for 2 chunks');
    const count = g.nodes.length;
    expect(() => g.split(x, 2, 0, { names: ['p', 'p'] })).toThrow("node 'p' already exists");
    expect(() => g.split(x, 2, 0, { names: ['q', 'a'] })).toThrow("node 'a' already exists");
    expect(g.nodes.length).toBe(count);
    expect(() => g.split(x, 2, 0, { name: 'p' } as never)).toThrow("split: unknown option 'name'");
    expect(() => g.split(x, 0)).toThrow('split: split_size must be a positive integer, got 0');
    expect(() => g.split(x, 'dh')).toThrow("split: dimension 'dh' is not defined");
  });

  it('cat joins any number of nodes along a dimension', () => {
    const g = graph({ check: true });
    const a = g.input('a', [2, 1], [[1], [2]]);
    const b = g.input('b', [2, 2], [[3, 4], [5, 6]]);
    const c = g.cat([a, b, a], 1);
    expect(c.inputs).toEqual([a, b, a]);
    expect(c.value.toNested()).toEqual([[1, 3, 4, 1], [2, 5, 6, 2]]);
    g.sum(g.mul(c, g.input('w', [4], [1, 2, 3, 4])));
    expect(a.grad!.toNested()).toEqual([[5], [5]]);
    expect(b.grad!.toNested()).toEqual([[2, 3], [2, 3]]);
    expect(g.cat([a]).value.toNested()).toEqual([[1], [2]]);
    expect(g.cat([a, a]).shape).toEqual([4, 1]);
    expect(g.cat([a, b], { dim: -1 }).shape).toEqual([2, 3]);
  });

  it('cat rejects empty lists, non-nodes and mismatched shapes', () => {
    const g = graph();
    const a = g.input('a', [2, 1], [[1], [2]]);
    const b = g.input('b', [3, 1], [[1], [2], [3]]);
    expect(() => g.cat([])).toThrow("cat: argument 'tensors' must be a non-empty list of tensors");
    expect(() => g.cat(a as unknown as Node[])).toThrow("cat: argument 'tensors' must be a non-empty list of tensors");
    expect(() => g.cat([a, b], 1)).toThrow('cat: tensors (2, 1) and (3, 1) differ outside dim 1');
    const other = graph().input('o', [2, 1], [[1], [2]]);
    expect(() => g.cat([a, other])).toThrow("cat: node 'o' belongs to another graph");
  });
});

describe('dimension names as sizes', () => {
  it('narrow and split resolve dims for the toy size and real for realShape', () => {
    const g = graph({ dims: { T: 3, dh: 2, qd: 4 }, real: { dh: 128, qd: 4096, name: 'Qwen3-8B' } });
    const q = g.input('q', ['T', 'qd'], Array.from({ length: 12 }, (_, i) => i));
    const heads = g.split(q, 'dh', 1);
    expect(heads.map((h) => h.shape)).toEqual([[3, 2], [3, 2]]);
    expect(heads.map((h) => h.realShape)).toEqual([[3, 128], [3, 128]]);
    expect(heads[1].attrs).toEqual({ dim: 1, start: 2, length: 2, step: 1 });
    expect(heads[1].value.toNested()).toEqual([[2, 3], [6, 7], [10, 11]]);
    const single = g.narrow(q, 1, 0, 'dh', { name: 'head' });
    expect(single.realShape).toEqual([3, 128]);
    expect(() => g.narrow(q, 1, 0, 'nope')).toThrow("narrow: dimension 'nope' is not defined");
  });

  it('uses the dims value when real does not name the dimension', () => {
    const g = graph({ dims: { T: 3, dh: 2, qd: 4 }, real: { qd: 4096 } });
    const q = g.input('q', ['T', 'qd'], Array.from({ length: 12 }, (_, i) => i));
    expect(g.split(q, 'dh', 1).map((h) => h.realShape)).toEqual([[3, 2], [3, 2]]);
  });
});

describe('buffer', () => {
  it('has no gradient, can be set and reset, and holds -Infinity', () => {
    const g = graph({ check: true });
    const s = g.input('s', [2, 2], [[1, 2], [3, 4]]);
    const M = g.buffer('M', [2, 2], [[0, -Infinity], [0, 0]]);
    expect(M.kind).toBe('buffer');
    const masked = g.add(s, M, { name: 'masked' });
    const p = g.softmax(masked, -1, { name: 'p' });
    g.sum(g.mul(p, g.input('w', [2, 2], [[1, 2], [3, 4]])), { name: 'loss' });
    expect(p.value.toNested()).toEqual([[1, 0], expect.any(Array)]);
    expect(M.grad).toBeNull();
    expect(s.grad).not.toBeNull();
    M.set([[0, 0], [0, 0]]);
    expect(p.value.get([0, 1])).toBeGreaterThan(0);
    g.reset();
    expect(M.value.get([0, 1])).toBe(-Infinity);
  });

  it('is left out of the backward steps and its gradient cannot appear in a template', () => {
    const g = graph();
    const x = g.input('x', [2], [1, 2]);
    const c = g.buffer('c', [2], [0.5, 0.5]);
    const y = g.mul(x, c, { name: 'y' });
    const loss = g.sum(y, { name: 'loss' });
    const steps = createSteps(g, { forward: [{ show: [x, c], title: 'x' }, { show: [y, loss], title: 'y' }] });
    const backward = steps.list.filter((s) => s.dir === 'backward');
    expect(backward.flatMap((s) => s.nodes.map((n) => n.name))).toEqual(['loss', 'y', 'x']);
    expect(() => createSteps(g, { forward: [{ show: c, title: 'c', text: '{{c.grad[0]}}' }] })).toThrow(
      "cannot parse template '{{c.grad[0]}}': buffer 'c' has no gradient",
    );
  });
});

describe('defineOp fields for lists, scalars and sizes', () => {
  const base = {
    shape: ([a]: number[][]) => [...a],
    forward: ([a]: Tensor[]) => a,
    backward: (gradOut: Tensor) => [gradOut],
  };

  it('validates list, scalars and sizes', () => {
    expect(() => defineOp({ ...base, name: 'list_mix', args: ['tensors', 'other'], tensors: ['tensors', 'other'], list: 'tensors' })).toThrow(
      "op 'list_mix': list 'tensors' must be the last tensor argument",
    );
    expect(() =>
      defineOp({ ...base, name: 'list_flags', args: ['tensors'], tensors: ['tensors'], list: 'tensors', differentiable: [true, false] }),
    ).toThrow("op 'list_flags': differentiable must have one entry per tensor");
    expect(() => defineOp({ ...base, name: 'scalar_bad', args: ['input', 'k'], tensors: ['input'], scalars: ['k'] })).toThrow(
      "op 'scalar_bad': scalar 'k' is not a tensor argument",
    );
    expect(() => defineOp({ ...base, name: 'size_bad', args: ['input'], tensors: ['input'], sizes: ['input'] })).toThrow(
      "op 'size_bad': size 'input' is not a non-tensor argument",
    );
  });

  it('names list inputs by position in gradient check failures', () => {
    defineOp({
      name: 'bad_concat_sum',
      args: ['tensors'],
      tensors: ['tensors'],
      list: 'tensors',
      shape: ([a]) => [...a],
      forward: (inputs) => new Tensor(inputs[0].shape, inputs[0].data.map((_, i) => inputs.reduce((s, t) => s + t.data[i], 0))),
      backward: (gradOut, inputs) => inputs.map((_, k) => new Tensor(gradOut.shape, gradOut.data.map((v) => (k === 0 ? v : 2 * v)))),
    });
    const g = graph({ check: true });
    const a = g.input('a', [2], [1, 2]);
    const call = (g as unknown as { bad_concat_sum(t: Node[], o?: object): Node }).bad_concat_sum;
    expect(() => call([a, a], { name: 'u' })).toThrow(/^gradient check failed: op 'bad_concat_sum' \(node 'u'\), input 1 'tensors\[1\]' at \[0\]/);
  });
});

describe('creation ops and the linear composite', () => {
  it('arange rejects a zero step and causal_mask fills with the dtype minimum', () => {
    const g = graph();
    expect(() => g.arange(0, 4, 0)).toThrow('arange: step must be nonzero');
    const mask = g.causal_mask(2, 'bf16');
    expect(mask.dtype).toBe('bf16');
    expect(Array.from(mask.value.data)).toEqual([0, -3.3895313892515355e38, 0, 0]);
    expect(mask.grad).toBeNull();
  });

  it('linear transposes the weight and adds the bias only when one is given', () => {
    const g = graph({ check: true });
    const x = g.input('x', [1, 2], [[1, 2]]);
    const w = g.param('w', [3, 2], [[1, 0], [0, 1], [1, 1]]);
    const b = g.param('b', [3], [0.5, 0.5, 0.5]);
    close(g.linear(x, w).value, [1, 2, 3]);
    const y = g.linear(x, w, b, { name: 'y' });
    expect([y.name, y.op, y.inputs[0].op]).toEqual(['y', 'add', 'matmul']);
    close(y.value, [1.5, 2.5, 3.5]);
    expect(() => g.linear(x, w, 2 as unknown as Node)).toThrow("linear: argument 'bias' must be a tensor or None");
  });
});
