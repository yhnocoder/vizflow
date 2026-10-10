import { describe, expect, it } from 'vitest';
import { type Node, type OpOptions, Tensor, defineOp, graph } from '../../src/core';

defineOp({
  name: 'test_total',
  args: ['input'],
  tensors: ['input'],
  shape: () => [],
  forward: ([a]) => Tensor.from(a.data.reduce((s, v) => s + v, 0)),
  backward: (gradOut, [a]) => [new Tensor(a.shape, new Float64Array(a.size).fill(gradOut.data[0]))],
});

declare module '../../src/core' {
  interface Graph {
    test_total(input: Node, options?: OpOptions): Node;
  }
}

describe('forward and backward', () => {
  it('sums broadcast gradients back to the shape of each input', () => {
    const g = graph();
    const a = g.input('a', [2, 3], [1, 2, 3, 4, 5, 6]);
    const b = g.input('b', [3], [10, 20, 30]);
    const c = g.input('c', [], 2);
    const total = g.test_total(g.mul(g.add(a, b), c));
    expect(total.name).toBe('test_total1');
    expect(total.value.get([])).toBe(282);
    expect(a.grad!.toNested()).toEqual([[2, 2, 2], [2, 2, 2]]);
    expect(b.grad!.toNested()).toEqual([4, 4, 4]);
    expect(c.grad!.toNested()).toBe(141);
  });

  it('keeps size-one dimensions when unbroadcasting', () => {
    const g = graph();
    const a = g.input('a', [2, 3], [1, 2, 3, 4, 5, 6]);
    const b = g.input('b', [1, 3], [1, 1, 1]);
    const row = g.input('r', [2, 1], [1, 10]);
    g.test_total(g.mul(g.sub(a, b), row));
    expect(b.grad!.toNested()).toEqual([[-11, -11, -11]]);
    expect(row.grad!.toNested()).toEqual([[3], [12]]);
    expect(a.grad!.toNested()).toEqual([[1, 1, 1], [10, 10, 10]]);
  });

  it('adds gradients when a node feeds two paths', () => {
    const g = graph();
    const x = g.input('x', [], 3);
    const y = g.add(g.mul(x, x), g.exp(x));
    g.loss(y);
    expect(y.value.get([])).toBeCloseTo(9 + Math.exp(3), 10);
    expect(x.grad!.get([])).toBeCloseTo(6 + Math.exp(3), 10);
  });

  it('leaves nodes created after the loss and nodes off the path with zero gradient', () => {
    const g = graph();
    const x = g.input('x', [2], [1, 2]);
    const unused = g.exp(x);
    const l = g.mul(g.input('one', [], 1), g.input('two', [], 2));
    const after = g.neg(l);
    expect(g.loss()).toBe(after);
    g.loss(l);
    expect(unused.grad!.toNested()).toEqual([0, 0]);
    expect(after.grad!.toNested()).toBe(0);
    expect(l.grad!.toNested()).toBe(1);
    expect(x.grad!.toNested()).toEqual([0, 0]);
  });

  it('picks the last single-element op node as the loss', () => {
    const g = graph();
    const x = g.input('x', [2], [1, 2]);
    const s = g.exp(x);
    const scalar = g.exp(g.input('z', [], 0.5));
    expect(g.loss()).toBe(scalar);
    const later = g.neg(scalar);
    expect(g.loss()).toBe(later);
    expect(s.grad!.toNested()).toEqual([0, 0]);
    expect(() => graph().loss()).toThrow('graph has no loss node');
  });

  it('refuses a loss node with more than one element', () => {
    const g = graph();
    const x = g.input('x', [2], [1, 2]);
    const y = g.exp(x);
    g.loss(y);
    expect(() => x.grad).toThrow("loss node 'exp1' must have one element, got shape (2,)");
  });
});

describe('set, subscribe and batch', () => {
  it('recomputes forward and backward after set and notifies once', () => {
    const g = graph();
    const x = g.input('x', [], 1);
    const y = g.exp(x);
    let calls = 0;
    const unsubscribe = g.subscribe(() => calls++);
    expect(g.version).toBe(0);
    x.set(2);
    expect(calls).toBe(1);
    expect(g.version).toBe(1);
    expect(y.value.get([])).toBeCloseTo(Math.exp(2), 12);
    expect(x.grad!.get([])).toBeCloseTo(Math.exp(2), 12);
    unsubscribe();
    x.set(3);
    expect(calls).toBe(1);
    expect(g.version).toBe(2);
  });

  it('merges sets inside batch into one recompute', () => {
    const g = graph();
    const x = g.input('x', [], 1);
    const y = g.input('y', [], 1);
    const z = g.mul(x, y);
    let calls = 0;
    g.subscribe(() => calls++);
    g.batch(() => {
      x.set(2);
      y.set(3);
      expect(calls).toBe(0);
    });
    expect(calls).toBe(1);
    expect(g.version).toBe(1);
    expect(z.value.get([])).toBe(6);
    g.batch(() => {});
    expect(calls).toBe(1);
  });

  it('restores leaf values on reset with a single notification', () => {
    const g = graph();
    const x = g.input('x', [2], [1, 2]);
    const t = g.label('t', [1]);
    const y = g.exp(x);
    let calls = 0;
    g.subscribe(() => calls++);
    x.set([5, 6]);
    t.set([0]);
    g.reset();
    expect(calls).toBe(3);
    expect(x.value.toNested()).toEqual([1, 2]);
    expect(t.index(0)).toBe(1);
    expect(y.value.get([0])).toBeCloseTo(Math.exp(1), 12);
  });

  it('rejects values that do not match the node shape', () => {
    const g = graph();
    const x = g.input('x', [2], [1, 2]);
    expect(() => x.set([1, 2, 3])).toThrow("node 'x': expected 2 values for shape (2,), got 3");
  });
});

describe('gradient check in check mode', () => {
  it('reports a wrong backward with the op and node name', () => {
    defineOp({
      name: 'bad_square',
      args: ['input'],
      tensors: ['input'],
      shape: ([a]) => [...a],
      forward: ([a]) => new Tensor(a.shape, a.data.map((v) => v * v)),
      backward: (gradOut, [a]) => [new Tensor(a.shape, gradOut.data.map((gv, i) => gv * a.data[i]))],
    });
    const g = graph({ check: true });
    const x = g.input('x', [3], [1, 2, 3]);
    expect(() => (g as unknown as { bad_square(n: unknown, o?: unknown): unknown }).bad_square(x, { name: 'u' })).toThrow(
      /^gradient check failed: op 'bad_square' \(node 'u'\), input 0 'input' at \[0\]: backward -?\d+\.\d{6} vs numeric -?\d+\.\d{6}$/,
    );
    const off = graph({ check: false });
    const y = off.input('y', [3], [1, 2, 3]);
    expect(() => (off as unknown as { bad_square(n: unknown): unknown }).bad_square(y)).not.toThrow();
  });

  it('rejects redefining an op', () => {
    expect(() =>
      defineOp({
        name: 'exp',
        args: ['input'],
        tensors: ['input'],
        shape: ([a]) => [...a],
        forward: ([a]) => a,
        backward: (gradOut) => [gradOut],
      }),
    ).toThrow("op 'exp' is already defined");
  });
});
