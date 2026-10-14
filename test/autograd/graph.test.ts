import { describe, expect, it } from 'vitest';
import { Node, graph } from '../../src/autograd';

describe('graph building', () => {
  it('resolves named dimensions and rejects unknown ones', () => {
    const g = graph({ dims: { d: 4, V: 5 } });
    const h = g.input('h', [1, 'd'], [0.8, -1.1, 0.5, 1.4]);
    expect(h.shape).toEqual([1, 4]);
    expect(g.dims).toEqual({ d: 4, V: 5 });
    expect(() => g.param('W', ['V', 'k'], [])).toThrow("node 'W': dimension 'k' is not defined");
  });

  it('checks the number of values against the shape', () => {
    const g = graph();
    expect(() => g.input('x', [2, 2], [1, 2, 3])).toThrow("node 'x': expected 4 values for shape (2, 2), got 3");
    expect(g.input('y', [2, 2], [[1, 2], [3, 4]]).value.toNested()).toEqual([[1, 2], [3, 4]]);
  });

  it('rejects nested values whose shape differs from the declared one', () => {
    const g = graph({ dims: { d: 4, V: 5 } });
    const transposedW = [
      [0.9, 0.2, -0.4, 0.5, -0.7],
      [-0.6, -0.8, 0.2, 0.7, 0.4],
      [0.3, 0.6, 0.9, -0.6, -0.3],
      [0.6, 0.2, 0.3, -0.2, -0.5],
    ];
    expect(() => g.param('W', ['V', 'd'], transposedW)).toThrow(
      "node 'W': expected shape (5, 4), got nested values of shape (4, 5)",
    );
    const W = g.param('W', ['V', 'd'], new Array(20).fill(0));
    expect(() => W.set(transposedW)).toThrow("node 'W': expected shape (5, 4), got nested values of shape (4, 5)");
  });

  it('infers label shapes from values and keeps tokens', () => {
    const g = graph();
    expect(g.label('a', 0).shape).toEqual([]);
    expect(g.label('b', [0]).shape).toEqual([1]);
    const t = g.label('t', [[0], [1], [2]], { tokens: ['x', 'y', 'z'] });
    expect(t.shape).toEqual([3, 1]);
    expect(t.kind).toBe('label');
    expect(t.tokens).toEqual(['x', 'y', 'z']);
    expect(t.attrs).toEqual({ classes: 3 });
    expect(t.index(2, 0)).toBe(2);
    expect(t.grad).toBeNull();
    expect(g.label('u', [1], { classes: 7 }).attrs).toEqual({ classes: 7 });
  });

  it('rejects duplicate names, ignoring whitespace in lookups', () => {
    const g = graph();
    const x = g.input('log p', [1], [1]);
    expect(g.node('logp')).toBe(x);
    expect(g.node('missing')).toBeUndefined();
    expect(() => g.input('log p', [1], [2])).toThrow("node 'log p' already exists");
  });

  it('names op nodes automatically and accepts explicit names', () => {
    const g = graph();
    const x = g.input('x', [2], [1, 2]);
    expect(g.exp(x).name).toBe('exp1');
    expect(g.exp(x).name).toBe('exp2');
    expect(g.neg(x, { name: 'm' }).name).toBe('m');
    expect(g.nodes.map((n) => n.name)).toEqual(['x', 'exp1', 'exp2', 'm']);
    expect(g.nodes.map((n) => n.id)).toEqual([0, 1, 2, 3]);
  });

  it('derives tex and elem defaults from the name', () => {
    const g = graph();
    const h = g.input('h', [1, 2], [1, 2]);
    expect(h.tex).toBe('\\mathbf{h}');
    expect(h.elem).toBe('h');
    const s = g.input('s', [], 1);
    expect(s.tex).toBe('s');
    const lse = g.input('LSE', [1], [1], { elem: 'L' });
    expect(lse.tex).toBe('\\mathrm{LSE}');
    expect(lse.elem).toBe('L');
    expect(g.exp(h, { tex: 'e^{h}', card: { forward: { title: 'exp' } } }).card).toEqual({ forward: { title: 'exp' } });
  });

  it('creates and caches the transpose node', () => {
    const g = graph();
    const W = g.param('W', [2, 3], [1, 2, 3, 4, 5, 6]);
    const WT = W.T;
    expect(WT).toBe(W.T);
    expect(WT.name).toBe('W.T');
    expect(WT.op).toBe('transpose');
    expect(WT.tex).toBe('\\mathbf{W}^{\\top}');
    expect(WT.elem).toBe('W^{\\top}');
    expect(WT.shape).toEqual([3, 2]);
    expect(WT.value.toNested()).toEqual([[1, 4], [2, 5], [3, 6]]);
    expect(WT.inputs).toEqual([W]);
    expect(WT.attrs).toEqual({ dim0: 0, dim1: 1 });
    expect(() => g.input('v', [3], [1, 2, 3]).T).toThrow("transpose: node 'v' with shape (3,) is not two-dimensional");
  });

  it('computes realShape from the real model sizes', () => {
    const g = graph({ dims: { d: 4, V: 5 }, real: { d: 4096, V: 151936, name: 'Qwen3-8B' } });
    const h = g.input('h', [1, 'd'], [1, 2, 3, 4]);
    const W = g.param('W', ['V', 'd'], new Array(20).fill(0));
    expect(h.realShape).toEqual([1, 4096]);
    expect(W.T.realShape).toEqual([4096, 151936]);
    expect(g.real).toEqual({ d: 4096, V: 151936, name: 'Qwen3-8B' });
    const plain = graph({ dims: { d: 4 } });
    expect(plain.input('h', ['d'], [1, 2, 3, 4]).realShape).toEqual([4]);
    expect(plain.real).toBeUndefined();
  });

  it('adds a dimension after the graph is built', () => {
    const g = graph({ dims: { d: 4 }, real: { d: 4096 } });
    g.addDim('w', 8, 8192);
    expect(g.input('x', [1, 'w'], new Array(8).fill(0)).realShape).toEqual([1, 8192]);
    expect(g.dims).toEqual({ d: 4, w: 8 });
    expect(g.real).toEqual({ d: 4096, w: 8192 });
    g.addDim('w', 8, 8192);
    expect(() => g.addDim('w', 6, 8192)).toThrow("addDim: dimension 'w' is already defined");
    const plain = graph();
    plain.addDim('n', 3);
    expect(plain.input('y', ['n'], [1, 2, 3]).realShape).toEqual([3]);
  });

  it('validates op arguments and options', () => {
    const g = graph();
    const x = g.input('x', [2], [1, 2]);
    expect(() => g.add(x, 'a' as unknown as Node)).toThrow("add: argument 'other' must be a tensor or a number");
    expect(() => g.add(3 as unknown as Node, x)).toThrow("add: argument 'input' must be a tensor");
    expect(() => g.exp(3 as unknown as Node)).toThrow("exp: argument 'input' must be a tensor");
    expect(() => g.exp(x, { keepdim: true } as never)).toThrow("exp: unknown option 'keepdim'");
    expect(() => (g as unknown as { exp(...a: unknown[]): Node }).exp(x, x, x)).toThrow('takes at most');
    expect(() => (g as unknown as { transpose(...a: unknown[]): Node }).transpose(x, 0)).toThrow(
      "transpose: missing argument 'dim1'",
    );
    const other = graph().input('y', [2], [1, 2]);
    expect(() => g.add(x, other)).toThrow("add: node 'y' belongs to another graph");
    expect(() => g.exp(x).set([1, 2])).toThrow("node 'exp1' is computed by op 'exp' and cannot be set");
    expect('exp' in g).toBe(true);
  });

  it('refuses to add the same node name through an op', () => {
    const g = graph();
    const x = g.input('x', [1], [1]);
    expect(() => g.exp(x, { name: 'x' })).toThrow("node 'x' already exists");
  });
});

describe('findLoss', () => {
  it('returns undefined when no op node has a single element', () => {
    const g = graph();
    g.input('x', [2], [1, 2]);
    expect(g.findLoss()).toBeUndefined();
    expect(() => g.loss()).toThrow('graph has no loss node');
    g.exp(g.input('y', [2], [1, 2]));
    expect(g.findLoss()).toBeUndefined();
  });

  it('picks the last single-element op node unless one was set explicitly', () => {
    const g = graph();
    const x = g.input('x', [2], [1, 2]);
    const s = g.sum(x, { name: 's' });
    const m = g.mean(x, { name: 'm' });
    expect(g.findLoss()).toBe(m);
    expect(g.loss()).toBe(m);
    g.loss(s);
    expect(g.findLoss()).toBe(s);
    expect(g.loss()).toBe(s);
  });
});
