import { describe, expect, it } from 'vitest';
import { type Graph, type Mesh, type Node, type Placement, graph, placements } from '../../src/core';

const MESH: Mesh = { axis: 'tp', size: 2, nodes: 1, real: { size: 8, nodes: 1 } };

function values(shape: number[], seed: number): number[] {
  const size = shape.reduce((a, b) => a * b, 1);
  return Array.from({ length: size }, (_, i) => Math.round(Math.sin(seed * 7.1 + i * 1.3) * 100) / 100);
}

function setup(build: (g: Graph) => { leaves: [Node, Placement][]; out: Node }) {
  const g = graph({ check: false });
  const { leaves, out } = build(g);
  return { g, out, p: placements(g, MESH, new Map(leaves)) };
}

function sumOf(tensors: { data: Float64Array }[]): number[] {
  return Array.from(tensors[0].data, (_, i) => tensors.reduce((s, t) => s + t.data[i], 0));
}

describe('placement propagation', () => {
  it('splits the output columns when a replicated input meets a column-sharded weight', () => {
    let x!: Node;
    const { p, out } = setup((g) => {
      x = g.input('x', [3, 4], values([3, 4], 1));
      const W = g.param('W', [6, 4], values([6, 4], 2));
      const y = g.matmul(x, W.T, { name: 'y' });
      return { leaves: [[W, { kind: 'shard', dim: 0 }]], out: g.sum(g.mul(y, y)) };
    });
    const y = out.graph.node('y')!;
    expect(p.of(out.graph.node('W.T')!)).toEqual({ kind: 'shard', dim: 1 });
    expect(p.of(y)).toEqual({ kind: 'shard', dim: 1 });
    expect(p.owner(y, [0, 2])).toBe(0);
    expect(p.owner(y, [0, 3])).toBe(1);
    expect(p.grad(x)).toEqual({ placement: { kind: 'partial' }, reduce: true });
    const parts = p.gradRanks(x)!;
    sumOf(parts).forEach((v, i) => expect(v).toBeCloseTo(x.grad!.data[i], 12));
    expect(parts[0].data.some((v, i) => Math.abs(v - x.grad!.data[i]) > 1e-9)).toBe(true);
  });

  it('gives a Partial result when the contraction dimension is sharded, and all_reduce makes it Replicate', () => {
    let partial!: Node;
    let reduced!: Node;
    const { p } = setup((g) => {
      const h = g.input('h', [3, 4], values([3, 4], 3));
      const up = g.param('up', [6, 4], values([6, 4], 4));
      const down = g.param('down', [4, 6], values([4, 6], 5));
      partial = g.matmul(g.silu(g.matmul(h, up.T)), down.T, { name: 'partial' });
      reduced = g.all_reduce(partial, 'tp', { name: 'reduced' });
      return { leaves: [[up, { kind: 'shard', dim: 0 }], [down, { kind: 'shard', dim: 1 }]], out: g.sum(g.mul(reduced, reduced)) };
    });
    expect(p.of(partial)).toEqual({ kind: 'partial' });
    expect(p.of(reduced)).toEqual({ kind: 'replicate' });
    expect(p.grad(partial)).toEqual({ placement: { kind: 'replicate' }, reduce: false });
    const ranks = p.ranks(partial)!;
    expect(ranks).toHaveLength(2);
    sumOf(ranks).forEach((v, i) => expect(v).toBeCloseTo(partial.value.data[i], 12));
    expect(Array.from(reduced.value.data)).toEqual(Array.from(partial.value.data));
  });

  it('rejects a Partial tensor that reaches a nonlinear op or is added to a Replicate one', () => {
    const { p, out } = setup((g) => {
      const h = g.input('h', [2, 4], values([2, 4], 6));
      const W = g.param('W', [3, 4], values([3, 4], 7));
      const y = g.matmul(h, W.T, { name: 'y' });
      g.exp(y, { name: 'e' });
      g.add(y, g.input('r', [2, 3], values([2, 3], 8)), { name: 'r2' });
      return { leaves: [[h, { kind: 'shard', dim: 1 }], [W, { kind: 'shard', dim: 1 }]], out: y };
    });
    expect(p.of(out)).toEqual({ kind: 'partial' });
    expect(() => p.of(out.graph.node('e')!)).toThrow(/节点 'e'（exp）：Partial 不能直接进入非线性运算/);
    expect(() => p.of(out.graph.node('r2')!)).toThrow(/Partial 与非 Partial 相加不合法/);
  });

  it('keeps attention heads on their own device and concatenates them back into a column shard', () => {
    let heads!: Node[];
    let cat!: Node;
    const { p } = setup((g) => {
      const x = g.input('x', [3, 4], values([3, 4], 9));
      const W = g.param('W', [4, 4], values([4, 4], 10));
      const q = g.view(g.matmul(x, W.T), [3, 2, 2], { name: 'q' });
      heads = [0, 1].map((h) => {
        const k = g.select(q, 1, h, { name: `k${h}` });
        const S = g.select_matmul(q, g.transpose(k, 0, 1), h, { name: `S${h}` });
        const P = g.softmax(S, -1, { name: `P${h}` });
        return g.matmul(P, k, { name: `o${h}` });
      });
      cat = g.cat(heads, 1, { name: 'o' });
      return { leaves: [[W, { kind: 'shard', dim: 0 }]], out: g.sum(cat) };
    });
    expect(p.of(heads[0].graph.node('q')!)).toEqual({ kind: 'shard', dim: 1 });
    expect(p.of(heads[0].graph.node('S1')!)).toEqual({ kind: 'local', rank: 1 });
    expect(p.of(heads[0])).toEqual({ kind: 'local', rank: 0 });
    expect(p.of(heads[1])).toEqual({ kind: 'local', rank: 1 });
    expect(p.of(cat)).toEqual({ kind: 'shard', dim: 1 });
  });

  it('rejects a head split that does not fall on a head boundary', () => {
    const { p, out } = setup((g) => {
      const x = g.input('x', [2, 4], values([2, 4], 11));
      const W = g.param('W', [6, 4], values([6, 4], 12));
      const q = g.view(g.matmul(x, W.T), [2, 3, 2], { name: 'q' });
      return { leaves: [[W, { kind: 'shard', dim: 0 }]], out: q };
    });
    expect(() => p.of(out)).toThrow(/节点 'q'（view）：切分的边界没有落在/);
  });

  it('rejects ops without a placement rule and row-wise ops along the sharded dimension', () => {
    const { p, out } = setup((g) => {
      const ids = g.label('ids', [0, 1], { classes: 4 });
      const E = g.param('E', [4, 4], values([4, 4], 13));
      g.embedding(ids, E, { name: 'e' });
      const x = g.input('x', [2, 4], values([2, 4], 14));
      g.softmax(x, 1, { name: 's' });
      const out = g.causal_conv1d(x, g.input('w', [4, 2], values([4, 2], 15)), { name: 'conv' });
      return { leaves: [[E, { kind: 'shard', dim: 1 }], [x, { kind: 'shard', dim: 1 }]], out };
    });
    expect(() => p.of(out)).toThrow(/运算 causal_conv1d 没有放置规则/);
    expect(() => p.of(out.graph.node('e')!)).toThrow(/embedding 表要是 Replicate/);
    expect(() => p.of(out.graph.node('s')!)).toThrow(/沿被切分的第 1 维做 softmax 需要先通信/);
  });

  it('requires a Partial input for all_reduce', () => {
    const { p, out } = setup((g) => {
      const x = g.input('x', [2, 2], values([2, 2], 15));
      return { leaves: [], out: g.all_reduce(x, 'tp', { name: 'r' }) };
    });
    expect(() => p.of(out)).toThrow(/all_reduce 的输入必须是 Partial，这里是 Replicate/);
  });
});
