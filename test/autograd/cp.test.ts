import { describe, expect, it } from 'vitest';
import { type Graph, type Mesh, type Node, type Placement, Tensor, checkOp, commTransfers, graph, headTailChunks, opDef, placements, ringSendRecv } from '../../src/core';

const CP: Mesh = { axis: 'cp', size: 2, nodes: 1, real: { size: 8, nodes: 1 } };
const HEAD_TAIL: Placement = { kind: 'shard', dim: 0, chunks: [0, 1, 1, 0] };

function values(shape: number[], seed: number): number[] {
  const size = shape.reduce((a, b) => a * b, 1);
  return Array.from({ length: size }, (_, i) => Math.round(Math.sin(seed * 7.1 + i * 1.3) * 100) / 100);
}

function tensor(shape: number[], seed: number): Tensor {
  return new Tensor(shape, Float64Array.from(values(shape, seed)));
}

function blockMask(T: number, keep: (t: number, s: number) => boolean): number[][] {
  return Array.from({ length: T }, (_, t) => Array.from({ length: T }, (_, s) => (keep(t, s) ? 0 : -Infinity)));
}

interface Attention {
  g: Graph;
  q: Node;
  k: Node;
  v: Node;
  out: Node;
}

function fullAttention(T: number, d: number): Attention {
  const g = graph({ check: false });
  const q = g.param('q', [T, d], values([T, d], 1));
  const k = g.param('k', [T, d], values([T, d], 2));
  const v = g.param('v', [T, d], values([T, d], 3));
  const M = g.buffer('M', [T, T], blockMask(T, (t, s) => s <= t));
  const out = g.matmul(g.softmax(g.add(g.matmul(q, k.T), M), -1), v, { name: 'out' });
  g.loss(g.sum(g.mul(out, g.input('u', [T, d], values([T, d], 4)))));
  return { g, q, k, v, out };
}

function blockedAttention(T: number, d: number, blocks: number[][]): Attention {
  const g = graph({ check: false });
  const q = g.param('q', [T, d], values([T, d], 1));
  const k = g.param('k', [T, d], values([T, d], 2));
  const v = g.param('v', [T, d], values([T, d], 3));
  const S = g.matmul(q, k.T);
  let o: Node | undefined;
  let lse: Node | undefined;
  blocks.forEach((cols, j) => {
    const M = g.buffer(`M${j}`, [T, T], blockMask(T, (t, s) => s <= t && cols.includes(s)));
    const Sm = g.add(S, M);
    const oj = g.matmul(g.softmax(Sm, -1), v);
    const lj = g.logsumexp(Sm, -1, { keepdim: true });
    if (o === undefined || lse === undefined) {
      o = oj;
      lse = lj;
      return;
    }
    const merged = g.lse_merge(o, lse, oj, lj);
    lse = g.logsumexp(g.cat([lse, lj], { dim: -1 }), -1, { keepdim: true });
    o = merged;
  });
  const out = g.identity(o!, { name: 'out' });
  g.loss(g.sum(g.mul(out, g.input('u', [T, d], values([T, d], 4)))));
  return { g, q, k, v, out };
}

function expectClose(a: Tensor, b: Tensor): void {
  expect(a.shape).toEqual(b.shape);
  a.data.forEach((x, i) => expect(Math.abs(x - b.data[i])).toBeLessThan(1e-12));
}

describe('lse_merge', () => {
  it('passes the finite difference gradient check, also with a row that one block does not see', () => {
    checkOp(opDef('lse_merge')!, [tensor([3, 2], 1), tensor([3, 1], 2), tensor([3, 2], 3), tensor([3, 1], 4)], {});
    const lb = new Tensor([3, 1], Float64Array.from([-Infinity, 0.3, -0.8]));
    const ob = new Tensor([3, 2], Float64Array.from([0, 0, 0.4, -0.2, 1.1, 0.7]));
    checkOp(opDef('lse_merge')!, [tensor([2, 3, 2], 5), tensor([2, 3, 1], 6), tensor([2, 3, 2], 7), tensor([2, 3, 1], 8)], {});
    checkOp(opDef('lse_merge')!, [tensor([3, 2], 9), tensor([3, 1], 10), ob, lb], {});
  });

  it('weights each block by its share of the softmax denominator', () => {
    const g = graph({ check: false });
    const oa = g.input('oa', [2, 1], [[1], [2]]);
    const la = g.input('la', [2, 1], [[Math.log(1)], [-Infinity]]);
    const ob = g.input('ob', [2, 1], [[4], [5]]);
    const lb = g.input('lb', [2, 1], [[Math.log(3)], [0.7]]);
    const out = g.lse_merge(oa, la, ob, lb);
    expect(out.value.data[0]).toBeCloseTo(0.25 * 1 + 0.75 * 4, 14);
    expect(out.value.data[1]).toBe(5);
  });

  it('rejects inputs whose shapes do not line up', () => {
    const g = graph({ check: false });
    const o = g.input('o', [2, 3], values([2, 3], 1));
    expect(() => g.lse_merge(o, g.input('l', [2, 3], values([2, 3], 2)), o, g.input('m', [2, 1], values([2, 1], 3)))).toThrow(/lse_merge/);
  });

  it('returns 0 from softmax and -inf from logsumexp on a fully masked row, with zero gradients', () => {
    const x = new Tensor([2, 3], Float64Array.from([-Infinity, -Infinity, -Infinity, 0.5, -Infinity, 1]));
    const p = opDef('softmax')!.forward([x], { dim: -1 });
    expect(Array.from(p.data.subarray(0, 3))).toEqual([0, 0, 0]);
    const l = opDef('logsumexp')!.forward([x], { dim: -1, keepdim: true });
    expect(l.data[0]).toBe(-Infinity);
    const [gl] = opDef('logsumexp')!.backward(new Tensor([2, 1], Float64Array.from([1, 1])), [x], l, { dim: -1, keepdim: true });
    expect(Array.from(gl!.data.subarray(0, 3))).toEqual([0, 0, 0]);
    const [gp] = opDef('softmax')!.backward(new Tensor([2, 3], Float64Array.from([1, 2, 3, 4, 5, 6])), [x], p, { dim: -1 });
    expect(Array.from(gp!.data.subarray(0, 3))).toEqual([0, 0, 0]);
  });

  it.each([
    ['two head-tail blocks', [[0, 3], [1, 2]]],
    ['three blocks with fully masked rows', [[4, 5], [0, 1], [2, 3]]],
  ])('merges %s into the same values and gradients as unsplit causal attention', (_name, blocks) => {
    const T = blocks.flat().length;
    const full = fullAttention(T, 3);
    const split = blockedAttention(T, 3, blocks);
    expectClose(split.out.value, full.out.value);
    for (const key of ['q', 'k', 'v'] as const) expectClose(split[key].grad!, full[key].grad!);
  });
});

describe('context parallel placement', () => {
  function ring(mask: (t: number, s: number) => boolean = (t, s) => s <= t) {
    const g = graph({ check: false });
    const q = g.input('q', [4, 2], values([4, 2], 1));
    const k = g.input('k', [4, 2], values([4, 2], 2));
    const v = g.input('v', [4, 2], values([4, 2], 3));
    const k1 = g.send_recv(k, 'cp', { name: 'k1' });
    const v1 = g.send_recv(v, 'cp', { name: 'v1' });
    const S = g.matmul(q, k1.T, { name: 'S' });
    const M = g.buffer('M', [4, 4], blockMask(4, mask));
    const Sm = g.add(S, M, { name: 'Sm' });
    const P = g.softmax(Sm, -1, { name: 'P' });
    const lse = g.logsumexp(Sm, -1, { keepdim: true, name: 'lse' });
    const o = g.matmul(P, v1, { name: 'o' });
    g.loss(g.sum(g.mul(o, o)));
    const p = placements(g, CP, new Map([[q, HEAD_TAIL], [k, HEAD_TAIL], [v, HEAD_TAIL]]));
    return { g, k, k1, S, Sm, P, lse, o, p };
  }

  const owner = (t: number) => headTailChunks(2)[t];

  it('gives rank r the chunks r and 2n-1-r', () => {
    expect(headTailChunks(2)).toEqual([0, 1, 1, 0]);
    expect(headTailChunks(4)).toEqual([0, 1, 2, 3, 3, 2, 1, 0]);
    const { k, p } = ring();
    expect([0, 1, 2, 3].map((t) => p.owner(k, [t, 0]))).toEqual([0, 1, 1, 0]);
  });

  it('moves every block to the next rank with send_recv and keeps the values', () => {
    const { k, k1, p } = ring();
    expect(p.of(k1)).toEqual({ kind: 'shard', dim: 0, chunks: [1, 0, 0, 1] });
    expect(Array.from(k1.value.data)).toEqual(Array.from(k.value.data));
    expect(p.grad(k1).placement).toEqual(p.of(k1));
  });

  it('computes a score only on the rank that holds both the query row and the key row', () => {
    const { S, Sm, P, lse, o, p } = ring((t, s) => s <= t && (owner(s) + 1) % 2 === owner(t));
    expect(p.of(S)).toEqual({ kind: 'block', rows: [0, 1, 1, 0], cols: [1, 0, 0, 1], fill: 'any' });
    expect(p.of(Sm)).toMatchObject({ kind: 'block', fill: 'neg_inf' });
    expect(p.of(P)).toMatchObject({ kind: 'block', fill: 'zero' });
    expect(p.of(lse)).toEqual(HEAD_TAIL);
    expect(p.of(o)).toEqual(HEAD_TAIL);
    expect(p.owner(S, [3, 1])).toBe(0);
    expect(p.owner(S, [3, 0])).toBeUndefined();
  });

  it('rejects a softmax over scores that the round mask does not cover', () => {
    const { P, p } = ring();
    expect(() => p.of(P)).toThrow(/softmax 之前，不在本卡的格子要是负无穷/);
  });

  it('rejects multiplying the weights of one round with the values of another round', () => {
    const g = graph({ check: false });
    const q = g.input('q', [4, 2], values([4, 2], 1));
    const k = g.input('k', [4, 2], values([4, 2], 2));
    const v = g.input('v', [4, 2], values([4, 2], 3));
    const M = g.buffer('M', [4, 4], blockMask(4, (t, s) => s <= t && owner(s) === owner(t)));
    const o = g.matmul(g.softmax(g.add(g.matmul(q, k.T), M), -1), g.send_recv(v, 'cp'), { name: 'o' });
    const p = placements(g, CP, new Map([[q, HEAD_TAIL], [k, HEAD_TAIL], [v, HEAD_TAIL]]));
    expect(() => p.of(o)).toThrow(/右边要按行切开，并且和左边的列在同一张卡上/);
  });

  it('needs a sharded input for send_recv and the mesh axis as its group', () => {
    const g = graph({ check: false });
    const x = g.input('x', [4, 2], values([4, 2], 1));
    const a = g.send_recv(x, 'cp', { name: 'a' });
    const b = g.send_recv(x, 'tp', { name: 'b' });
    const y = g.input('y', [4, 2], values([4, 2], 2));
    const c = g.send_recv(y, 'cp', { name: 'c' });
    const p = placements(g, CP, new Map([[x, HEAD_TAIL]]));
    expect(p.of(a)).toEqual({ kind: 'shard', dim: 0, chunks: [1, 0, 0, 1] });
    expect(() => p.of(b)).toThrow(/通信组 'tp' 不是 mesh 的轴/);
    expect(() => p.of(c)).toThrow(/send_recv 的输入要按某一维切开/);
  });

  it('lists one transfer per rank and round, to the next rank forward and back to the previous rank backward', () => {
    const { k1, p } = ring();
    const forward = commTransfers(p, k1, { tokens: 4096 });
    expect(forward.map((t) => [t.src, t.dst])).toEqual([[0, 1], [1, 0]]);
    expect(forward[0].bytes).toBe((4096 / 8) * 2 * 4);
    expect(ringSendRecv({ size: 4, nodes: 1 }, 10, 'backward').map((t) => [t.src, t.dst])).toEqual([[0, 3], [1, 0], [2, 1], [3, 2]]);
    expect(ringSendRecv({ size: 4, nodes: 2 }, 10).map((t) => t.link)).toEqual(['nvlink', 'rdma', 'nvlink', 'rdma']);
    expect(() => commTransfers(p, k1)).toThrow(/needs the number of tokens/);
  });
});
