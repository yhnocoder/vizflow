import { describe, expect, it } from 'vitest';
import { type Mesh, Tensor, checkOp, commTransfers, graph, linkKind, opDef, placements, rankDevice, ringAllReduce, routedPairs } from '../../src/core';
import { dispatchTokenBytes } from '../../src/sim';

const EP: Mesh = { axis: 'ep', size: 4, nodes: 2, real: { size: 4, nodes: 2 } };
const SEL = [
  [1, 1, 0, 0, 0, 0, 0, 0],
  [0, 0, 0, 0, 1, 0, 1, 0],
  [0, 0, 1, 0, 0, 1, 0, 0],
  [0, 0, 0, 1, 0, 0, 0, 1],
];

function values(shape: number[], seed: number): number[] {
  const size = shape.reduce((a, b) => a * b, 1);
  return Array.from({ length: size }, (_, i) => Math.round(Math.sin(seed * 7.1 + i * 1.3) * 100) / 100);
}

function moe(fp8 = false) {
  const g = graph({ check: false });
  const x = g.param('x', [4, 4], values([4, 4], 1));
  const sel = g.label('sel', SEL);
  const w = g.param('w', [4, 8], values([4, 8], 2));
  const W = g.param('W', [8, 3, 4], values([8, 3, 4], 3));
  const routed = g.dispatch(x, sel, 2, 'ep', fp8, { name: 'routed' });
  const y = g.grouped_mm(routed, W, g.route_experts(sel, 2), { name: 'y' });
  const out = g.combine(y, sel, w, 'ep', { name: 'out' });
  g.loss(g.sum(g.mul(out, out)));
  const p = placements(g, EP, new Map([[x, { kind: 'shard', dim: 0 }], [sel, { kind: 'shard', dim: 0 }], [w, { kind: 'shard', dim: 0 }], [W, { kind: 'shard', dim: 0 }]]));
  return { g, x, sel, w, W, routed, y, out, p };
}

describe('mesh topology', () => {
  it('numbers ranks node by node and tells NVLink from RDMA', () => {
    const topology = { size: 4, nodes: 2 };
    expect([0, 1, 2, 3].map((r) => rankDevice(topology, r))).toEqual([{ node: 0, gpu: 0 }, { node: 0, gpu: 1 }, { node: 1, gpu: 0 }, { node: 1, gpu: 1 }]);
    expect(linkKind(topology, 0, 1)).toBe('nvlink');
    expect(linkKind(topology, 1, 2)).toBe('rdma');
    expect(linkKind(topology, 3, 3)).toBe('local');
  });

  it('lists a ring all_reduce as reduce_scatter and all_gather, each rank sending 2(n-1)/n of the message', () => {
    const transfers = ringAllReduce({ size: 4, nodes: 2 }, 1024);
    expect(transfers).toHaveLength(8);
    expect(transfers.filter((t) => t.src === 1).reduce((s, t) => s + t.bytes, 0)).toBe(1536);
    expect(transfers.filter((t) => t.phase === 'reduce_scatter').map((t) => [t.src, t.dst, t.link])).toEqual([[0, 1, 'nvlink'], [1, 2, 'rdma'], [2, 3, 'nvlink'], [3, 0, 'rdma']]);
  });
});

describe('dispatch and combine', () => {
  it('compute the same MoE output as the unsharded loop and have each other as backward', () => {
    const { x, w, W, out } = moe();
    for (let t = 0; t < 4; t++) {
      for (let i = 0; i < 3; i++) {
        let s = 0;
        for (let e = 0; e < 8; e++) {
          if (SEL[t][e] === 0) continue;
          let dot = 0;
          for (let c = 0; c < 4; c++) dot += W.value.data[(e * 3 + i) * 4 + c] * x.value.data[t * 4 + c];
          s += w.value.data[t * 8 + e] * dot;
        }
        expect(out.value.data[t * 3 + i]).toBeCloseTo(s, 12);
      }
    }
    const dispatch = opDef('dispatch')!;
    const combine = opDef('combine')!;
    const sel = new Tensor([4, 8], Float64Array.from(SEL.flat()));
    const xs = new Tensor([4, 4], Float64Array.from(values([4, 4], 4)));
    const ys = new Tensor([8, 4], Float64Array.from(values([8, 4], 5)));
    const ws = new Tensor([4, 8], Float64Array.from(values([4, 8], 6)));
    expect(() => checkOp(dispatch, [xs, sel], { k: 2, group: 'ep', fp8: false })).not.toThrow();
    expect(() => checkOp(combine, [ys, sel, ws], { group: 'ep' })).not.toThrow();
    const [back] = dispatch.backward(ys, [xs, sel], dispatch.forward([xs, sel], { k: 2 }), { k: 2 });
    expect(Array.from(back!.data)).toEqual(Array.from(combine.forward([ys, sel, sel], {}).data));
    const [backY] = combine.backward(xs, [ys, sel, sel], combine.forward([ys, sel, sel], {}), {});
    expect(Array.from(backY!.data)).toEqual(Array.from(dispatch.forward([xs, sel], { k: 2 }).data));
  });

  it('place each routed row on the device of its expert and return tokens to their own device', () => {
    const { p, routed, y, out } = moe();
    expect(p.of(routed)).toMatchObject({ kind: 'routed' });
    expect(p.of(y)).toMatchObject({ kind: 'routed' });
    expect(p.of(out)).toEqual({ kind: 'shard', dim: 0 });
    expect([0, 1, 2, 3, 4, 5, 6, 7].map((r) => p.owner(routed, [r, 0]))).toEqual([0, 0, 1, 1, 2, 2, 3, 3]);
  });

  it('keep each row on the device of its expert through view and gather, so every row can pick its own weight', () => {
    const { g, sel, w, routed, p } = moe();
    const ex = g.route_experts(sel, 2);
    const weights = g.dispatch(w, sel, 2, 'ep', false, { name: 'weights' });
    const own = g.gather(weights, 1, g.view(ex, 8, 1), { name: 'own' });
    expect(p.of(own)).toEqual(p.of(routed));
    const pairs = routedPairs(sel.value, 8, 'dispatch');
    expect(Array.from(own.value.data)).toEqual(pairs.map((pair) => w.value.get([pair.token, pair.expert])));
    expect(() => p.of(g.gather(routed, 0, g.view(ex, 8, 1)))).toThrow(/gather 只支持/);
  });

  it('send a token once per destination device, in two hops across nodes', () => {
    const { p, routed, out } = moe();
    const hybrid = commTransfers(p, routed);
    const of = (token: number) => hybrid.filter((t) => t.plan!.tokenRank === token);
    expect(of(0).map((t) => [t.src, t.dst, t.link, t.hop, t.plan!.experts])).toEqual([[0, 0, 'local', 0, [0, 1]], [0, 0, 'local', 1, [0, 1]]]);
    expect(of(1).map((t) => [t.src, t.dst, t.link, t.hop])).toEqual([[1, 3, 'rdma', 0], [3, 2, 'nvlink', 1], [3, 3, 'local', 1]]);
    expect(hybrid.every((t) => t.bytes === dispatchTokenBytes(4, 'bf16', 2))).toBe(true);
    const direct = commTransfers(p, routed, { mode: 'direct' }).filter((t) => t.plan!.tokenRank === 1);
    expect(direct.map((t) => [t.src, t.dst, t.link, t.hop])).toEqual([[1, 2, 'rdma', 0], [1, 3, 'rdma', 0]]);
    const back = commTransfers(p, out).filter((t) => t.plan!.tokenRank === 1);
    expect(back.map((t) => [t.src, t.dst, t.link, t.hop])).toEqual([[2, 3, 'nvlink', 0], [3, 3, 'local', 0], [3, 1, 'rdma', 1]]);
    const fp8 = moe(true);
    expect(commTransfers(fp8.p, fp8.routed)[0].bytes).toBe(dispatchTokenBytes(4, 'fp8', 2));
  });

  it('rejects a dispatch whose tokens are not split over the mesh', () => {
    const g = graph({ check: false });
    const x = g.param('x', [4, 4], values([4, 4], 1));
    const r = g.dispatch(x, g.label('sel', SEL), 2, 'ep', false, { name: 'r' });
    const p = placements(g, EP, new Map());
    expect(() => p.of(r)).toThrow(/节点 'r'（dispatch）：dispatch 的输入要按 token 切分/);
  });
});
