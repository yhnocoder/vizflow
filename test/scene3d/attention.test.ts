import { describe, expect, it } from 'vitest';
import type { Node } from '../../src/autograd';
import { graph } from '../../src/core';
import { explain } from '../../src/explain';
import type { SceneController } from '../../src/scene3d/controller';
import { buildEdges } from '../../src/scene3d/edges';
import { chunkAxis, elementAt, rowCol } from '../../src/scene3d/layout';
import { type LevelLayout, levelPlan } from '../../src/scene3d/levels';
import type { Renderable } from '../../src/scene3d/renderable';
import type { StepsSpec } from '../../src/steps';
import { forEachIndex } from '../../src/tensor';

const T = 3;
const H = 2;
const HKV = 1;
const D = 4;
const MODEL = 4;

function values(shape: number[], seed: number): number[] {
  const size = shape.reduce((a, b) => a * b, 1);
  return Array.from({ length: size }, (_, i) => Math.sin(seed + i * 0.73) * 0.8);
}

function attention(showTranspose = false) {
  const g = graph({ dims: {} });
  const x = g.input('x', [T, MODEL], values([T, MODEL], 1));
  const Wq = g.param('Wq', [H * D, MODEL], values([H * D, MODEL], 2));
  const Wk = g.param('Wk', [HKV * D, MODEL], values([HKV * D, MODEL], 3));
  const Wv = g.param('Wv', [HKV * D, MODEL], values([HKV * D, MODEL], 4));
  const Wo = g.param('Wo', [MODEL, H * D], values([MODEL, H * D], 5));
  const gq = g.param('gq', [D], [0.9, 1.1, 1.0, 0.8]);
  const gk = g.param('gk', [D], [1.2, 0.7, 1.0, 0.9]);
  const cos = g.input('cos', [T, D], Array.from({ length: T }, (_, t) => Array.from({ length: D }, (_, c) => Math.cos(t * (c % 2 === 0 ? 1 : 0.1)))));
  const sin = g.input('sin', [T, D], Array.from({ length: T }, (_, t) => Array.from({ length: D }, (_, c) => Math.sin(t * (c % 2 === 0 ? 1 : 0.1)))));
  const mask = g.input('mask', [T, T], Array.from({ length: T }, (_, t) => Array.from({ length: T }, (_, s) => (s > t ? -1e9 : 0))));
  const q_linear = g.matmul(x, Wq.T, { name: 'q_linear' });
  const q = g.view(q_linear, [T, H, D], { name: 'q' });
  const qn = g.mul(q, gq, { name: 'qn' });
  const qt = g.transpose(qn, 0, 1, { name: 'qt' });
  const k_linear = g.matmul(x, Wk.T, { name: 'k_linear' });
  const k = g.view(k_linear, [T, HKV, D], { name: 'k' });
  const kn = g.mul(k, gk, { name: 'kn' });
  const kt0 = g.transpose(kn, 0, 1, { name: 'kt0' });
  const v_linear = g.matmul(x, Wv.T, { name: 'v_linear' });
  const v = g.view(v_linear, [T, HKV, D], { name: 'v' });
  const vt = g.transpose(v, 0, 1, { name: 'vt' });
  const rope = (input: Node, p: string) => {
    const c = g.mul(input, cos, { name: `${p}c` });
    const first = g.narrow(input, -1, 0, D / 2, { name: `${p}1` });
    const second = g.narrow(input, -1, D / 2, D / 2, { name: `${p}2` });
    const negated = g.neg(second, { name: `n${p}2` });
    const rotated = g.cat([negated, first], { dim: -1, name: `${p}rh` });
    const s = g.mul(rotated, sin, { name: `${p}s` });
    return { c, first, second, negated, rotated, s, out: g.add(c, s, { name: `${p}r` }) };
  };
  const rq = rope(qt, 'q');
  const rk = rope(kt0, 'k');
  const krep = g.repeat_kv(rk.out, H / HKV, { name: 'krep' });
  const kT = g.transpose(krep, 1, 2, { name: 'kT' });
  const S = g.matmul(rq.out, kT, { name: 'S' });
  const Ss = g.mul(S, 0.5, { name: 'Ss' });
  const Sm = g.add(Ss, mask, { name: 'Sm' });
  const ex = g.exp(Sm, { name: 'ex' });
  const Z = g.sum(ex, -1, { keepdim: true, name: 'Z' });
  const P = g.div(ex, Z, { name: 'P' });
  const vrep = g.repeat_kv(vt, H / HKV, { name: 'vrep' });
  const pv = g.matmul(P, vrep, { name: 'pv' });
  const back = g.transpose(pv, 0, 1, { name: 'back' });
  const o = g.reshape(back, [T, H * D], { name: 'o' });
  const att = g.matmul(o, Wo.T, { name: 'att' });
  const loss = g.loss(g.sum(att, null, { name: 'loss' }));
  const step = (title: string, show: Node[]) => ({ title, segment: 'attn', show });
  const spec: StepsSpec = {
    forward: [
      { title: 'inputs', kind: 'inputs', show: [x, Wq, Wk, Wv, Wo, gq, gk, cos, sin, mask] },
      step('q_proj', [Wq.T, q_linear]),
      step('k_proj', [Wk.T, k_linear]),
      step('v_proj', [Wv.T, v_linear]),
      step('norm', [qn, kn]),
      ...(showTranspose ? [step('transpose', [qt])] : []),
      step('cos', [rq.c, rk.c]),
      step('half', [rq.first, rq.second, rk.first, rk.second]),
      step('neg', [rq.negated, rk.negated]),
      step('rh', [rq.rotated, rk.rotated]),
      step('sin', [rq.s, rk.s]),
      step('add', [rq.out, rk.out]),
      step('krep', [krep]),
      step('vrep', [vrep]),
      step('kT', [kT]),
      step('S', [S]),
      step('Ssc', [Ss]),
      step('mask', [Sm]),
      step('exp', [ex]),
      step('sum', [Z]),
      step('div', [P]),
      step('pv', [pv]),
      step('reshape', [o]),
      step('o_proj', [Wo.T, att]),
      { title: 'loss', segment: 'loss', show: [loss] },
    ],
    backward: 'reverse',
  };
  const nodes = { x, cos, sin, mask, q_linear, q, qn, qt, kn, kt0, v, vt, krep, kT, S, Ss, Sm, ex, Z, P, vrep, pv, back, o, att, qr: rq.out, kr: rk.out, qc: rq.c, q1: rq.first, q2: rq.second, qrh: rq.rotated };
  return { g, spec, nodes };
}

function resolve(spec: StepsSpec): LevelLayout {
  return levelPlan(spec).resolve({ folded: () => false, aligned: new Map() });
}

const blockOf = (layout: LevelLayout, node: Node) => layout.blocks.find((b) => b.node === node && b.copyOf === undefined);

function sceneOf(g: ReturnType<typeof attention>['g'], layout: LevelLayout): SceneController {
  const renderables = layout.blocks.map((block) => ({ block, far: false, transition: { current: new Float64Array(block.elements.length) } })) as unknown as Renderable[];
  return {
    lod: false,
    frontRowOnly: false,
    g,
    boards: layout.boards,
    renderables,
    explainer: explain(g),
    levelLayout: layout,
    stepNodes: new Set(g.nodes),
    state: { dir: 'forward', index: 1, focus: 0, revealed: () => true, active: () => true },
  } as unknown as SceneController;
}

describe('chunk axis', () => {
  it('follows view, transpose, repeat_kv, batched matmul and broadcasting through the attention graph', () => {
    const { nodes: n } = attention();
    const axes = Object.fromEntries(Object.entries(n).filter(([, node]) => node.shape.length === 3).map(([name, node]) => [name, chunkAxis(node)]));
    expect(axes).toEqual({ q: 1, qn: 1, qt: 0, kn: 1, kt0: 0, kr: 0, v: 1, vt: 0, krep: 0, kT: 0, S: 0, Ss: 0, Sm: 0, ex: 0, Z: 0, P: 0, vrep: 0, pv: 0, back: 1, qr: 0, qc: 0, q1: 0, q2: 0, qrh: 0 });
  });

  it('draws (T, H, d) with chunk axis 1 exactly like its transpose (H, T, d) with chunk axis 0', () => {
    const { nodes: n } = attention();
    forEachIndex(n.qn.shape, ([t, h, c]) => {
      expect(rowCol(n.qt, [h, t, c])).toEqual(rowCol(n.qn, [t, h, c]));
      expect(rowCol(n.qn, [t, h, c])).toEqual([t, h * D + c]);
    });
  });

  it('takes the first dimension of a rank-3 leaf as its chunk axis', () => {
    const g = graph({ dims: {} });
    const w = g.param('w', [2, 3, 4], values([2, 3, 4], 9));
    expect(chunkAxis(w)).toBe(0);
    expect(rowCol(w, [1, 2, 3])).toEqual([2, 7]);
  });
});

describe('layout ops in the level layout', () => {
  it('draws a transpose that no step shows on its input, at the input positions', () => {
    const { spec, nodes: n } = attention();
    const layout = resolve(spec);
    expect(layout.sourceOf.get(n.qt)).toBe(n.qn);
    expect(layout.sourceOf.get(n.back)).toBe(n.pv);
    expect(blockOf(layout, n.qt)).toBeUndefined();
    const qn = blockOf(layout, n.qn)!;
    const qc = blockOf(layout, n.qc)!;
    forEachIndex(n.qn.shape, ([t, h, c]) => {
      expect(elementAt(qc, [h, t, c]).x).toBeCloseTo(elementAt(qn, [t, h, c]).x, 9);
      expect(elementAt(qc, [h, t, c]).z).toBeCloseTo(elementAt(qn, [t, h, c]).z, 9);
    });
    expect(qc.elements[0].y).toBeGreaterThan(qn.elements[0].y);
  });

  it('gives a layout op that a step shows its own level, column by column over its input', () => {
    const { g, spec, nodes: n } = attention(true);
    const layout = resolve(spec);
    expect(layout.sourceOf.has(n.qt)).toBe(false);
    const qt = blockOf(layout, n.qt)!;
    const qn = blockOf(layout, n.qn)!;
    expect(qt.elements[0].y).toBeGreaterThan(qn.elements[0].y);
    forEachIndex(n.qn.shape, ([t, h, c]) => expect(elementAt(qt, [h, t, c]).x).toBeCloseTo(elementAt(qn, [t, h, c]).x, 9));
    const identity = explain(g).edges(n.qt);
    expect(identity).toHaveLength(T * H * D);
    for (const e of identity) {
      if (!('index' in e.from) || !('index' in e.to)) throw new Error('transpose edges connect elements');
      const [h, t, c] = e.to.index;
      expect(e.from.index).toEqual([t, h, c]);
    }
  });

  it('draws the reshape back to (T, H d) on its own level with the head columns of attn_output', () => {
    const { spec, nodes: n } = attention();
    const layout = resolve(spec);
    const pv = blockOf(layout, n.pv)!;
    const o = blockOf(layout, n.o)!;
    expect([o.rows, o.cols]).toEqual([T, H * D]);
    expect(o.elements[0].y).toBeGreaterThan(pv.elements[0].y);
    forEachIndex(n.o.shape, ([t, j]) => expect(elementAt(o, [t, j]).x).toBeCloseTo(elementAt(pv, [Math.floor(j / D), t, j % D]).x, 9));
  });

  it('splits a (T, H d) projection into head groups when a (T, H, d) view is drawn on it', () => {
    const { spec, nodes: n } = attention();
    const layout = resolve(spec);
    const block = blockOf(layout, n.q_linear)!;
    const xs = Array.from({ length: H * D }, (_, col) => block.elements.find((e) => e.row === 0 && e.col === col)!.x);
    const steps = xs.slice(1).map((v, i) => v - xs[i]);
    expect(steps[D - 1]).toBeGreaterThan(steps[0] + 1e-6);
  });
});

describe('batched matmul boards', () => {
  it('draws q k^T as one plate per head with the k^T tile behind the scores', () => {
    const { spec, nodes: n } = attention();
    const layout = resolve(spec);
    const board = layout.boards.find((b) => b.op === n.S)!;
    expect([board.a.node, board.b.node, board.out.node]).toEqual([n.qr, n.kT, n.S]);
    expect(board.b.shape).toBe('tile');
    expect(board.out.chunks).toHaveLength(H);
    expect(board.b.chunks).toHaveLength(H);
    expect(board.out.elements[0].y).toBe(board.b.elements[0].y);
    for (let h = 0; h < H; h++) {
      for (let i = 0; i < T; i++) {
        const out = elementAt(board.out, [h, 0, i]);
        for (let k = 0; k < D; k++) expect(elementAt(board.b, [h, k, i]).x).toBeCloseTo(out.x, 9);
      }
    }
    const right = (h: number) => Math.max(...board.out.elements.filter((e) => e.index[0] === h).map((e) => e.x));
    const left = (h: number) => Math.min(...board.out.elements.filter((e) => e.index[0] === h).map((e) => e.x));
    expect(board.out.chunks![1].x - board.out.chunks![1].w / 2).toBeGreaterThan(board.out.chunks![0].x + board.out.chunks![0].w / 2 - 1e-9);
    expect(left(1) - right(0)).toBeGreaterThan(elementAt(board.out, [0, 0, 1]).x - elementAt(board.out, [0, 0, 0]).x);
    const pv = layout.boards.find((b) => b.op === n.pv)!;
    expect([pv.a.node, pv.b.node, pv.out.node]).toEqual([n.P, n.vrep, n.pv]);
  });

  it('runs the rows of q along the k^T plate of the same head and lands each sum on the score cell', () => {
    const { g, spec, nodes: n } = attention();
    const layout = resolve(spec);
    const drawn = buildEdges(sceneOf(g, layout)).filter((d) => d.edge.op === n.S);
    const board = layout.boards.find((b) => b.op === n.S)!;
    const qr = blockOf(layout, n.qr)!;
    const cellOf = (d: (typeof drawn)[number]) => ('cell' in d.edge.to ? d.edge.to.cell.index : 'cell' in d.edge.from ? d.edge.from.cell.index : []);
    expect(drawn.filter((d) => d.edge.arg === 0)).toHaveLength(H * T * D * T);
    expect(drawn.filter((d) => d.edge.arg === -1)).toHaveLength(H * T * D * T);
    for (const d of drawn) {
      const [h, r, k, i] = cellOf(d);
      const [from, to] = d.endpoints();
      const tile = elementAt(board.b, [h, k, i]);
      if (d.edge.arg === 0) {
        expect(to[0]).toBeLessThan(tile.x);
        expect(to[2]).toBeCloseTo(tile.z, 9);
        if (i === 0) {
          expect(from[0]).toBeCloseTo(elementAt(qr, [h, r, k]).x, 9);
          expect(from[1]).toBeGreaterThan(qr.elements[0].y - 1e-9);
        } else {
          expect(from[0]).toBeGreaterThan(elementAt(board.b, [h, k, i - 1]).x);
          expect(from[0]).toBeLessThan(tile.x);
        }
      } else {
        expect(from[0]).toBeCloseTo(tile.x, 9);
        const target = k === D - 1 ? elementAt(board.out, [h, r, i]) : elementAt(board.b, [h, k + 1, i]);
        expect(to[0]).toBeCloseTo(target.x, 9);
      }
    }
  });

  it('connects a broadcast mask element to the same cell of every head, and the transposed q to its own columns', () => {
    const { g, spec, nodes: n } = attention();
    const layout = resolve(spec);
    const drawn = buildEdges(sceneOf(g, layout));
    const sm = blockOf(layout, n.Sm)!;
    const mask = blockOf(layout, n.mask);
    const fromMask = drawn.filter((d) => d.edge.op === n.Sm && d.fromNode === n.mask);
    expect(fromMask).toHaveLength(H * T * T);
    for (const d of fromMask) {
      if (!('index' in d.edge.from) || !('index' in d.edge.to)) throw new Error('mask edges connect elements');
      const [h, t, s] = d.edge.to.index;
      expect(d.edge.from.index).toEqual([t, s]);
      expect(h).toBeGreaterThanOrEqual(0);
      if (mask !== undefined) expect(d.endpoints()[1][0]).toBeCloseTo(elementAt(sm, [h, t, s]).x, 9);
    }
    const qn = blockOf(layout, n.qn)!;
    const fromQ = drawn.filter((d) => d.edge.op === n.qc && d.fromNode === n.qt);
    expect(fromQ).toHaveLength(H * T * D);
    for (const d of fromQ) {
      if (!('index' in d.edge.from)) throw new Error('q edges start at elements');
      const [h, t, c] = d.edge.from.index;
      expect(d.endpoints()[0][0]).toBeCloseTo(elementAt(qn, [t, h, c]).x, 9);
    }
  });
});

