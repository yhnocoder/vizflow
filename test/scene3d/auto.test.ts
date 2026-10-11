import { describe, expect, it } from 'vitest';
import type { Node } from '../../src/autograd';
import { graph } from '../../src/core';
import '../../src/explain';
import { AUTO_LEVEL, AUTO_SIDE_GAP, type LayoutEntry, type LayoutTree, autoLayout } from '../../src/scene3d/auto';
import { type Block, classAlignment, resolveLayout, targetColumn } from '../../src/scene3d/layout';
import { blockSpan } from '../../src/scene3d/modules';

const leaf = (node: Node): LayoutEntry => ({ kind: 'node', node });
const leaves = (...nodes: Node[]) => nodes.map(leaf);
const tree = (title: string | undefined, children: LayoutEntry[], subtitle?: string): LayoutTree => ({ kind: 'module', title, subtitle, children });

function values(rows: number, cols: number, seed: number): number[][] {
  return Array.from({ length: rows }, (_, r) => Array.from({ length: cols }, (_, c) => Math.sin(seed + r * 1.7 + c * 0.9)));
}

function model() {
  const T = 2;
  const g = graph({ dims: { T, d: 4, f: 6, V: 5, dh: 2, qd: 4 } });
  const ids = g.label('ids', [1, 3], { tokens: ['a', 'b', 'c', 'd', 'e'] });
  const labels = g.label('y', [3, 0], { tokens: ['a', 'b', 'c', 'd', 'e'] });
  const E = g.param('E', ['V', 'd'], values(5, 4, 1));
  const e = g.embedding(ids, E, { name: 'e' });

  const norm = (x: Node, gamma: Node, tag: string) => {
    const sq = g.pow(x, 2, { name: `${tag}²` });
    const ms = g.mean(sq, -1, { keepdim: true, name: `ms_${tag}` });
    const shifted = g.add(ms, 1e-6, { name: `ms_${tag}+ε` });
    const r = g.rsqrt(shifted, { name: `r_${tag}` });
    const hat = g.mul(x, r, { name: `${tag}̂` });
    const out = g.mul(hat, gamma, { name: `${tag}̄` });
    return { out, parts: [sq, ms, shifted, r, hat] };
  };

  const g1 = g.param('γ1', ['d'], [1, 1, 1, 1]);
  const n1 = norm(e, g1, 'e');
  const Wq = g.param('Wq', ['qd', 'd'], values(4, 4, 2));
  const Wk = g.param('Wk', ['dh', 'd'], values(2, 4, 3));
  const Wv = g.param('Wv', ['dh', 'd'], values(2, 4, 4));
  const Wo = g.param('Wo', ['d', 'qd'], values(4, 4, 5));
  const q = g.matmul(n1.out, Wq.T, { name: 'q' });
  const k = g.matmul(n1.out, Wk.T, { name: 'k' });
  const v = g.matmul(n1.out, Wv.T, { name: 'v' });
  const heads = g.split(q, 'dh', 1, { names: ['q1', 'q2'] });
  const gq = g.param('γq', ['dh'], [1, 1]);
  const M = g.buffer('M', [T, T], [[0, -Infinity], [0, 0]]);
  const branches = heads.map((qh, h) => {
    const qn = norm(qh, gq, `q${h + 1}`);
    const kT = g.transpose(k, 0, 1, { name: `kᵀ${h + 1}` });
    const S = g.matmul(qn.out, kT, { name: `S${h + 1}` });
    const masked = g.add(S, M, { name: `S̄${h + 1}` });
    const ex = g.exp(masked, { name: `ex${h + 1}` });
    const Z = g.sum(ex, -1, { keepdim: true, name: `Z${h + 1}` });
    const P = g.div(ex, Z, { name: `P${h + 1}` });
    const vh = g.narrow(v, 1, 0, 'dh', { name: `v${h + 1}` });
    const o = g.matmul(P, vh, { name: `o${h + 1}` });
    const body = [
      leaf(qh),
      tree('q_norm', [...leaves(...qn.parts), ...(h === 0 ? [leaf(gq)] : []), leaf(qn.out)]),
      ...leaves(kT, S, ...(h === 0 ? [M] : []), masked),
      tree('softmax', leaves(ex, Z, P)),
      ...leaves(vh, o),
    ];
    return { o, S, P, qn, body };
  });
  const o = g.cat(branches.map((b) => b.o), 1, { name: 'o' });
  const att = g.matmul(o, Wo.T, { name: 'att' });
  const a = g.add(e, att, { name: 'a' });
  const Wg = g.param('Wg', ['f', 'd'], values(6, 4, 6));
  const Wu = g.param('Wu', ['f', 'd'], values(6, 4, 7));
  const Wd = g.param('Wd', ['d', 'f'], values(4, 6, 8));
  const gate = g.matmul(a, Wg.T, { name: 'gate' });
  const up = g.matmul(a, Wu.T, { name: 'up' });
  const s = g.silu(gate, { name: 's' });
  const m = g.mul(s, up, { name: 'm' });
  const down = g.matmul(m, Wd.T, { name: 'down' });
  const x = g.add(a, down, { name: 'x' });
  const W = g.param('W', ['V', 'd'], values(5, 4, 9));
  const z = g.matmul(x, W.T, { name: 'z' });
  const L = g.logsumexp(z, -1, { keepdim: true, name: 'LSE' });
  const logp = g.sub(z, L, { name: 'log p' });
  const lt = g.nll_loss(logp, labels, { reduction: 'none', name: 'ℓt' });
  const loss = g.mean(lt, { name: 'loss' });

  const root: LayoutTree = tree(undefined, [
    ...leaves(ids, labels),
    tree('embed', leaves(E, e)),
    tree('layer', [
      tree('input_layernorm', [...leaves(...n1.parts), leaf(g1), leaf(n1.out)]),
      tree('self_attn', [
        ...leaves(Wq, q, Wk, k, Wv, v),
        { kind: 'parallel', branches: branches.map((b) => tree(undefined, b.body)) },
        ...leaves(o, Wo, att),
      ]),
      leaf(a),
      tree('mlp', leaves(Wg, gate, Wu, up, s, m, Wd, down)),
      leaf(x),
    ], 'DecoderLayer'),
    tree('lm_head', leaves(W, z)),
    tree('loss', leaves(L, logp, lt, loss)),
  ]);
  return { g, root, E, e, g1, n1, q, k, v, gq, M, branches, o, att, a, gate, up, s, m, down, x, z, L, logp, loss, W, Wq };
}

function resolve(g: ReturnType<typeof model>['g'], root: LayoutTree) {
  const result = autoLayout(root);
  const ctx = { visible: g.nodes, aligned: classAlignment(g.nodes), targetOf: targetColumn, focus: 0 };
  const { blocks } = resolveLayout(result.layout, ctx);
  const of = (n: Node) => blocks.find((b) => b.node === n)!;
  return { ...result, blocks, of };
}

const ys = (block: Block) => block.elements.map((el) => el.y);
const xs = (block: Block) => block.elements.map((el) => el.x);
const baseY = (block: Block) => Math.min(...ys(block));

function box(block: Block) {
  const [w, d] = block.footprint;
  const x = xs(block);
  const z = block.elements.map((el) => el.z);
  const plate = block.plate;
  return {
    x0: Math.min(...x) - w / 2, x1: Math.max(...x) + w / 2,
    z0: Math.min(...z) - d / 2, z1: Math.max(...z) + d / 2,
    px0: plate === undefined ? Infinity : plate.x - plate.w / 2, px1: plate === undefined ? -Infinity : plate.x + plate.w / 2,
    y0: blockSpan(block).bottom, y1: blockSpan(block).top,
  };
}

describe('autoLayout', () => {
  it('stacks the main column bottom-up, one level per planar node', () => {
    const t = model();
    const { of } = resolve(t.g, t.root);
    expect(baseY(of(t.E))).toBeCloseTo(0, 9);
    expect(baseY(of(t.e))).toBeCloseTo(AUTO_LEVEL, 9);
    const column = [t.e, t.n1.parts[0], t.n1.parts[4], t.n1.out];
    column.slice(1).forEach((n, i) => expect(baseY(of(n)) - baseY(of(column[i]))).toBeCloseTo(AUTO_LEVEL, 9));
    expect(baseY(of(t.a)) - baseY(of(t.att))).toBeCloseTo(AUTO_LEVEL, 9);
    expect(baseY(of(t.x)) - baseY(of(t.down))).toBeCloseTo(AUTO_LEVEL, 9);
    expect(baseY(of(t.logp)) - baseY(of(t.z))).toBeCloseTo(AUTO_LEVEL, 9);
    expect(baseY(of(t.loss))).toBeGreaterThan(baseY(of(t.logp)));
  });

  it('draws matmuls with a two-dimensional right operand as boards and puts boards sharing an input side by side', () => {
    const t = model();
    const { layout, of } = resolve(t.g, t.root);
    const boards = layout.filter((item) => item.template === 'board');
    expect(boards.map((item) => item.template === 'board' && item.out.name)).toEqual(['q', 'k', 'v', 'S1', 'o1', 'S2', 'o2', 'att', 'gate', 'up', 'down', 'z']);
    const placed = layout.flatMap((item) => (item.template === 'stack' ? item.nodes : item.template === 'board' ? [item.a, item.b, item.out] : [item.node]));
    expect(placed).not.toContain(t.Wq);
    expect(placed).not.toContain(t.W);
    const row = [t.q, t.k, t.v].map((n) => box(of(n)));
    for (const b of row) expect(b.y0).toBeCloseTo(row[0].y0, 9);
    expect(row[0].px1).toBeLessThan(row[1].px0);
    expect(row[1].px1).toBeLessThan(row[2].px0);
    expect(box(of(t.gate)).px1).toBeLessThan(box(of(t.up)).px0);
    expect(baseY(of(t.q)) - baseY(of(t.n1.out))).toBeCloseTo(AUTO_LEVEL, 9);
    expect(Math.max(...of(t.z).elements.map((el) => el.z))).toBeCloseTo(0, 9);
  });

  it('gives each parallel branch its own lane and continues above the highest branch', () => {
    const t = model();
    const { of } = resolve(t.g, t.root);
    const [b1, b2] = t.branches;
    const lane = (b: typeof b1) => [b.qn.out, b.S, b.P, b.o].map((n) => box(of(n)));
    const right1 = Math.max(...lane(b1).map((b) => Math.max(b.x1, b.px1)));
    const left2 = Math.min(...lane(b2).map((b) => Math.min(b.x0, b.px0)));
    expect(right1).toBeLessThan(left2);
    expect(baseY(of(b1.S))).toBeCloseTo(baseY(of(b2.S)), 9);
    expect(baseY(of(t.o)) - baseY(of(b1.o))).toBeCloseTo(AUTO_LEVEL, 9);
  });

  it('puts one-dimensional weights and buffers on a side lane at the height of their first user', () => {
    const t = model();
    const { of, blocks } = resolve(t.g, t.root);
    for (const [side, user] of [[t.g1, t.n1.out], [t.gq, t.branches[0].qn.out], [t.M, t.g.node('S̄1')!]] as const) {
      expect(baseY(of(side))).toBeCloseTo(baseY(of(user)), 9);
      const others = blocks.filter((b) => b !== of(side) && Math.abs(baseY(b) - baseY(of(side))) < 1e-9);
      const right = Math.max(...others.map((b) => Math.max(box(b).x1, box(b).px1)));
      expect(box(of(side)).px0).toBeGreaterThan(right + AUTO_SIDE_GAP - 1e-9);
    }
  });

  it('keeps every pair of blocks apart', () => {
    const t = model();
    const { blocks } = resolve(t.g, t.root);
    const boxes = blocks.map((b) => ({ node: b.node.name, ...box(b) }));
    for (let i = 0; i < boxes.length; i++) {
      for (let j = i + 1; j < boxes.length; j++) {
        const p = boxes[i];
        const q = boxes[j];
        const sameBand = p.y0 < q.y1 - 1e-9 && q.y0 < p.y1 - 1e-9;
        const overlapX = p.x0 < q.x1 && q.x0 < p.x1;
        const overlapZ = p.z0 < q.z1 && q.z0 < p.z1;
        expect(sameBand && overlapX && overlapZ, `${p.node} and ${q.node}`).toBe(false);
      }
    }
  });

  it('puts outermost modules on the left, nested ones on the right, and merges same-named modules across branches', () => {
    const t = model();
    const { modules } = resolve(t.g, t.root);
    expect(modules.map((m) => `${m.side} ${m.title}`)).toEqual([
      'left embed', 'left layer', 'right input_layernorm', 'right self_attn', 'right q_norm', 'right softmax', 'right mlp', 'left lm_head', 'left loss',
    ]);
    const qNorm = modules.find((m) => m.title === 'q_norm')!;
    expect(qNorm.nodes).toContain(t.branches[0].qn.out);
    expect(qNorm.nodes).toContain(t.branches[1].qn.out);
    expect(modules.find((m) => m.title === 'layer')!.subtitle).toBe('DecoderLayer');
    expect(modules.find((m) => m.title === 'lm_head')!.nodes).toContain(t.W.T);
  });

  it('accepts a tree without modules or parallels', () => {
    const g = graph({ dims: { d: 3 } });
    const h = g.input('h', [2, 'd'], [[1, 2, 3], [4, 5, 6]]);
    const y = g.exp(h, { name: 'y' });
    const result = autoLayout(tree(undefined, leaves(h, y)));
    expect(result.modules).toEqual([]);
    expect(result.layout).toHaveLength(1);
  });
});
