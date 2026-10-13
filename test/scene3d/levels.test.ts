import { describe, expect, it } from 'vitest';
import type { Node } from '../../src/autograd';
import { graph } from '../../src/core';
import '../../src/explain';
import type { Block } from '../../src/scene3d/layout';
import { levelPlan } from '../../src/scene3d/levels';
import type { StepModule, StepsSpec } from '../../src/steps';

const T = 2;
const H = 3;
const DH = 2;

function headGraph() {
  const g = graph({ dims: { T, H, dh: DH } });
  const x = g.input('x', [T, H, DH], Array.from({ length: T }, (_, t) => Array.from({ length: H }, (_, h) => Array.from({ length: DH }, (_, c) => Math.sin(1 + t * 1.3 + h * 0.7 + c * 0.4)))));
  const gamma = g.param('gamma', [DH], [0.9, 1.1]);
  const sq = g.pow(x, 2, { name: 'sq' });
  const ms = g.mean(sq, -1, { keepdim: true, name: 'ms' });
  const r = g.rsqrt(g.add(ms, 1e-6, { name: 'shifted' }), { name: 'r' });
  const hat = g.mul(x, r, { name: 'hat' });
  const y = g.mul(hat, gamma, { name: 'y' });
  const loss = g.loss(g.sum(y, null, { name: 'loss' }));
  const inside = [sq, ms, g.node('shifted')!, r, hat, y];
  const module: StepModule = {
    id: 'norm',
    parent: null,
    title: 'norm',
    subtitle: '',
    inputs: [x],
    outputs: [y],
    nodes: inside,
    params: [gamma],
    fixed: false,
    folded: false,
    segment: 'norm',
    forward: {},
    backward: {},
  };
  const step = (show: Node[], extra: Partial<StepsSpec['forward'][number]> = {}) => ({ show, title: show.map((n) => n.name).join(', '), segment: 'norm', module: 'norm', ...extra });
  const spec: StepsSpec = {
    forward: [
      { show: [x, gamma], title: 'inputs', kind: 'inputs' },
      step([sq]),
      step([ms, g.node('shifted')!, r]),
      step([hat]),
      step([y]),
      { show: [loss], title: 'loss', segment: 'loss' },
    ],
    backward: 'reverse',
    modules: [module],
  };
  return { spec, x, ms, r, hat, y };
}

const blockOf = (blocks: Block[], node: Node) => blocks.find((b) => b.node === node && b.copyOf === undefined)!;

describe('levelPlan with rank-3 tensors', () => {
  it('draws (T, H, dh) as T rows of H side-by-side boards with a gap between heads', () => {
    const { spec, x, hat } = headGraph();
    const { blocks } = levelPlan(spec).resolve({ folded: () => false, aligned: new Map() });
    for (const node of [x, hat]) {
      const block = blockOf(blocks, node);
      expect([block.rows, block.cols, block.shape]).toEqual([T, H * DH, 'bar']);
      const element = block.elements.find((e) => e.index.join() === '1,2,1')!;
      expect([element.row, element.col]).toEqual([1, 2 * DH + 1]);
      const xs = Array.from({ length: H * DH }, (_, col) => block.elements.find((e) => e.row === 0 && e.col === col)!.x);
      const steps = xs.slice(1).map((v, i) => v - xs[i]);
      const inside = steps.filter((_, i) => (i + 1) % DH !== 0);
      const between = steps.filter((_, i) => (i + 1) % DH === 0);
      expect(new Set(inside.map((v) => v.toFixed(6))).size).toBe(1);
      for (const gap of between) expect(gap).toBeGreaterThan(inside[0]);
    }
  });

  it('draws one disc per row and head for a (T, H, 1) reduction, beside each head', () => {
    const { spec, x, ms, r } = headGraph();
    const { blocks } = levelPlan(spec).resolve({ folded: () => false, aligned: new Map() });
    const xBlock = blockOf(blocks, x);
    for (const node of [ms, r]) {
      const block = blockOf(blocks, node);
      expect([block.shape, block.rows, block.cols]).toEqual(['disc', T, H]);
      for (let h = 0; h < H; h++) {
        const disc = block.elements.find((e) => e.row === 0 && e.col === h)!;
        const last = xBlock.elements.find((e) => e.row === 0 && e.col === (h + 1) * DH - 1)!;
        const next = xBlock.elements.find((e) => e.row === 0 && e.col === (h + 1) * DH);
        expect(disc.x).toBeGreaterThan(last.x);
        if (next !== undefined) expect(disc.x).toBeLessThan(next.x);
      }
    }
  });

  it('hides the inside of a folded module and keeps its output', () => {
    const { spec, hat, y } = headGraph();
    const { blocks, boxes } = levelPlan(spec).resolve({ folded: (id) => id === 'norm', aligned: new Map() });
    expect(blockOf(blocks, hat).hidden).toBe(true);
    expect(blockOf(blocks, y).hidden).toBe(false);
    expect(boxes.map((b) => b.module.id)).toEqual(['norm']);
  });
});

function flatGraph() {
  const D = 4;
  const g = graph({ dims: { T, d: D } });
  const ids = g.label('ids', [1, 0], { tokens: ['a', 'b', 'c'] });
  const E = g.param('E', [3, D], Array.from({ length: 3 }, (_, v) => Array.from({ length: D }, (_, c) => Math.cos(v + c * 0.6))));
  const e = g.embedding(ids, E, { name: 'e' });
  const gamma = g.param('gamma', [D], [0.9, 1.1, 1.0, 0.8]);
  const sq = g.pow(e, 2, { name: 'sq' });
  const ms = g.mean(sq, -1, { keepdim: true, name: 'ms' });
  const r = g.rsqrt(g.add(ms, 1e-6, { name: 'shifted' }), { name: 'r' });
  const hat = g.mul(e, r, { name: 'hat' });
  const y = g.mul(hat, gamma, { name: 'y' });
  const loss = g.loss(g.sum(y, null, { name: 'loss' }));
  const module: StepModule = {
    id: 'norm',
    parent: null,
    title: 'norm',
    subtitle: '',
    inputs: [e],
    outputs: [y],
    nodes: [sq, ms, g.node('shifted')!, r, hat, y],
    params: [gamma],
    fixed: false,
    folded: false,
    segment: 'norm',
    forward: {},
    backward: {},
  };
  const step = (show: Node[]) => ({ show, title: show.map((n) => n.name).join(', '), segment: 'norm', module: 'norm' });
  const spec: StepsSpec = {
    forward: [
      { show: [ids, E, gamma], title: 'inputs', kind: 'inputs' },
      { show: [e], title: 'e', segment: 'embed' },
      step([sq]),
      step([ms, g.node('shifted')!, r]),
      step([hat]),
      step([y]),
      { show: [loss], title: 'loss', segment: 'loss' },
    ],
    backward: 'reverse',
    modules: [module],
  };
  return { spec, e, sq, ms, r, hat, y, D };
}

describe('levelPlan with rank-2 activations', () => {
  it('draws (T, d) activations as plates of bars and only the (T, 1) statistics as discs', () => {
    const { spec, e, sq, ms, r, hat, y, D } = flatGraph();
    const { blocks } = levelPlan(spec).resolve({ folded: () => false, aligned: new Map() });
    for (const node of [e, sq, hat, y]) {
      const block = blockOf(blocks, node);
      expect([node.name, block.shape, block.rows, block.cols, block.plate !== undefined]).toEqual([node.name, 'bar', T, D, true]);
    }
    for (const node of [ms, r]) {
      const block = blockOf(blocks, node);
      expect([node.name, block.shape, block.rows, block.cols]).toEqual([node.name, 'disc', T, 1]);
    }
  });
});

describe('levelPlan with a matrix operand shaped like a statistic', () => {
  it('draws the right operand of a matmul as bars even when the output has the shape of the left operand', () => {
    const g = graph({ dims: {} });
    const q = g.input('q', [2, 3, 3], Array.from({ length: 2 }, (_, t) => Array.from({ length: 3 }, (_, h) => [t + h, t - h, 0.5])));
    const k = g.input('k', [3, 3], [[1, 0, 2], [0, 1, -1], [2, 1, 0]]);
    const kt = g.transpose(k, 0, 1, { name: 'kt' });
    const s = g.matmul(q, kt, { name: 's' });
    const loss = g.loss(g.sum(s, null, { name: 'loss' }));
    const spec: StepsSpec = {
      forward: [
        { show: [q, k], title: 'inputs', kind: 'inputs' },
        { show: [kt], title: 'kt', segment: 'kt' },
        { show: [s], title: 's', segment: 's' },
        { show: [loss], title: 'loss', segment: 'loss' },
      ],
      backward: 'reverse',
    };
    const { blocks } = levelPlan(spec).resolve({ folded: () => false, aligned: new Map() });
    const block = blockOf(blocks, kt);
    expect([block.shape, block.cols]).toEqual(['bar', 3]);
    expect(blocks.every((b) => b.elements.every((e) => Number.isFinite(e.x) && Number.isFinite(e.z)))).toBe(true);
  });
});

describe('levelPlan with views and head-selecting boards', () => {
  it('draws a view on its input and reads each head of a rank-3 operand under its own columns', () => {
    const g = graph({ dims: { T, H: 2, dh: DH } });
    const x = g.input('x', [T, 2 * DH], [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]);
    const K = g.param('K', [DH, 3], [[1, 0, 1], [0, 1, 1]]);
    const flat = g.mul(x, 2, { name: 'flat' });
    const heads = g.view(flat, T, 2, DH);
    const S = g.select_matmul(heads, K, 1, { name: 'S' });
    const loss = g.loss(g.sum(S, null, { name: 'loss' }));
    const module: StepModule = {
      id: 'att', parent: null, title: 'att', subtitle: '', inputs: [x], outputs: [S], nodes: [flat, heads, S], params: [K], fixed: false, folded: false, segment: 'att', forward: {}, backward: {},
    };
    const spec: StepsSpec = {
      forward: [
        { show: [x, K], title: 'inputs', kind: 'inputs' },
        { show: [flat, heads], title: 'flat', segment: 'att', module: 'att' },
        { show: [S], title: 'S', segment: 'att', module: 'att' },
        { show: [loss], title: 'loss', segment: 'loss' },
      ],
      backward: 'reverse',
      modules: [module],
    };
    const layout = levelPlan(spec).resolve({ folded: () => false, aligned: new Map() });
    expect(layout.blocks.some((b) => b.node === heads)).toBe(false);
    expect(layout.sourceOf.get(heads)).toBe(flat);
    const board = layout.boards.find((b) => b.op === S)!;
    expect(board.a.node).toBe(flat);
    const flatXs = blockOf(layout.blocks, flat).elements.filter((e) => e.row === 0).map((e) => e.x);
    const outXs = board.out.elements.filter((e) => e.row === 0).map((e) => e.x);
    expect(outXs[0]).toBeCloseTo(flatXs[DH]);
    expect(board.y).toBeGreaterThan(blockOf(layout.blocks, flat).elements[0].y);
  });
});

function groupedGraph() {
  const g = graph({ dims: {} });
  const x = g.input('x', [4, 2], [[1, 2], [3, 4], [5, 6], [7, 8]]);
  const ids = g.input('ids', [4], [1, 1, 3, 3]);
  const W = g.param('W', [4, 3, 2], Array.from({ length: 4 }, (_, e) => Array.from({ length: 3 }, (_, o) => [e + o * 0.1, e - o * 0.2])));
  const xs = g.mul(x, 2, { name: 'xs' });
  const group = g.add(ids, 0, { name: 'expert_idx' });
  const out = g.grouped_linear(xs, W, group, { name: 'out' });
  const loss = g.loss(g.sum(out, null, { name: 'loss' }));
  const spec: StepsSpec = {
    forward: [
      { show: [x, ids, W], title: 'inputs', kind: 'inputs' },
      { show: [xs, group], title: 'xs', segment: 'a' },
      { show: [W, out], title: 'out', segment: 'a' },
      { show: [loss], title: 'loss', segment: 'loss' },
    ],
    backward: 'reverse',
  };
  return { spec, xs, W, out };
}

describe('levelPlan with grouped_linear', () => {
  it('draws the weight as one slab per group beside the rows of that group', () => {
    const { spec, xs, W, out } = groupedGraph();
    const { blocks } = levelPlan(spec).resolve({ folded: () => false, aligned: new Map() });
    const weight = blockOf(blocks, W);
    expect(weight.slabs!.map((s) => s.used)).toEqual([false, true, false, true]);
    expect(weight.rowLabels.map((l) => l.text)).toEqual(['expert_idx=0', 'expert_idx=1', 'expert_idx=2', 'expert_idx=3']);
    const zOf = (block: Block, pick: (index: number[]) => boolean) => block.elements.filter((e) => pick(e.index)).map((e) => e.z);
    const mid = (zs: number[]) => (Math.min(...zs) + Math.max(...zs)) / 2;
    const xsBlock = blockOf(blocks, xs);
    expect(mid(zOf(weight, (i) => i[0] === 1))).toBeCloseTo(mid(zOf(xsBlock, (i) => i[0] < 2)), 9);
    expect(mid(zOf(weight, (i) => i[0] === 3))).toBeCloseTo(mid(zOf(xsBlock, (i) => i[0] >= 2)), 9);
    for (let g = 1; g < 4; g++) expect(Math.min(...zOf(weight, (i) => i[0] === g))).toBeGreaterThan(Math.max(...zOf(weight, (i) => i[0] === g - 1)));
    expect(blockOf(blocks, out).rowGroups).toEqual([1, 1, 3, 3]);
    expect(xsBlock.rowLabels.filter((l) => l.text !== '').map((l) => l.text)).toEqual(['expert_idx=1', 'expert_idx=3']);
  });
});

