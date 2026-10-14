import { describe, expect, it } from 'vitest';
import type { Node } from '../../src/autograd';
import { graph } from '../../src/core';
import { explain } from '../../src/explain';
import type { SceneController } from '../../src/scene3d/controller';
import { buildEdges } from '../../src/scene3d/edges';
import { classAlignment, elementAt, plates, resolveLayout, rowCol, rowsCols, stack, targetColumn } from '../../src/scene3d/layout';
import { type LevelLayout, levelPlan } from '../../src/scene3d/levels';
import type { Renderable } from '../../src/scene3d/renderable';
import type { StepsSpec } from '../../src/steps';
import { forEachIndex } from '../../src/tensor';

const H = 2;
const N = 2;
const C = 3;
const D = 2;

function values(shape: number[], seed: number): number[] {
  const size = shape.reduce((a, b) => a * b, 1);
  return Array.from({ length: size }, (_, i) => Math.sin(seed + i * 0.73) * 0.8);
}

function chunked() {
  const g = graph({ dims: {} });
  const x = g.input('x', [H, N * C, D], values([H, N * C, D], 1));
  const decay = g.input('decay', [H, N, 1, 1], values([H, N, 1, 1], 2));
  const xc = g.reshape(x, [H, N, C, D], { name: 'xc' });
  const xt = g.transpose(xc, -1, -2, { name: 'xt' });
  const s = g.matmul(xc, xt, { name: 's' });
  const sd = g.add(s, decay, { name: 'sd' });
  const e = g.exp(sd, { name: 'e' });
  const p = g.select(e, 1, 1, { name: 'p' });
  const back = g.reshape(xc, [H, N * C, D], { name: 'back' });
  const loss = g.loss(g.add(g.sum(p, null, { name: 'ps' }), g.sum(back, null, { name: 'bs' }), { name: 'loss' }));
  const spec: StepsSpec = {
    forward: [
      { title: 'inputs', show: [x, decay] },
      { title: 'xc', show: [xc] },
      { title: 's', show: [xt, s] },
      { title: 'sd', show: [sd] },
      { title: 'e', show: [e] },
      { title: 'p', show: [p] },
      { title: 'back', show: [back] },
      { title: 'loss', show: [loss] },
    ],
    backward: 'reverse',
  };
  return { g, spec, nodes: { x, decay, xc, xt, s, sd, e, p, back } };
}

function resolve(spec: StepsSpec): LevelLayout {
  const lines = spec.forward.map((step) => (Array.isArray(step.show) ? step.show : [step.show]));
  return levelPlan(spec, { lines, sides: [], merges: [] }).resolve({ folded: () => false, aligned: new Map() });
}

const blockOf = (layout: LevelLayout, node: Node) => layout.blocks.find((b) => b.node === node && b.copyOf === undefined)!;

function sceneOf(g: ReturnType<typeof chunked>['g'], layout: LevelLayout): SceneController {
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

const elementEdges = (g: ReturnType<typeof chunked>['g'], op: Node) =>
  explain(g)
    .edges(op)
    .map((e) => {
      if (!('index' in e.from) || !('index' in e.to)) throw new Error(`${op.name} edges connect elements`);
      return { from: e.from.node, a: e.from.index, b: e.to.index };
    });

describe('rank-4 tensors', () => {
  it('draws (A, B, R, C) as A B plates of R rows and C columns, plate a B + b', () => {
    const { nodes: n } = chunked();
    expect(rowsCols(n.xc)).toEqual([C, H * N * D]);
    expect(plates(n.xc)).toEqual({ count: H * N, width: D });
    forEachIndex(n.xc.shape, ([h, i, r, c]) => expect(rowCol(n.xc, [h, i, r, c])).toEqual([r, (h * N + i) * D + c]));
  });

  it('rejects tensors with more than four dimensions', () => {
    const g = graph();
    const x = g.input('x', [1, 2, 1, 2, 1], [1, 2, 3, 4]);
    const ctx = { visible: g.nodes, aligned: classAlignment(g.nodes), targetOf: targetColumn, focus: 0 };
    expect(() => resolveLayout([stack(x)], ctx)).toThrow("scene3d: node 'x' with shape (1, 2, 1, 2, 1) has more than four dimensions");
  });

  it('connects a rank-4 broadcast add to the same (h, i) decay for every cell of that plate', () => {
    const { g, spec, nodes: n } = chunked();
    const fromDecay = elementEdges(g, n.sd).filter((e) => e.from === n.decay);
    expect(fromDecay).toHaveLength(H * N * C * C);
    for (const { a, b } of fromDecay) expect(a).toEqual([b[0], b[1], 0, 0]);
    const layout = resolve(spec);
    expect(blockOf(layout, n.sd).chunks).toHaveLength(H * N);
  });

  it('draws a rank-4 batched matmul as one product plate per (h, i) and routes each cell to its own plate', () => {
    const { g, spec, nodes: n } = chunked();
    const layout = resolve(spec);
    const board = layout.boards.find((b) => b.op === n.s)!;
    expect([board.a.node, board.b.node, board.out.node]).toEqual([n.xc, n.xt, n.s]);
    expect(board.out.chunks).toHaveLength(H * N);
    const drawn = buildEdges(sceneOf(g, layout)).filter((d) => d.edge.op === n.s);
    expect(drawn.filter((d) => d.edge.arg === -1)).toHaveLength(H * N * C * D * C);
    for (const d of drawn) {
      const cell = 'cell' in d.edge.to ? d.edge.to.cell.index : 'cell' in d.edge.from ? d.edge.from.cell.index : [];
      const [h, i, r, k, j] = cell;
      const [from, to] = d.endpoints();
      const tile = elementAt(board.b, [h, i, k, j]);
      if (d.edge.arg === 0 && j === 0) expect(from[0]).toBeCloseTo(elementAt(board.a, [h, i, r, k]).x, 9);
      if (d.edge.arg === -1 && k === D - 1) expect(to[0]).toBeCloseTo(elementAt(board.out, [h, i, r, j]).x, 9);
      if (d.edge.arg === -1) expect(from[0]).toBeCloseTo(tile.x, 9);
    }
  });

  it('selects along dim 1 from rank 4 to rank 3 and keeps each head under its selected plate', () => {
    const { g, spec, nodes: n } = chunked();
    const edges = elementEdges(g, n.p);
    expect(edges).toHaveLength(H * C * C);
    for (const { a, b } of edges) expect(a).toEqual([b[0], 1, b[1], b[2]]);
    const layout = resolve(spec);
    const e = blockOf(layout, n.e);
    const p = blockOf(layout, n.p);
    forEachIndex(n.p.shape, ([h, r, c]) => expect(elementAt(p, [h, r, c]).x).toBeCloseTo(elementAt(e, [h, 1, r, c]).x, 9));
  });

  it('maps reshape between (H, n C, d) and (H, n, C, d) element by element', () => {
    const { g, spec, nodes: n } = chunked();
    for (const { a, b } of elementEdges(g, n.xc)) expect(a).toEqual([b[0], b[1] * C + b[2], b[3]]);
    for (const { a, b } of elementEdges(g, n.back)) expect(a).toEqual([b[0], Math.floor(b[1] / C), b[1] % C, b[2]]);
    const layout = resolve(spec);
    const drawn = buildEdges(sceneOf(g, layout));
    for (const op of [n.xc, n.back]) expect(drawn.filter((d) => d.edge.op === op)).toHaveLength(H * N * C * D);
  });
});
