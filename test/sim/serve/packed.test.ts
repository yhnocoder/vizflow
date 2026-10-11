// @vitest-environment jsdom
import { afterEach, describe, expect, it } from 'vitest';
import type { Node } from '../../../src/core';
import { classAlignment, resolveLayout, targetColumn } from '../../../src/scene3d/layout';
import { type PackedRow, packedAttention, packedMask, packedRequests } from '../../../src/sim/serve/packed';
import { createSteps } from '../../../src/steps';
import { type Mounted, mountPanels, mountSteps } from '../../../src/ui';

const SHAPES: Record<string, PackedRow[]> = {
  mixed: [
    { id: 'C', cached: 4, n: 2 },
    { id: 'B', cached: 0, n: 6 },
    { id: 'A', cached: 8, n: 1 },
  ],
  prefill: [
    { id: 'A', cached: 0, n: 4 },
    { id: 'B', cached: 0, n: 3 },
  ],
  decode: [
    { id: 'A', cached: 5, n: 1 },
    { id: 'B', cached: 3, n: 1 },
    { id: 'C', cached: 7, n: 1 },
  ],
};

type Matrix = number[][];

function matrix(node: Node): Matrix {
  const [rows, cols] = node.shape;
  return Array.from({ length: rows }, (_, r) => Array.from({ length: cols }, (_, c) => node.value.get([r, c])));
}

function rowsOf(node: Node, from: number, count: number): Matrix {
  return matrix(node).slice(from, from + count);
}

function times(a: Matrix, bT: Matrix): Matrix {
  return a.map((row) => bT.map((col) => row.reduce((sum, v, i) => sum + v * col[i], 0)));
}

function causalAttention(q: Matrix, K: Matrix, V: Matrix, offset: number): Matrix {
  const scale = 1 / Math.sqrt(q[0].length);
  return q.map((query, j) => {
    const visible = K.slice(0, offset + j + 1);
    const scores = visible.map((key) => key.reduce((sum, v, i) => sum + v * query[i], 0) * scale);
    const peak = Math.max(...scores);
    const weights = scores.map((s) => Math.exp(s - peak));
    const total = weights.reduce((a, b) => a + b, 0);
    return V[0].map((_, c) => weights.reduce((sum, w, p) => sum + (w / total) * V[p][c], 0));
  });
}

function spans(rows: PackedRow[]) {
  let start = 0;
  let keyStart = 0;
  return rows.map((row) => {
    const span = { ...row, start, keyStart };
    start += row.n;
    keyStart += row.cached + row.n;
    return span;
  });
}

describe('packedAttention', () => {
  const mounted: Mounted[] = [];
  afterEach(() => {
    mounted.splice(0).forEach((m) => m.dispose());
    document.body.replaceChildren();
  });

  for (const [name, rows] of Object.entries(SHAPES)) {
    describe(name, () => {
      it('builds a block-diagonal causal mask that includes the cached columns', () => {
        const { nodes } = packedAttention(rows);
        const M = matrix(nodes.M);
        const owners = spans(rows);
        const queryOwner = owners.flatMap((r) => Array.from({ length: r.n }, (_, j) => ({ id: r.id, position: r.cached + j })));
        const keyOwner = owners.flatMap((r) => Array.from({ length: r.cached + r.n }, (_, p) => ({ id: r.id, position: p })));
        expect(M.length).toBe(queryOwner.length);
        M.forEach((row, r) => {
          expect(row.length).toBe(keyOwner.length);
          row.forEach((value, c) => {
            const visible = queryOwner[r].id === keyOwner[c].id && keyOwner[c].position <= queryOwner[r].position;
            expect(value).toBe(visible ? 0 : -Infinity);
          });
        });
      });

      it('gives every request the same output as causal attention over its own cached and new rows', () => {
        const { graph, nodes } = packedAttention(rows, { seed: 7 });
        const W = (n: string) => matrix(graph.node(n)!);
        const [Wq, Wk, Wv] = [W('Wq'), W('Wk'), W('Wv')];
        for (const r of spans(rows)) {
          const x = rowsOf(nodes.x, r.start, r.n);
          const cachedK = r.cached === 0 ? [] : matrix(graph.node(`K_cache_${r.id}`)!);
          const cachedV = r.cached === 0 ? [] : matrix(graph.node(`V_cache_${r.id}`)!);
          const expected = causalAttention(times(x, Wq), [...cachedK, ...times(x, Wk)], [...cachedV, ...times(x, Wv)], r.cached);
          const actual = rowsOf(nodes.O, r.start, r.n);
          expected.forEach((row, j) => row.forEach((v, c) => expect(actual[j][c]).toBeCloseTo(v, 12)));
        }
      });

      it('takes the last row of each request for logits', () => {
        const { nodes } = packedAttention(rows);
        const O = matrix(nodes.O);
        const expected = spans(rows).map((r) => O[r.start + r.n - 1]);
        expect(matrix(nodes.logits)).toEqual(expected);
      });

      it('builds steps, panels and the scene layout without errors and has no backward steps', () => {
        const page = packedAttention(rows);
        const steps = createSteps(page.graph, page.steps);
        expect(steps.list.every((info) => info.dir === 'forward')).toBe(true);
        expect(steps.list.map((info) => info.title)).toContain('加上分块的因果 mask');
        for (let i = 0; i < steps.list.length; i++) expect(steps.text(i).body).not.toMatch(/\{\{/);
        const stepsEl = document.createElement('div');
        const side = document.createElement('div');
        document.body.append(stepsEl, side);
        mounted.push(mountSteps(stepsEl, steps), mountPanels(side, page.graph, page.panels, { steps }));
        const ctx = { visible: steps.visible, aligned: classAlignment(page.graph.nodes), targetOf: targetColumn, focus: 0 };
        const { blocks } = resolveLayout(page.layout, ctx);
        expect(new Set(blocks.map((b) => b.node))).toEqual(new Set(steps.visible.filter((n) => n.kind !== 'label')));
        steps.dispose();
      });
    });
  }

  it('matches the packed_mask buffer function of research/base for the same lengths', () => {
    const researchBasePackedMask = (lens: number[], cached: number[]) => {
      const queries: { seq: number; pos: number }[] = [];
      const keys: { seq: number; pos: number }[] = [];
      lens.forEach((n, seq) => {
        for (let j = 0; j < cached[seq] + n; j++) keys.push({ seq, pos: j });
        for (let j = 0; j < n; j++) queries.push({ seq, pos: cached[seq] + j });
      });
      return queries.map((q) => keys.map((k) => (k.seq === q.seq && k.pos <= q.pos ? 0 : -Infinity)));
    };
    for (const rows of Object.values(SHAPES)) {
      expect(packedMask(packedRequests(rows))).toEqual(researchBasePackedMask(rows.map((r) => r.n), rows.map((r) => r.cached)));
    }
  });

  it('rejects malformed rows', () => {
    expect(() => packedAttention([])).toThrow('at least one request');
    expect(() => packedAttention([{ id: 'A', cached: 0, n: 0 }])).toThrow("n = 0");
    expect(() => packedAttention([{ id: 'A', cached: 0, n: 1 }, { id: 'A', cached: 2, n: 1 }])).toThrow('appears twice');
  });
});
