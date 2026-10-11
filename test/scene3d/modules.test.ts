import { describe, expect, it } from 'vitest';
import { graph } from '../../src/core';
import '../../src/explain';
import { classAlignment, resolveLayout, stack, targetColumn } from '../../src/scene3d/layout';
import {
  BRACE_RADIUS,
  MODULE_GAP,
  MODULE_PAD,
  MODULE_STEP,
  blockSpan,
  bracePoints,
  braceTip,
  checkModules,
  layoutBraces,
} from '../../src/scene3d/modules';
import { HEIGHT_CAP } from '../../src/scene3d/scales';

function nodes(count: number) {
  const g = graph();
  return Array.from({ length: count }, (_, i) => g.input(`n${i}`, [], [i]));
}

const bounds = { xMin: -5, xMax: 7 };

describe('blockSpan', () => {
  it('runs from the plate to the capped bar top, and to the node top for scalars', () => {
    const g = graph({ dims: { V: 3 } });
    const x = g.input('x', ['V'], [1, -2, 3]);
    const s = g.sum(x, -1, { keepdim: true, name: 'S' });
    const ctx = { visible: g.nodes, aligned: classAlignment(g.nodes), targetOf: targetColumn, focus: 0 };
    const { blocks } = resolveLayout([stack(x, s, { at: [0, 2, 0] })], ctx);
    const bar = blockSpan(blocks[0]);
    expect(bar.bottom).toBeCloseTo(2);
    expect(bar.top).toBeCloseTo(2 + HEIGHT_CAP);
    const disc = blockSpan(blocks[1]);
    expect(disc.top - disc.bottom).toBeCloseTo(0.3);
  });
});

describe('layoutBraces', () => {
  it('covers the y range of the placed nodes with the padding', () => {
    const [a, b, c] = nodes(3);
    const spans = [
      { node: a, bottom: 0, top: 2, z: -1 },
      { node: b, bottom: 5, top: 7, z: -3 },
    ];
    const [brace] = layoutBraces([{ title: 'M', nodes: [a, b, c] }], spans, bounds);
    expect(brace.y0).toBeCloseTo(-MODULE_PAD);
    expect(brace.y1).toBeCloseTo(7 + MODULE_PAD);
    expect(brace.nodes).toEqual([a, b]);
    expect(brace.z).toBe(-2);
    expect(brace.side).toBe('left');
    expect(brace.x).toBeCloseTo(bounds.xMin - MODULE_GAP);
  });

  it('puts right braces outside the right edge', () => {
    const [a] = nodes(1);
    const [brace] = layoutBraces([{ title: 'M', nodes: [a], side: 'right' }], [{ node: a, bottom: 0, top: 1, z: -1 }], bounds);
    expect(brace.x).toBeCloseTo(bounds.xMax + MODULE_GAP);
  });

  it('moves an outer module one step further out than the modules it contains', () => {
    const [a, b, c] = nodes(3);
    const spans = [a, b, c].map((node, i) => ({ node, bottom: i * 3, top: i * 3 + 2, z: -1 }));
    const braces = layoutBraces(
      [
        { title: 'outer', nodes: [a, b, c] },
        { title: 'middle', nodes: [a, b] },
        { title: 'inner', nodes: [a] },
      ],
      spans,
      bounds,
    );
    expect(braces.map((brace) => brace.column)).toEqual([2, 1, 0]);
    expect(braces[0].x).toBeCloseTo(bounds.xMin - MODULE_GAP - 2 * MODULE_STEP);
    expect(braces[1].x).toBeCloseTo(bounds.xMin - MODULE_GAP - MODULE_STEP);
  });

  it('separates overlapping modules on the same side and keeps disjoint ones in one column', () => {
    const [a, b, c, d] = nodes(4);
    const spans = [
      { node: a, bottom: 0, top: 2, z: -1 },
      { node: b, bottom: 1, top: 3, z: -1 },
      { node: c, bottom: 10, top: 12, z: -1 },
      { node: d, bottom: 1, top: 3, z: -1 },
    ];
    const braces = layoutBraces(
      [
        { title: 'A', nodes: [a] },
        { title: 'B', nodes: [b] },
        { title: 'C', nodes: [c] },
        { title: 'D', nodes: [d], side: 'right' },
      ],
      spans,
      bounds,
    );
    expect(braces.map((brace) => brace.column)).toEqual([0, 1, 0, 0]);
  });

  it('treats equal node sets as overlapping modules, not nested ones', () => {
    const [a, b] = nodes(2);
    const spans = [
      { node: a, bottom: 0, top: 2, z: -1 },
      { node: b, bottom: 3, top: 5, z: -1 },
    ];
    const braces = layoutBraces(
      [
        { title: 'A', nodes: [a, b] },
        { title: 'B', nodes: [b, a] },
      ],
      spans,
      bounds,
    );
    expect(braces.map((brace) => brace.column)).toEqual([0, 1]);
  });
});

describe('brace shape', () => {
  it('bends the ends toward the graph and points the tip outward', () => {
    const [a] = nodes(1);
    for (const side of ['left', 'right'] as const) {
      const [brace] = layoutBraces([{ title: 'M', nodes: [a], side }], [{ node: a, bottom: 0, top: 4, z: -1 }], bounds);
      const toward = side === 'left' ? 1 : -1;
      const points = bracePoints(brace);
      const first = points.slice(0, 3);
      const last = points.slice(-3);
      expect(first[0]).toBeCloseTo(brace.x + toward * BRACE_RADIUS);
      expect(first[1]).toBeCloseTo(brace.y1);
      expect(last[0]).toBeCloseTo(brace.x + toward * BRACE_RADIUS);
      expect(last[1]).toBeCloseTo(brace.y0);
      const tip = braceTip(brace);
      expect(tip[0]).toBeCloseTo(brace.x - toward * BRACE_RADIUS);
      expect(tip[1]).toBeCloseTo((brace.y0 + brace.y1) / 2);
      for (let i = 0; i < points.length; i += 3) {
        expect(points[i + 2]).toBe(brace.z);
        expect(toward * (points[i] - tip[0])).toBeGreaterThanOrEqual(-1e-9);
      }
    }
  });

  it('shrinks the curves for short modules', () => {
    const [a] = nodes(1);
    const [brace] = layoutBraces([{ title: 'M', nodes: [a] }], [{ node: a, bottom: 0, top: 0, z: -1 }], bounds);
    const tip = braceTip(brace);
    expect(brace.x - tip[0]).toBeCloseTo((brace.y1 - brace.y0) / 4);
  });
});

describe('checkModules', () => {
  it('ignores unplaced nodes and throws when none is placed', () => {
    const [a, b] = nodes(2);
    expect(() => checkModules([{ title: 'ok', nodes: [a, b] }], [a])).not.toThrow();
    expect(() => checkModules([{ title: 'Attention', nodes: [b] }], [a])).toThrow("modules: 'Attention' has no placed node");
  });
});
