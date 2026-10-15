import { describe, expect, it } from 'vitest';
import type { Node } from '../../src/autograd';
import { graph } from '../../src/core';
import '../../src/explain';
import { controlPoints, curveAt } from '../../src/scene3d/edgelines';
import { LOOP_GAP, LOOP_SPAN, edgeRound, loopEnds } from '../../src/scene3d/edges';
import type { Block, Element } from '../../src/scene3d/layout';
import { FRAME_HALF } from '../../src/scene3d/levels';
import { REGION_LIFT, frameRects, takenFrom, tileRegions } from '../../src/scene3d/regions';

function grid(rows: number, cols: number): Block {
  const elements: Element[] = [];
  for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) elements.push({ index: [r, c], x: c, y: 1, z: r, row: r, col: c } as Element);
  return { elements, footprint: [0.6, 0.6] } as unknown as Block;
}

describe('takenFrom', () => {
  const g = graph({ dims: {} });
  const x = g.input('x', [4, 3, 2], Array.from({ length: 4 }, () => Array.from({ length: 3 }, () => [1, 2])));
  const rows = g.narrow(x, 0, 2, 2, { name: 'rows' });
  const head = g.select(rows, 1, 1, { name: 'head' });
  const doubled = g.mul(head, 2, { name: 'doubled' });

  it('follows a chain of narrow and select back to the first drawn tensor', () => {
    const taken = takenFrom(head, (n) => n === x);
    expect(taken?.source).toBe(x);
    expect(taken?.indices).toEqual([[2, 1, 0], [2, 1, 1], [3, 1, 0], [3, 1, 1]]);
  });

  it('stops at an intermediate tensor that is drawn', () => {
    const taken = takenFrom(head, (n) => n === x || n === rows);
    expect(taken?.source).toBe(rows);
    expect(taken?.indices).toEqual([[0, 1, 0], [0, 1, 1], [1, 1, 0], [1, 1, 1]]);
  });

  it('ignores tensors that are computed, or that take every element', () => {
    expect(takenFrom(doubled, (n) => n === x)).toBeUndefined();
    const whole = g.narrow(x, 0, 0, 4, { name: 'whole' });
    expect(takenFrom(whole, (n) => n === x)).toBeUndefined();
  });
});

describe('tileRegions', () => {
  it('lists the elements each tile writes', () => {
    const g = graph({ dims: {} });
    const base = g.input('base', [4, 2], [[0, 0], [0, 0], [0, 0], [0, 0]]);
    const a = g.input('a', [2, 2], [[1, 1], [1, 1]]);
    const b = g.input('b', [2], [2, 2]);
    const out = g.scatter_tiles(base, [a, b], [[[0, 2], null], [3, null]], false, { name: 'out' });
    expect(tileRegions(out)).toEqual([
      { tile: a, indices: [[0, 0], [0, 1], [1, 0], [1, 1]] },
      { tile: b, indices: [[3, 0], [3, 1]] },
    ]);
    expect(tileRegions(a)).toEqual([]);
  });
});

describe('frameRects', () => {
  const block = grid(3, 3);
  const [hx, hz] = FRAME_HALF;

  it('draws one frame around elements that fill a rectangle', () => {
    const picked = block.elements.filter((e) => e.index[0] < 2 && e.index[1] >= 1);
    expect(frameRects(block, picked)).toEqual([[1 - hx, 1 + REGION_LIFT, 0 - hz, 2 + hx, 1 + REGION_LIFT, 0 - hz, 2 + hx, 1 + REGION_LIFT, 1 + hz, 1 - hx, 1 + REGION_LIFT, 1 + hz, 1 - hx, 1 + REGION_LIFT, 0 - hz]]);
  });

  it('frames each element when they do not fill a rectangle', () => {
    const picked = [block.elements[0], block.elements[8]];
    expect(frameRects(block, picked)).toHaveLength(2);
    expect(frameRects(block, [])).toEqual([]);
  });
});

describe('edges across rounds', () => {
  const g = graph({ dims: {} });
  const [r0, r1, r2] = ['s[0]', 's[1]', 's[2]'].map((name) => g.input(name, [2], [1, 2]));
  const slot: readonly Node[] = [r0, r1, r2];
  const to = (node: Node, absent = false) => ({ node, absent, block: { slot } });

  it('draws an edge between the shown members of two blocks', () => {
    expect(edgeRound({ node: r1 }, r1, to(r1), r1)).toBe('same');
  });

  it('draws an edge from an earlier round of the same slot as a previous-round edge', () => {
    expect(edgeRound({ node: r1, slot }, r0, to(r1), r1)).toBe('previous');
  });

  it('drops edges that end on a member not shown, start from a later round, or end on an absent block', () => {
    expect(edgeRound({ node: r1, slot }, r1, to(r1), r2)).toBeUndefined();
    expect(edgeRound({ node: r1, slot }, r2, to(r1), r1)).toBeUndefined();
    expect(edgeRound({ node: r1, slot }, r1, to(r1, true), r1)).toBeUndefined();
  });

  it('starts and ends a self-loop on the right side of the block and bends it outward', () => {
    const block = grid(2, 3);
    const [a, b] = loopEnds(block);
    expect(a).toEqual([2 + 0.3 + LOOP_GAP, a[1], 0.5 - LOOP_SPAN]);
    expect(b).toEqual([2 + 0.3 + LOOP_GAP, a[1], 0.5 + LOOP_SPAN]);
    const middle = curveAt(controlPoints(a, b, false, undefined, true), 0.5);
    expect(middle[0]).toBeGreaterThan(a[0]);
    expect(middle[1]).toBeGreaterThan(a[1]);
  });
});
