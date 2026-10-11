import { describe, expect, it } from 'vitest';
import { graph } from '../../src/core';
import '../../src/explain';
import { PlateMesh } from '../../src/scene3d/blocks';
import { classAlignment, resolveLayout, stack, targetColumn } from '../../src/scene3d/layout';
import type { ThemeColors } from '../../src/theme';

const theme: ThemeColors = {
  bg: [0.9, 0.9, 0.9],
  surface: [1, 1, 1],
  ink: [0, 0, 0],
  fg: [0, 0, 0],
  muted: [0.5, 0.5, 0.5],
  line: [0.8, 0.8, 0.8],
  lineStrong: [0.6, 0.6, 0.6],
  pos: [1, 0.4, 0.2],
  neg: [0.2, 0.5, 1],
};

describe('PlateMesh axis', () => {
  it('draws every tick when the tick count grows and frees the replaced geometry', () => {
    const g = graph();
    const x = g.input('x', [3], [1, -2, 3]);
    const ctx = { visible: g.nodes, aligned: classAlignment(g.nodes), targetOf: targetColumn, focus: 0 };
    const [block] = resolveLayout([stack(x)], ctx).blocks;
    const plate = new PlateMesh(block, theme);
    const state = { revealed: true, active: true, backward: false };
    plate.update(state, [0], [-1, 0], 1);
    const first = plate.axis!.geometry;
    expect(first.getAttribute('position').count).toBe(2 + 2 * 2);
    let disposed = false;
    first.addEventListener('dispose', () => {
      disposed = true;
    });
    plate.update(state, [0], [-2, 0, 2], 1);
    const position = () => plate.axis!.geometry.getAttribute('position');
    expect(position().count).toBe(2 + 2 * 3);
    expect(disposed).toBe(true);
    expect([3, 5, 7].map((i) => position().getY(i))).toEqual([-2, 0, 2]);
    plate.update(state, [0], [-1, 0, 1], 1);
    expect(position().count).toBe(8);
    expect([3, 5, 7].map((i) => position().getY(i))).toEqual([-1, 0, 1]);
    expect(plate.axis!.visible).toBe(true);
    plate.update({ revealed: false, active: false, backward: true }, [0], [-1, 0, 1], 1);
    expect(plate.axis!.visible).toBe(false);
    plate.dispose();
  });
});
