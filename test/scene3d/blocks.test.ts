import { describe, expect, it } from 'vitest';
import { graph } from '../../src/core';
import '../../src/explain';
import { PlateMesh } from '../../src/scene3d/blocks';
import { classAlignment, resolveLayout, stack, targetColumn } from '../../src/scene3d/layout';

describe('PlateMesh axis', () => {
  it('draws one segment per tick plus the axis line and hides the axis when the block is not revealed', () => {
    const g = graph();
    const x = g.input('x', [3], [1, -2, 3]);
    const ctx = { visible: g.nodes, aligned: classAlignment(g.nodes), targetOf: targetColumn, focus: 0 };
    const [block] = resolveLayout([stack(x)], ctx).blocks;
    const plate = new PlateMesh(block);
    const state = { revealed: true, far: false, seeThrough: false };
    plate.update(state, [0], [-1, 0], 1);
    expect(plate.axis.count).toBe(2 + 2 * 2);
    plate.update(state, [0], [-2, 0, 2], 1);
    expect(plate.axis.count).toBe(2 + 2 * 3);
    plate.update(state, [0], [-1, 0, 1], 1);
    expect(plate.axis.count).toBe(8);
    expect([1, 2, 3].map((k) => plate.axis.positions[k * 6 + 1] - block.elements[0].y)).toEqual([-1, 0, 1]);
    plate.update({ revealed: false, far: false, seeThrough: false }, [0], [-1, 0, 1], 1);
    expect(plate.axis.count).toBe(0);
  });
});
