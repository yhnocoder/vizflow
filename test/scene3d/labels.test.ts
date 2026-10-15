// @vitest-environment jsdom
import { describe, expect, it } from 'vitest';
import { AVOID_STEP } from '../../src/scene3d/avoid';
import { CLEAR_MAX_STEPS, LabelLayer, clearShift } from '../../src/scene3d/labels';

describe('clearShift', () => {
  const bounds = { x: 0, y: 0, w: 400, h: 400 };

  it('keeps a label that touches nothing in place', () => {
    expect(clearShift([{ x: 100, y: 100, w: 20, h: 10 }], [{ x: 200, y: 200, w: 20, h: 20 }], bounds)).toEqual([0, 0]);
  });

  it('moves a label up first, by whole steps, until it clears a block', () => {
    const shift = clearShift([{ x: 100, y: 100, w: 20, h: 10 }], [{ x: 90, y: 105, w: 40, h: 40 }], bounds);
    expect(shift).toEqual([0, -2 * AVOID_STEP]);
  });

  it('moves a group of labels together', () => {
    const boxes = [{ x: 100, y: 100, w: 20, h: 10 }, { x: 100, y: 112, w: 20, h: 10 }];
    const shift = clearShift(boxes, [{ x: 95, y: 118, w: 30, h: 30 }], bounds);
    expect(shift).toEqual([0, -AVOID_STEP]);
  });

  it('does not leave the canvas to clear a block', () => {
    const shift = clearShift([{ x: 100, y: 0, w: 20, h: 10 }], [{ x: 95, y: 5, w: 30, h: 10 }], bounds);
    expect(shift[1]).toBeGreaterThanOrEqual(0);
    expect(shift).not.toEqual([0, 0]);
  });

  it('takes the shift with the least overlap when no shift within reach clears every block', () => {
    const reach = (CLEAR_MAX_STEPS + 2) * AVOID_STEP;
    const solid = [{ x: 100 - reach, y: 100 - reach, w: 2 * reach + 20, h: 2 * reach + 10 }];
    expect(clearShift([{ x: 100, y: 100, w: 20, h: 10 }], solid, bounds)).toEqual([0, 0]);
  });
});

describe('LabelLayer', () => {
  it('puts a label element into the page only when the label is first measured', () => {
    const parent = document.createElement('div');
    const layer = new LabelLayer(parent, () => {});
    const shown = layer.add('val');
    const never = layer.add('val');
    shown.setText('1.5');
    never.setText('2.5');
    expect(layer.root.children.length).toBe(0);
    shown.measure();
    expect([...layer.root.children]).toEqual([shown.el]);
    expect(never.el.parentElement).toBeNull();
  });
});
