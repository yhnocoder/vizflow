// @vitest-environment jsdom
import { describe, expect, it } from 'vitest';
import { LabelLayer } from '../../src/scene3d/labels';

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
