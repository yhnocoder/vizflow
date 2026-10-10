// @vitest-environment jsdom
import { afterEach, expect, it, vi } from 'vitest';
import { graph } from '../../src/core';
import '../../src/element';

vi.mock('../../src/scene3d', () => ({
  mountScene: () => {
    throw new Error('scene layout is invalid');
  },
}));

afterEach(() => {
  document.body.replaceChildren();
  vi.unstubAllGlobals();
});

it('unmounts and reports the error when the scene fails to mount', async () => {
  const reported = vi.fn();
  vi.stubGlobal('reportError', reported);
  const g = graph({ dims: { V: 3 } });
  const x = g.input('x', ['V'], [1, 0, -1]);
  const el = document.createElement('vizflow-page');
  document.body.append(el);
  el.graph = g;
  el.steps = { forward: [{ show: x, title: 'x' }] };
  el.scene = { layout: [] };
  expect(el.createdSteps).toBeDefined();
  expect(el.querySelector('.vf-stage')).not.toBeNull();
  await vi.waitFor(() => expect(reported).toHaveBeenCalledTimes(1));
  expect(String(reported.mock.calls[0][0])).toContain('scene layout is invalid');
  expect(el.querySelector('.vf-stage')).toBeNull();
  expect(el.createdSteps).toBeUndefined();
});
