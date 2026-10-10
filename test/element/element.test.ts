// @vitest-environment jsdom
import { afterEach, describe, expect, it, vi } from 'vitest';
import { graph } from '../../src/core';
import { VizflowPage } from '../../src/element';
import { board } from '../../src/scene3d';
import type { StepsSpec } from '../../src/steps';

function softmaxPage() {
  const g = graph({ dims: { V: 3 } });
  const x = g.input('x', ['V'], [1, 0, -1]);
  const t = g.label('t', 0, { tokens: ['a', 'b', 'c'] });
  const p = g.softmax(x, -1, { name: 'p' });
  const logp = g.log(p, { name: 'log p' });
  const loss = g.nll_loss(logp, t, { name: 'loss' });
  const steps: StepsSpec = {
    overview: { title: '总览', text: '{{loss:3}}' },
    forward: [
      { show: x, title: 'logits', text: 'x = {{x[t]:2}}' },
      { show: [p, logp, loss], title: 'loss', text: '{{loss:3}}' },
    ],
  };
  return { g, x, t, p, loss, steps };
}

function attach(): VizflowPage {
  const el = document.createElement('vizflow-page');
  document.body.append(el);
  return el;
}

const settle = () => Promise.resolve();

afterEach(() => {
  document.body.replaceChildren();
  vi.restoreAllMocks();
});

describe('<vizflow-page>', () => {
  it('is defined once as VizflowPage', () => {
    expect(customElements.get('vizflow-page')).toBe(VizflowPage);
    expect(document.createElement('vizflow-page')).toBeInstanceOf(VizflowPage);
  });

  it('waits for graph and steps before mounting', async () => {
    const { g, steps } = softmaxPage();
    const el = attach();
    el.graph = g;
    await settle();
    expect(el.children.length).toBe(0);
    expect(el.createdSteps).toBeUndefined();
    el.steps = steps;
    await settle();
    expect(el.querySelector('.vf-stack > .vf-main .vf-stepper')).not.toBeNull();
    expect(el.querySelector('.vf-side')).toBeNull();
    expect(el.createdSteps?.list[0].title).toBe('总览');
  });

  it('lays out steps and panels like the example pages and mounts once for several property sets', async () => {
    const { g, x, t, steps } = softmaxPage();
    const el = attach();
    const subscribe = vi.spyOn(g, 'subscribe');
    el.graph = g;
    el.steps = steps;
    el.panels = ['loss', { tokens: t, columns: [x] }];
    await settle();
    expect(el.querySelector('.vf-split > .vf-main > .vf-steps .vf-stepper')).not.toBeNull();
    expect(el.querySelector('.vf-split > aside.vf-side .vf-panels')).not.toBeNull();
    expect(el.querySelector('.vf-stage')).toBeNull();
    const mountedSubscriptions = subscribe.mock.calls.length;
    el.graph = g;
    el.graph = g;
    await settle();
    expect(subscribe.mock.calls.length).toBe(mountedSubscriptions * 2);
  });

  it('reading createdSteps mounts pending properties at once', () => {
    const { g, steps } = softmaxPage();
    const el = attach();
    el.graph = g;
    el.steps = steps;
    const created = el.createdSteps;
    expect(created).toBeDefined();
    created!.go(2);
    expect(el.querySelector('.vf-st[aria-current="step"]')?.textContent).toBe('loss');
  });

  it('remounts with a new Steps object when a property changes after mounting', async () => {
    const { g, steps } = softmaxPage();
    const el = attach();
    el.graph = g;
    el.steps = steps;
    const first = el.createdSteps!;
    const disposeFirst = vi.spyOn(first, 'dispose');
    el.steps = { ...steps, overview: { title: '整张图', text: '' } };
    await settle();
    expect(disposeFirst).toHaveBeenCalledTimes(1);
    expect(el.createdSteps).not.toBe(first);
    expect(el.querySelector('.vf-st')?.textContent).toBe('整张图');
    expect(el.querySelectorAll('.vf-stepper').length).toBe(1);
  });

  it('disposes everything when disconnected and mounts again when connected', async () => {
    const { g, x, steps } = softmaxPage();
    const el = attach();
    const unsubscribed = vi.fn();
    const subscribe = g.subscribe.bind(g);
    vi.spyOn(g, 'subscribe').mockImplementation((fn) => {
      const off = subscribe(fn);
      return () => {
        unsubscribed();
        off();
      };
    });
    el.graph = g;
    el.steps = steps;
    el.panels = ['loss'];
    const created = el.createdSteps!;
    const disposeSteps = vi.spyOn(created, 'dispose');
    const subscriptions = vi.mocked(g.subscribe).mock.calls.length;
    el.remove();
    expect(disposeSteps).toHaveBeenCalledTimes(1);
    expect(unsubscribed).toHaveBeenCalledTimes(subscriptions);
    expect(el.children.length).toBe(0);
    expect(el.createdSteps).toBeUndefined();
    x.set([0, 0, 0]);
    document.body.append(el);
    await settle();
    expect(el.createdSteps).toBeDefined();
    expect(el.querySelector('.vf-stepper')).not.toBeNull();
  });

  it('mounts the scene in the stage above the steps', async () => {
    vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(null);
    const { g, x, p, steps } = softmaxPage();
    const el = attach();
    el.graph = g;
    el.steps = steps;
    el.scene = { layout: [board(x, p, p)] };
    el.edit = true;
    await settle();
    expect(el.querySelector('.vf-main > .vf-stage + .vf-steps')).not.toBeNull();
    expect(el.querySelector('.vf-stage > .vf-scene')).not.toBeNull();
  });
});
