// @vitest-environment jsdom
import { afterEach, describe, expect, it } from 'vitest';
import { graph } from '../../src/core';
import { createSteps, type StepInfo } from '../../src/steps';
import { mountSteps, type Mounted } from '../../src/ui';
import { shown } from './shown';

type Extra = { kind?: string; label?: string; groups: string[]; module?: string };

const mounted: Mounted[] = [];

afterEach(() => {
  mounted.splice(0).forEach((m) => m.dispose());
  document.body.replaceChildren();
});

const text = (el: Element | null) => (el === null ? '' : shown(el));

function timelineSteps() {
  const g = graph({ dims: { V: 3 } });
  const x = g.input('x', ['V'], [1, 2, 3]);
  const a = g.exp(x, { name: 'a' });
  const b = g.log(a, { name: 'b' });
  const c = g.exp(b, { name: 'c' });
  const d = g.sum(c, -1, { name: 'loss' });
  const steps = createSteps(g, {
    overview: { title: '整张图', label: '总览' },
    forward: [
      { show: x, title: 'x', label: 'x' },
      { show: a, title: 'a', label: 'a' },
      { show: b, title: 'b', label: 'b' },
      { show: c, title: 'c', label: 'c' },
      { show: d, title: 'loss', label: '$\\ell$' },
    ],
  });
  const extra: Extra[] = [
    { kind: 'overview', groups: [] },
    { kind: 'inputs', label: '输入', groups: [] },
    { kind: 'op', groups: ['attn'] },
    { kind: 'op', groups: ['attn', 'qk'] },
    { kind: 'op', groups: ['attn', 'qk'] },
    { kind: 'module', groups: ['layers'], module: 'mlp' },
  ];
  steps.list.forEach((info: StepInfo, i) => Object.assign(info, extra[i] ?? { kind: 'op', groups: [] }));
  Object.assign(steps, {
    modules: [{ id: 'mlp' }],
    groups: [
      { id: 'attn', title: 'self_attn' },
      { id: 'qk', title: 'q_norm $\\cdot$ k_norm' },
      { id: 'layers', title: '$\\cdots$ 第 1 到 $N-1$ 层' },
    ],
  });
  return steps;
}

describe('mountSteps timeline', () => {
  it('draws segments, the position and the steps of the current segment from StepInfo', () => {
    const steps = timelineSteps();
    const el = document.createElement('div');
    document.body.append(el);
    mounted.push(mountSteps(el, steps));
    expect(el.querySelector('.vf-stepper')!.classList.contains('vf-timeline')).toBe(true);
    expect(el.querySelector('.vf-grp')).toBeNull();
    const forwardCount = steps.list.filter((s) => s.dir === 'forward').length;
    const barsOf = () => [...el.querySelectorAll<HTMLButtonElement>('.vf-tb')];
    let bars = barsOf();
    expect(bars.map((b) => b.getAttribute('aria-label'))).toEqual(['总览', '输入', 'self_attn', '⋯ 第 1 到 N−1 层']);
    expect(bars.map((b) => b.style.flex)).toEqual(['1 1 0px', '1 1 0px', '3 1 0px', '1 1 0px']);
    expect(bars[0].classList.contains('vf-on')).toBe(true);
    expect(bars[0].style.getPropertyValue('--vf-fr')).toBe('100%');
    expect(text(el.querySelector('.vf-tl-pos'))).toBe(`1/${forwardCount}`);
    expect(el.querySelector('.vf-tl-inner')).toBeNull();
    expect([...el.querySelectorAll('.vf-tl-sub .vf-st')].map(text)).toEqual(['总览']);

    steps.go(3);
    bars = barsOf();
    expect(bars[2].classList.contains('vf-on')).toBe(true);
    expect(bars[2].style.getPropertyValue('--vf-fr')).toBe('67%');
    expect(bars[1].classList.contains('vf-done')).toBe(true);
    expect([...el.querySelectorAll('.vf-tl-sub > *')].map((n) => `${n.className}:${text(n)}`)).toEqual([
      'vf-tl-inner:self_attn',
      'vf-st:a',
      'vf-tl-inner vf-sub:q_norm $\\cdot$ k_norm',
      'vf-st:b',
      'vf-st:c',
    ]);
    expect(el.querySelector('.vf-tl-sub [aria-current="step"]')!.textContent).toBe('b');
    expect(text(el.querySelector('.vf-tl-pos'))).toBe(`4/${forwardCount}`);

    steps.go(5);
    expect(text(el.querySelector('.vf-tl-inner'))).toBe('$\\cdots$ 第 1 到 $N-1$ 层（已折叠）');

    barsOf()[1].click();
    expect(steps.current).toBe(1);
  });

  it('shows a preview of a segment while the pointer is over its bar', () => {
    const steps = timelineSteps();
    const el = document.createElement('div');
    document.body.append(el);
    mounted.push(mountSteps(el, steps));
    steps.go(3);
    const bars = [...el.querySelectorAll<HTMLButtonElement>('.vf-tb')];
    expect(el.querySelector('.vf-peek')).toBeNull();
    bars[2].dispatchEvent(new Event('pointerenter'));
    const peek = el.querySelector('.vf-peek')!;
    expect(text(peek.querySelector('.vf-peek-name'))).toBe('self_attn');
    expect(text(peek.querySelector('.vf-peek-count'))).toBe('第 2 / 3 步');
    expect([...peek.querySelectorAll('li')].map((li) => `${li.className}:${text(li)}`)).toEqual(['vf-done:a', 'vf-on:b', ':c']);
    bars[3].dispatchEvent(new Event('pointerenter'));
    expect(text(el.querySelector('.vf-peek-name'))).toBe('$\\cdots$ 第 1 到 $N-1$ 层（已折叠）');
    expect(text(el.querySelector('.vf-peek-count'))).toBe('1 步');
    el.querySelector('.vf-tl-bar')!.dispatchEvent(new Event('pointerleave'));
    expect(el.querySelector('.vf-peek')).toBeNull();
  });

  it('switches direction with the tabs and counts positions inside a direction', () => {
    const steps = timelineSteps();
    const el = document.createElement('div');
    document.body.append(el);
    mounted.push(mountSteps(el, steps));
    const tabs = [...el.querySelectorAll<HTMLButtonElement>('.vf-dirseg button')];
    expect(tabs.map(text)).toEqual(['前向', '反向']);
    expect(tabs[0].getAttribute('aria-selected')).toBe('true');
    tabs[1].click();
    const first = steps.list.findIndex((s) => s.dir === 'backward');
    expect(steps.current).toBe(first);
    const backwardCount = steps.list.length - first;
    expect(text(el.querySelector('.vf-tl-pos'))).toBe(`1/${backwardCount}`);
    expect(el.querySelectorAll('.vf-dirseg button')[1].getAttribute('aria-selected')).toBe('true');
    tabs[1].click();
    expect(steps.current).toBe(first);
  });

  it('keeps the flat track when the steps have no modules', () => {
    const g = graph({ dims: { V: 3 } });
    const x = g.input('x', ['V'], [1, 2, 3]);
    const steps = createSteps(g, { forward: [{ show: x, title: 'x' }] });
    const el = document.createElement('div');
    document.body.append(el);
    mounted.push(mountSteps(el, steps));
    expect(el.querySelector('.vf-timeline')).toBeNull();
    expect([...el.querySelectorAll('.vf-grp')].map(text)).toEqual(['前向']);
  });
});
