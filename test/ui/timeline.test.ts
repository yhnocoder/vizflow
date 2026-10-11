// @vitest-environment jsdom
import { afterEach, describe, expect, it } from 'vitest';
import { graph } from '../../src/core';
import { createSteps, type StepInfo } from '../../src/steps';
import { mountSteps, type Mounted } from '../../src/ui';

type Extra = { kind?: string; segment?: string; sub?: string; module?: string };

const mounted: Mounted[] = [];

afterEach(() => {
  mounted.splice(0).forEach((m) => m.dispose());
  document.body.replaceChildren();
});

const text = (el: Element | null) => el?.textContent ?? '';

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
      { show: d, title: 'loss', label: 'ℓ' },
    ],
  });
  const extra: Extra[] = [
    { kind: 'overview', segment: '总览' },
    { kind: 'inputs', segment: '输入' },
    { kind: 'op', segment: 'self_attn' },
    { kind: 'op', segment: 'self_attn', sub: 'q_norm · k_norm' },
    { kind: 'op', segment: 'self_attn', sub: 'q_norm · k_norm' },
    { kind: 'module', segment: 'mlp', module: 'mlp' },
  ];
  steps.list.forEach((info: StepInfo, i) => Object.assign(info, extra[i] ?? { kind: 'op', segment: 'nll_loss' }));
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
    expect(bars.map((b) => b.title)).toEqual(['总览', '输入', 'self_attn', 'mlp']);
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
      'vf-tl-inner vf-sub:q_norm · k_norm',
      'vf-st:b',
      'vf-st:c',
    ]);
    expect(el.querySelector('.vf-tl-sub [aria-current="step"]')!.textContent).toBe('b');
    expect(text(el.querySelector('.vf-tl-pos'))).toBe(`4/${forwardCount}`);

    steps.go(5);
    expect(text(el.querySelector('.vf-tl-inner'))).toBe('mlp（已折叠）');

    barsOf()[1].click();
    expect(steps.current).toBe(1);
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

  it('keeps the flat track when no step has a segment', () => {
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
