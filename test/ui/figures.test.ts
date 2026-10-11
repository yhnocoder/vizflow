// @vitest-environment jsdom
import { afterEach, describe, expect, it } from 'vitest';
import { graph } from '../../src/core';
import { renderText } from '../../src/formula';
import { createSteps } from '../../src/steps';
import { mountSteps, renderFigures, type Mounted } from '../../src/ui';
import { checkFigure, figureTag } from '../../src/ui/figure-spec';
import { shown } from './shown';

const THETA = 10;
const DH = 4;

function table(fn: (x: number) => number, rows: number) {
  return Array.from({ length: rows }, (_, t) => Array.from({ length: DH }, (_, c) => fn(t * Math.pow(THETA, (-2 * (c % (DH / 2))) / DH))));
}

function ropeGraph() {
  const g = graph({ dims: { T: 3, d: DH } });
  const cos = g.buffer('rope.cos', ['T', 'd'], table(Math.cos, 3));
  const sin = g.buffer('rope.sin', ['T', 'd'], table(Math.sin, 3));
  const x = g.input('x', ['T', 'd'], table(() => 1, 3));
  const y = g.mul(x, cos, { name: 'y' });
  return { g, cos, sin, x, y };
}

const mounted: Mounted[] = [];

afterEach(() => {
  mounted.splice(0).forEach((m) => m.dispose());
  document.body.replaceChildren();
});

describe('figures in step text', () => {
  it('draws the rope wave with the frequencies read from the table', () => {
    const { g } = ropeGraph();
    const el = document.createElement('div');
    renderText(`第一段\n\n${figureTag('rope_wave', 'rope.sin', ['sin'])}\n\n第二段`, el);
    expect([...el.children].map((c) => c.tagName.toLowerCase())).toEqual(['p', 'vf-figure', 'p']);
    renderFigures(el, g);
    const svg = el.querySelector('vf-figure svg.vf-wave')!;
    expect(svg.getAttribute('aria-label')).toBe('sin 表延长到 24 个位置');
    expect(svg.querySelectorAll('path')).toHaveLength(2);
    expect(svg.querySelectorAll('path')[1].getAttribute('stroke-dasharray')).toBe('4 3');
    expect(svg.querySelectorAll('circle')).toHaveLength(50);
    expect([...svg.querySelectorAll('circle')].filter((c) => c.getAttribute('r') === '2.8')).toHaveLength(6);
    expect([...svg.querySelectorAll('foreignObject')].map(shown)).toEqual(['$1$', '$-1$', '$t=0$', '$t=6$', '$t=12$', '$t=18$']);
    expect(shown(el.querySelector('.vf-wave-note'))).toBe(
      '横轴是 position_id $t$（从 0 开始，第 1 个 token 是 $t = 0$），阴影是这里用到的 $t = 0, 1, 2$。实线 $\\omega = 1$（第 1、3 列），虚线 $\\omega = 0.316$（第 2、4 列）。',
    );
  });

  it('takes the number of positions as the second argument', () => {
    const { g } = ropeGraph();
    const el = document.createElement('div');
    el.innerHTML = figureTag('rope_wave', 'rope.cos', ['cos', '12']);
    renderFigures(el, g);
    expect(el.querySelectorAll('circle')).toHaveLength(26);
    expect([...el.querySelectorAll('foreignObject')].map(shown).slice(2)).toEqual(['$t=0$', '$t=6$']);
  });

  it('checks the arguments and the table shape', () => {
    expect(checkFigure('rope_wave', [3, 4], ['tan'])).toBe("rope_wave needs cos or sin as its first argument, got 'tan'");
    expect(checkFigure('rope_wave', [3, 4], ['cos', '0'])).toBe("rope_wave positions must be a positive integer, got '0'");
    expect(checkFigure('rope_wave', [1, 4], ['cos'])).toBe('rope_wave needs a (T, d_h) table with at least 2 rows and an even d_h; dimensions of size 1 may sit between them');
    expect(checkFigure('rope_wave', [3, 1, 4], ['cos'])).toBeUndefined();
    expect(checkFigure('rope_wave', [3, 2, 4], ['cos'])).toMatch(/^rope_wave needs a \(T, d_h\) table/);
    expect(checkFigure('chart', [3, 4], [])).toBe("unknown figure 'chart'");
    const { g } = ropeGraph();
    const el = document.createElement('div');
    el.innerHTML = figureTag('rope_wave', 'missing', ['cos']);
    expect(() => renderFigures(el, g)).toThrow("figure rope_wave: no node 'missing'");
  });

  it('renders figures in the step caption', async () => {
    const { g, y } = ropeGraph();
    const steps = createSteps(g, { forward: [{ show: y, title: 'cos', text: `说明\n\n${figureTag('rope_wave', 'rope.cos', ['cos'])}` }] });
    steps.go(1);
    const el = document.createElement('div');
    document.body.append(el);
    mounted.push(mountSteps(el, steps));
    expect(el.querySelector('.vf-caption-body vf-figure svg.vf-wave')).not.toBeNull();
  });
});
