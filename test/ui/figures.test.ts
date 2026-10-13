// @vitest-environment jsdom
import { afterEach, describe, expect, it } from 'vitest';
import { type Program, ProgramBuilder, graph, runProgram } from '../../src/core';
import { renderText } from '../../src/formula';
import { createSteps } from '../../src/steps';
import { mountSteps, renderFigures, type Mounted } from '../../src/ui';
import { checkFigure, decodeProgram, figureTag } from '../../src/ui/figure-spec';
import { shown } from './shown';

function anglesProgram(then?: string): Program {
  const b = new ProgramBuilder();
  const rows = b.op('unsqueeze', [b.op('arange', [], { start: 5, end: null, step: 1, dtype: null })], { dim: 1 });
  const angles = b.op('mul', [rows, b.read('x', false)], {});
  return b.program(then === undefined ? angles : b.op(then, [angles], {}));
}

const angles = anglesProgram();
const cosines = anglesProgram('cos');

function smallGraph() {
  const g = graph({ dims: { K: 2 } });
  const w = g.buffer('w', ['K'], [1, 1]);
  const x = g.input('x', ['K'], [1, 0.5]);
  const y = g.mul(x, w, { name: 'y' });
  return { g, x, y };
}

const mounted: Mounted[] = [];

afterEach(() => {
  mounted.splice(0).forEach((m) => m.dispose());
  document.body.replaceChildren();
});

describe('figure programs', () => {
  it('runs a program of registry ops on values read from the graph', () => {
    const { g } = smallGraph();
    const value = runProgram(cosines, (name) => g.node(name)!.value);
    expect(value.shape).toEqual([5, 2]);
    expect(value.get([3, 1])).toBeCloseTo(Math.cos(1.5), 12);
  });

  it('round-trips through the figure tag, non-finite numbers included', () => {
    const b = new ProgramBuilder();
    const program = b.program(b.op('clamp', [b.read('x', false)], { min: -Infinity, max: Number.NaN }));
    const tag = figureTag('lines', program, ['shade=3']);
    expect(tag).toMatch(/^<vf-figure name="lines" program="[A-Za-z0-9+/=]+" args="shade=3"><\/vf-figure>$/);
    expect(decodeProgram(/program="([^"]+)"/.exec(tag)![1])).toEqual(program);
  });
});

describe('lines figure', () => {
  it('draws one curve per column with a legend and a shaded band', () => {
    const { g } = smallGraph();
    const el = document.createElement('div');
    renderText(`第一段\n\n${figureTag('lines', cosines, ['shade=3'])}\n\n第二段`, el);
    expect([...el.children].map((c) => c.tagName.toLowerCase())).toEqual(['p', 'vf-figure', 'p']);
    renderFigures(el, g);
    const svg = el.querySelector('vf-figure svg.vf-lines')!;
    expect(svg.getAttribute('aria-label')).toBe('2 条曲线，横轴是行号 0 到 4');
    expect(svg.querySelectorAll('path')).toHaveLength(2);
    expect(svg.querySelectorAll('path')[0].hasAttribute('stroke-dasharray')).toBe(false);
    expect(svg.querySelectorAll('path')[1].getAttribute('stroke-dasharray')).toBe('4 3');
    expect(svg.querySelectorAll('rect')).toHaveLength(1);
    expect([...svg.querySelectorAll('circle')].filter((c) => c.getAttribute('r') === '2.8')).toHaveLength(6);
    expect(svg.querySelectorAll('circle')).toHaveLength(10);
    expect([...svg.querySelectorAll('foreignObject')].map(shown)).toEqual(['$1.00$', '$-0.99$', '$0$']);
    expect([...el.querySelectorAll('.vf-lines-key')].map((k) => k.textContent)).toEqual(['列 0', '列 1']);
  });

  it('reads values again when the graph changes', () => {
    const { g, x } = smallGraph();
    const el = document.createElement('div');
    el.innerHTML = figureTag('lines', angles, []);
    renderFigures(el, g);
    expect(el.querySelectorAll('rect')).toHaveLength(0);
    expect(shown(el.querySelector('foreignObject'))).toBe('$4.00$');
    x.set([2, 0.5]);
    renderFigures(el, g);
    expect(shown(el.querySelector('foreignObject'))).toBe('$8.00$');
  });

  it('checks the arguments and the shape', () => {
    expect(checkFigure('lines', [3, 2], ['shade=4'])).toBe('lines shade must be an integer from 0 to 3, got 4');
    expect(checkFigure('lines', [3, 2], ['width=4'])).toBe("lines takes shade=value, got 'width=4'");
    expect(checkFigure('lines', [3], [])).toBe('lines needs a (N, K) tensor with at least 2 rows, got shape (3)');
    expect(checkFigure('lines', [1, 2], [])).toBe('lines needs a (N, K) tensor with at least 2 rows, got shape (1, 2)');
    expect(checkFigure('lines', [3, 2], ['shade=3'])).toBeUndefined();
    expect(checkFigure('chart', [3, 4], [])).toBe("unknown figure 'chart'");
    const { g } = smallGraph();
    const el = document.createElement('div');
    const missing = new ProgramBuilder();
    el.innerHTML = figureTag('lines', missing.program(missing.read('missing', false)), []);
    expect(() => renderFigures(el, g)).toThrow("figure lines: no node 'missing'");
  });

  it('renders figures in the step caption', () => {
    const { g, y } = smallGraph();
    const steps = createSteps(g, { forward: [{ show: y, title: 'y', text: `说明\n\n${figureTag('lines', angles, ['shade=2'])}` }] });
    steps.go(1);
    const el = document.createElement('div');
    document.body.append(el);
    mounted.push(mountSteps(el, steps));
    expect(el.querySelector('.vf-caption-body vf-figure svg.vf-lines')).not.toBeNull();
  });
});
