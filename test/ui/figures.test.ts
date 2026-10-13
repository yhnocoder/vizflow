// @vitest-environment jsdom
import { afterEach, describe, expect, it } from 'vitest';
import { type ValueExpr, evaluateValue, graph } from '../../src/core';
import { renderText } from '../../src/formula';
import { createSteps } from '../../src/steps';
import { mountSteps, renderFigures, type Mounted } from '../../src/ui';
import { checkFigure, decodeExpr, figureTag } from '../../src/ui/figure-spec';
import { shown } from './shown';

const angles: ValueExpr = {
  k: 'bin',
  op: '*',
  a: { k: 'index', obj: { k: 'call', fn: 'arange', args: [{ k: 'num', v: 5 }], kwargs: [] }, items: [{ k: 'slice' }, { k: 'new' }] },
  b: { k: 'node', name: 'x' },
};

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

describe('value expressions', () => {
  it('computes with arange, a new axis, broadcasting and ops', () => {
    const { g } = smallGraph();
    const read = (name: string) => g.node(name)!.value;
    const value = evaluateValue({ k: 'call', fn: 'cos', args: [angles], kwargs: [] }, read);
    expect(value.shape).toEqual([5, 2]);
    expect(value.get([3, 1])).toBeCloseTo(Math.cos(1.5), 12);
    const sliced = evaluateValue({ k: 'index', obj: angles, items: [{ k: 'at', i: -1 }, { k: 'slice', start: 1 }] }, read);
    expect([...sliced.shape, ...sliced.data]).toEqual([1, 2]);
    expect(() => evaluateValue({ k: 'call', fn: 'nope', args: [], kwargs: [] }, read)).toThrow("unknown function 'nope'");
  });

  it('round-trips through the figure tag', () => {
    const tag = figureTag('lines', angles, ['shade=3']);
    expect(tag).toMatch(/^<vf-figure name="lines" expr="[A-Za-z0-9+/=]+" args="shade=3"><\/vf-figure>$/);
    expect(decodeExpr(/expr="([^"]+)"/.exec(tag)![1])).toEqual(angles);
  });
});

describe('lines figure', () => {
  it('draws one curve per column with a legend and a shaded band', () => {
    const { g } = smallGraph();
    const el = document.createElement('div');
    renderText(`第一段\n\n${figureTag('lines', { k: 'call', fn: 'cos', args: [angles], kwargs: [] }, ['shade=3'])}\n\n第二段`, el);
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
    el.innerHTML = figureTag('lines', { k: 'node', name: 'missing' }, []);
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
