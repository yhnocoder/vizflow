import { describe, expect, it } from 'vitest';
import { graph } from '../../src/core';
import { explain } from '../../src/explain';

describe('explain for embedding, cat and narrow', () => {
  it('connects each embedding output element to the picked row of the weight only', () => {
    const g = graph();
    const ids = g.label('ids', [2, 0], { classes: 3 });
    const E = g.param('E', [3, 2], [[1, 2], [3, 4], [5, 6]], { tex: 'E', elem: 'E' });
    const e = g.embedding(ids, E, { name: 'e' });
    g.sum(e, { name: 'loss' });
    const ex = explain(g);
    const edges = ex.edges(e);
    expect(edges.map((edge) => [edge.arg, 'node' in edge.from ? edge.from.node.name : '', 'node' in edge.from ? edge.from.index : [], 'node' in edge.to ? edge.to.index : []])).toEqual([
      [1, 'E', [2, 0], [0, 0]],
      [1, 'E', [2, 1], [0, 1]],
      [1, 'E', [0, 0], [1, 0]],
      [1, 'E', [0, 1], [1, 1]],
    ]);
    expect(edges.map((edge) => ex.local(edge))).toEqual([1, 1, 1, 1]);
    expect(ex.nodeCard(e, 'forward').tex).toEqual(['\\mathbf{e}=E[\\mathrm{ids}]']);
    expect(ex.edgeCard(edges[0]).local.tex).toBe('\\frac{\\partial e_{1,1}}{\\partial E_{3,1}}=1');
  });

  it('connects each cat output element to the element it came from', () => {
    const g = graph();
    const a = g.input('a', [2, 1], [[1], [2]]);
    const b = g.input('b', [2, 2], [[3, 4], [5, 6]]);
    const c = g.cat([a, b], 1, { name: 'c' });
    const ex = explain(g);
    const sources = ex.edges(c).map((edge) => ('node' in edge.from ? `${edge.from.node.name}[${edge.from.index}]` : ''));
    expect(sources).toEqual(['a[0,0]', 'b[0,0]', 'b[0,1]', 'a[1,0]', 'b[1,0]', 'b[1,1]']);
    expect(ex.edges(c).every((edge) => ex.local(edge) === 1)).toBe(true);
    expect(ex.nodeCard(c, 'forward').tex).toEqual(['\\mathbf{c}=\\operatorname{cat}(\\mathbf{a},\\mathbf{b})']);
  });

  it('connects narrow outputs to the shifted input elements', () => {
    const g = graph();
    const x = g.input('x', [1, 4], [[1, 2, 3, 4]]);
    const n = g.narrow(x, 1, 2, 2, { name: 'n' });
    const ex = explain(g);
    expect(ex.edges(n).map((edge) => ('node' in edge.from ? edge.from.index : []))).toEqual([[0, 2], [0, 3]]);
    expect(ex.nodeCard(n, 'forward').tex).toEqual(['\\mathbf{n}=\\operatorname{narrow}(\\mathbf{x},1,2,2)']);
  });
});

describe('explain for scalar operands', () => {
  it('draws no edge for the number and writes it into the formulas', () => {
    const g = graph();
    const x = g.input('x', [2], [1, -2]);
    const y = g.mul(x, 0.5, { name: 'y' });
    const z = g.add(y, -1.25, { name: 'z' });
    g.sum(z, { name: 'loss' });
    const ex = explain(g);
    expect(ex.edges(y).map((edge) => edge.arg)).toEqual([0, 0]);
    expect(ex.edges(y).map((edge) => ex.local(edge))).toEqual([0.5, 0.5]);
    expect(ex.nodeCard(y, 'forward').tex).toEqual(['\\mathbf{y}=\\mathbf{x}\\cdot 0.500']);
    expect(ex.nodeCard(z, 'forward').tex).toEqual(['\\mathbf{z}=\\mathbf{y}+\\left(-1.25\\right)']);
    expect(ex.edgeCard(ex.edges(y)[1]).local.tex).toBe('\\frac{\\partial y_{2}}{\\partial x_{2}}=0.500');
  });
});

describe('explain for buffers', () => {
  it('draws no edge from a buffer and lists no consumers in its backward card', () => {
    const g = graph();
    const s = g.input('s', [2], [1, 2]);
    const M = g.buffer('M', [2], [0, -Infinity]);
    const y = g.add(s, M, { name: 'y' });
    g.sum(g.exp(y), { name: 'loss' });
    const ex = explain(g);
    expect(ex.edges(y).map((edge) => ('node' in edge.from ? edge.from.node.name : ''))).toEqual(['s', 's']);
    expect(ex.outEdges({ node: M, index: [0] })).toEqual([]);
    const card = ex.nodeCard(M, 'backward');
    expect(card.tex).toEqual(['\\frac{\\partial\\ell}{\\partial \\mathbf{M}}=0']);
    expect(card.rows.at(-1)).toEqual({ label: '当前梯度', value: '无' });
  });
});

describe('element subscripts', () => {
  it('merges the element index into a subscript that ends the elem', () => {
    const g = graph();
    const cases: [string, string][] = [
      ['\\mathrm{ms}_{1}', '\\mathrm{ms}_{1,2,1}'],
      ['r_1', 'r_{1,2,1}'],
      ['W_{q}', 'W_{q,2,1}'],
      ['x_{\\mathrm{a}}', 'x_{\\mathrm{a},2,1}'],
      ['{r_{1}}', '{r_{1}}_{2,1}'],
      ['x^{0}', 'x^{0}_{2,1}'],
      ['a\\_b', 'a\\_b_{2,1}'],
      ['h', 'h_{2,1}'],
    ];
    const ex = explain(g);
    cases.forEach(([elem, expected], i) => {
      const node = g.input(`n${i}`, [3, 2], [1, 2, 3, 4, 5, 6], { elem });
      expect(ex.elemTex({ node, index: [1, 0] })).toBe(expected);
    });
    const column = g.input('col', [3, 1], [1, 2, 3], { elem: 'r_1' });
    expect(ex.elemTex({ node: column, index: [2, 0] })).toBe('r_{1,3}');
    const scalar = g.input('one', [1], [1], { elem: 'r_1' });
    expect(ex.elemTex({ node: scalar, index: [0] })).toBe('r_1');
  });

  it('uses the merged subscript in edge cards', () => {
    const g = graph();
    const ms = g.input('ms', [2, 1], [[1], [4]], { tex: '\\mathrm{ms}_{1}', elem: '\\mathrm{ms}_{1}' });
    const r = g.rsqrt(ms, { name: 'r1', tex: 'r_1', elem: 'r_1' });
    g.sum(r, { name: 'loss' });
    const ex = explain(g);
    const card = ex.edgeCard(ex.edges(r)[1]);
    expect(card.title).toBe('\\mathrm{ms}_{1,2}\\to r_{1,2}');
    expect(card.local.tex).toBe('\\frac{\\partial r_{1,2}}{\\partial \\mathrm{ms}_{1,2}}=-\\frac{1}{2}{r_{1,2}}^{3}');
  });
});
