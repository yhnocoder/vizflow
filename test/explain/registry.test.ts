import { describe, expect, it } from 'vitest';
import { Tensor, defineOp, graph } from '../../src/core';
import { explain, explainOf, registerExplain } from '../../src/explain';

describe('explain registry', () => {
  it('rejects a second registration for the same op', () => {
    expect(() => registerExplain('exp', explainOf('exp')!)).toThrow("explain for op 'exp' is already registered");
  });

  it('falls back to the explain field of a defineOp definition', () => {
    defineOp({
      name: 'square',
      args: ['input'],
      tensors: ['input'],
      shape: ([a]) => [...a],
      forward: ([a]) => new Tensor(a.shape, a.data.map((v) => v * v)),
      backward: (gradOut, [a]) => [new Tensor(a.shape, gradOut.data.map((g, i) => 2 * a.data[i] * g))],
      explain: {
        pattern: 'elementwise',
        local: (ctx, _out, _arg, input) => 2 * ctx.inputs[0].get(input),
        tex: { forward: (s) => `${s.in[0]}^{2}`, local: (s) => `2${s.in}` },
      },
    });
    expect(explainOf('square')).toBeDefined();
    const g = graph();
    const x = g.input('x', [3], [1, 2, 3]);
    const y = (g as unknown as { square(n: typeof x): typeof x }).square(x);
    g.loss(g.sum(y));
    const ex = explain(g);
    const e = ex.outEdges({ node: x, index: [1] });
    expect(e).toHaveLength(1);
    expect(ex.local(e[0])).toBe(4);
    expect(ex.edgeCard(e[0]).local.tex).toBe('\\frac{\\partial \\mathrm{square1}_{2}}{\\partial x_{2}}=2x_{2}');
    expect(ex.nodeCard(y, 'forward').tex).toEqual(['\\mathrm{square1}=\\mathbf{x}^{2}']);
  });

  it('throws for an op without explain definition', () => {
    defineOp({
      name: 'opaque',
      args: ['input'],
      tensors: ['input'],
      shape: ([a]) => [...a],
      forward: ([a]) => a.clone(),
      backward: (gradOut) => [gradOut.clone()],
    });
    const g = graph();
    const x = g.input('x', [2], [1, 2]);
    g.loss(g.sum((g as unknown as { opaque(n: typeof x): typeof x }).opaque(x)));
    expect(() => explain(g).edges()).toThrow("explain: op 'opaque' of node 'opaque1' has no explain definition");
  });
});
