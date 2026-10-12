import { describe, expect, it } from 'vitest';
import { moduleJacobian } from '../../src/autograd';
import { moduleJacobians } from '../../src/autograd/jacobian';
import { graph } from '../../src/core';

describe('moduleJacobians', () => {
  it('gives every input the same Jacobian as moduleJacobian in one pass per output element', () => {
    const g = graph({ dims: {} });
    const a = g.param('a', [2], [1, 2]);
    const b = g.param('b', [2], [3, 5]);
    const c = g.param('c', [2], [7, 11]);
    const p = g.mul(a, b);
    const q = g.add(p, a);
    const r = g.mul(c, c);
    const module = { nodes: [p, q, r] };
    const all = moduleJacobians(g, module, [a, b, c], q);
    expect(all.get(a)).toEqual([[4, 0], [0, 6]]);
    expect(all.get(b)).toEqual([[1, 0], [0, 2]]);
    expect(all.get(c)).toEqual([[0, 0], [0, 0]]);
    for (const input of [a, b, c]) expect(all.get(input)).toEqual(moduleJacobian(g, module, input, q));
  });
});
