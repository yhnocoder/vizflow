import { describe, expect, it } from 'vitest';
import { graph } from '../../src/autograd';

function square() {
  const g = graph({ check: false });
  const x = g.input('x', [3], [1, -2, 0.5]);
  const w = g.param('w', [3], [0.3, 0.7, -1.2]);
  const before = g.nodes.length;
  const y = g.mul(x, x, { name: 'y' });
  const z = g.mul(y, w, { name: 'z' });
  const inside = g.nodes.slice(before);
  const [leaf, twice, gx, gw] = g.backwardPhase(() => {
    const leaf = g.grad_of(z, { name: 'grad_z' });
    const twice = g.mul(x, 2, { name: 'twice' });
    const gx = g.mul(g.mul(twice, w, { name: 'tw' }), leaf, { name: 'grad_x' });
    const gw = g.mul(y, leaf, { name: 'grad_w' });
    return [leaf, twice, gx, gw];
  });
  const backward = g.nodes.slice(g.nodes.indexOf(leaf));
  g.defineFunction({ inputs: [x, w], outputs: [z], inside, gradOutputs: [{ leaf, output: z }], gradInputs: [{ input: x, grad: gx }, { input: w, grad: gw }], backward });
  const loss = g.sum(z, { name: 'loss' });
  g.loss(loss);
  return { g, x, w, y, z, twice, gx };
}

describe('functions with their own backward', () => {
  it('send the gradients their backward computes to the inputs', () => {
    const { x, w, y, z, gx } = square();
    expect(Array.from(x.grad!.data)).toEqual([0.6, -2.8, -1.2]);
    expect(Array.from(w.grad!.data)).toEqual([1, 4, 0.25]);
    expect(Array.from(gx.value.data)).toEqual([0.6, -2.8, -1.2]);
    expect(y.grad).toBeNull();
    expect(z.grad).not.toBeNull();
    expect(gx.grad).toBeNull();
  });

  it('recompute the backward nodes when a leaf changes', () => {
    const { x, twice } = square();
    x.set([2, 2, 2]);
    expect(Array.from(twice.value.data)).toEqual([4, 4, 4]);
    expect(Array.from(x.grad!.data).map((v) => Number(v.toFixed(6)))).toEqual([1.2, 2.8, -4.8]);
  });
});
