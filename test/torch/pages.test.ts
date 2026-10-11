import { describe, expect, it } from 'vitest';
import { type Graph, graph } from '../../src/core';
import { type EncodedTensor, decode, expectFinite, expectTensor, loadFixture, nested } from './fixtures';

interface PageCase {
  page: 'lmhead' | 'softmax-single' | 'softmax-batch';
  title: string;
  inputs: Record<string, EncodedTensor>;
  loss: EncodedTensor;
  nodes: Record<string, { value: EncodedTensor; grad: EncodedTensor | null }>;
  finite?: boolean;
}

const fixture = loadFixture<PageCase>('pages.json');

const builders: Record<PageCase['page'], (inputs: PageCase['inputs']) => Graph> = {
  lmhead: (inputs) => {
    const g = graph();
    const h = g.input('h', inputs.h.shape, decode(inputs.h.values));
    const W = g.param('W', inputs.W.shape, decode(inputs.W.values));
    const t = g.label('t', nested(inputs.t));
    const z = g.matmul(h, W.T, { name: 'z' });
    const L = g.logsumexp(z, -1, { keepdim: true, name: 'LSE' });
    const logp = g.sub(z, L, { name: 'log p' });
    g.nll_loss(logp, t, { name: 'loss' });
    return g;
  },
  'softmax-single': (inputs) => {
    const g = graph();
    const x = g.input('x', inputs.x.shape, decode(inputs.x.values));
    const t = g.label('t', nested(inputs.t));
    const u = g.exp(x, { name: 'u' });
    const S = g.sum(u, -1, { keepdim: true, name: 'S' });
    const p = g.div(u, S, { name: 'p' });
    const y = g.gather(p, -1, t, { name: 'y' });
    g.neg(g.log(y, { name: 'log y' }), { name: 'l' });
    return g;
  },
  'softmax-batch': (inputs) => {
    const g = graph();
    const X = g.input('X', inputs.X.shape, decode(inputs.X.values));
    const t = g.label('t', nested(inputs.t));
    const U = g.exp(X, { name: 'U' });
    const S = g.sum(U, -1, { keepdim: true, name: 'S' });
    const P = g.div(U, S, { name: 'P' });
    const Y = g.gather(P, -1, t, { name: 'Y' });
    const lb = g.neg(g.log(Y, { name: 'log Y' }), { name: 'l_b' });
    g.mean(lb, { name: 'l' });
    return g;
  },
};

describe(`PyTorch ${fixture.torch} reference for the example pages`, () => {
  for (const c of fixture.cases) {
    it(`${c.page}: ${c.title}`, () => {
      const g = builders[c.page](c.inputs);
      expect(g.nodes.map((n) => n.name).sort()).toEqual(Object.keys(c.nodes).sort());
      expectTensor(g.loss().value, c.loss, 'loss');
      for (const [name, expected] of Object.entries(c.nodes)) {
        const node = g.node(name)!;
        expectTensor(node.value, expected.value, `value of ${name}`);
        if (expected.grad === null) expect(node.grad, `grad of ${name}`).toBeNull();
        else expectTensor(node.grad, expected.grad, `grad of ${name}`);
        if (c.finite) {
          expectFinite(node.value, `value of ${name}`);
          expectFinite(node.grad, `grad of ${name}`);
        }
      }
    });
  }
});
