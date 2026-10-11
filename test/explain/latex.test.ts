import katex from 'katex';
import { describe, expect, it } from 'vitest';
import { builtinOps, graph } from '../../src/core';
import { builtinExplains, explain } from '../../src/explain';

const renders = (tex: string) => {
  expect(() => katex.renderToString(tex, { throwOnError: true }), tex).not.toThrow();
};

describe('LaTeX templates render with KaTeX', () => {
  it('forward and local templates of every builtin op', () => {
    const g = graph();
    const x = g.input('x', [2, 3], [[1, 2, 3], [4, 5, 6]]);
    const node = g.exp(x);
    const ctx = { node, inputs: [x.value, x.value], out: node.value, attrs: { dim: -1, keepdim: true, dim0: 0, dim1: 1, reduction: 'mean', ignore_index: -100, exponent: 0.5, start: 1, length: 2, size: 2 } };
    for (const def of builtinOps) {
      const ex = builtinExplains[def.name];
      expect(ex, def.name).toBeDefined();
      renders(`\\mathbf{y}=${ex.tex.forward({ out: '\\mathbf{y}', in: ['\\mathbf{a}', '\\mathbf{b}'], attrs: ctx.attrs })}`);
      for (const arg of def.tensors.map((_, i) => i)) {
        for (const [outIndex, inIndex] of [
          [[0, 1], [0, 1]],
          [[0, 1], [1, 0]],
        ]) {
          renders(
            ex.tex.local({
              out: 'y_{1,2}',
              in: 'a_{1,2}',
              ins: ['a_{1,2}', 'b_{1,2}'],
              arg,
              attrs: ctx.attrs,
              outIndex,
              inIndex,
              ctx,
              elem: (_arg, index) => `y_{${index.map((i) => i + 1).join(',')}}`,
            }),
          );
        }
      }
    }
  });

  it('generated cards on the lm_head and softmax graphs', () => {
    const g = graph({ dims: { d: 4, V: 5 } });
    const h = g.input('h', [1, 'd'], [0.8, -1.1, 0.5, 1.4]);
    const W = g.param('W', ['V', 'd'], [[0.9, -0.6, 0.3, 0.6], [0.2, -0.8, 0.6, 0.2], [-0.4, 0.2, 0.9, 0.3], [0.5, 0.7, -0.6, -0.2], [-0.7, 0.4, -0.3, -0.5]]);
    const t = g.label('t', [1]);
    const z = g.matmul(h, W.T, { name: 'z' });
    const L = g.logsumexp(z, -1, { keepdim: true, name: 'LSE' });
    const logp = g.sub(z, L, { name: 'log p', tex: '\\log\\mathbf{p}', elem: '\\log p' });
    g.nll_loss(logp, t, { name: 'loss', tex: '\\ell', elem: '\\ell' });
    const p = g.softmax(z, -1, { name: 'p' });
    g.log_softmax(z, -1, { name: 'q' });
    g.mean(g.div(g.mul(p, z), g.max(z, -1, { keepdim: true })));
    const ex = explain(g);
    for (const node of g.nodes) {
      for (const dir of ['forward', 'backward'] as const) ex.nodeCard(node, dir).tex.forEach(renders);
    }
    for (const e of ex.edges()) {
      const card = ex.edgeCard(e);
      renders(card.title);
      renders(card.local.tex);
      renders(card.upstream.tex);
      renders(card.product.tex);
    }
  });

  it('cards of a decoder-style graph with scalar operands, buffers and subscripted elem', () => {
    const g = graph({ dims: { T: 3, d: 4, V: 5, dh: 2 }, real: { d: 4096, dh: 128, name: 'Qwen3-8B' } });
    const ids = g.label('ids', [2, 4, 3], { classes: 5 });
    const E = g.param('E', ['V', 'd'], Array.from({ length: 20 }, (_, i) => ((i * 7) % 11) / 5 - 1), { tex: 'E', elem: 'E' });
    const M = g.buffer('M', ['T', 'T'], [[0, -Infinity, -Infinity], [0, 0, -Infinity], [0, 0, 0]]);
    const e = g.embedding(ids, E, { name: 'e' });
    const sq = g.pow(e, 2, { name: 'sq', tex: '\\mathbf{e}^{2}', elem: '{e^{2}}' });
    const ms = g.mean(sq, -1, { keepdim: true, name: 'ms1', tex: '\\mathrm{ms}_{1}', elem: '\\mathrm{ms}_{1}' });
    const r = g.rsqrt(g.add(ms, 1e-6, { name: 'ms+eps' }), { name: 'r1', tex: 'r_1', elem: 'r_1' });
    const xh = g.mul(e, r, { name: 'xh' });
    const [a, b] = g.split(xh, 'dh', 1);
    const S = g.mul(g.matmul(a, b.T, { name: 'S' }), 0.5, { name: 'S2' });
    const P = g.softmax(g.add(S, M, { name: 'Sm' }), -1, { name: 'P' });
    const h = g.cat([g.sigmoid(a, { name: 'sa' }), g.silu(g.sqrt(g.sub(g.div(b, -2, { name: 'bd' }), -3, { name: 'bs' }), { name: 'sb' }), { name: 'tb' })], 1, { name: 'h' });
    g.sum(g.mul(g.matmul(P, h, { name: 'o' }), h), { name: 'loss' });
    const ex = explain(g);
    for (const node of g.nodes) {
      for (const dir of ['forward', 'backward'] as const) ex.nodeCard(node, dir).tex.forEach(renders);
    }
    for (const edge of ex.edges()) {
      const card = ex.edgeCard(edge);
      renders(card.title);
      renders(card.local.tex);
      renders(card.product.tex);
    }
  });
});
