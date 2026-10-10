import { describe, expect, it } from 'vitest';
import { graph } from '../../src/core';
import { type FlowEdge, explain } from '../../src/explain';

const H = [0.8, -1.1, 0.5, 1.4];
const W = [
  [0.9, -0.6, 0.3, 0.6],
  [0.2, -0.8, 0.6, 0.2],
  [-0.4, 0.2, 0.9, 0.3],
  [0.5, 0.7, -0.6, -0.2],
  [-0.7, 0.4, -0.3, -0.5],
];

function build() {
  const g = graph({ dims: { d: 4, V: 5 }, real: { d: 4096, V: 151936, name: 'Qwen3-8B' } });
  const h = g.input('h', [1, 'd'], H);
  const Wn = g.param('W', ['V', 'd'], W);
  const t = g.label('t', [0], { tokens: ['a', 'b', 'c', 'd', 'e'] });
  const z = g.matmul(h, Wn.T, { name: 'z' });
  const L = g.logsumexp(z, -1, { keepdim: true, name: 'LSE' });
  const logp = g.sub(z, L, { name: 'log p', tex: '\\log\\mathbf{p}', elem: '\\log p' });
  const loss = g.nll_loss(logp, t, { name: 'loss', tex: '\\ell', elem: '\\ell' });
  return { g, h, W: Wn, t, z, L, logp, loss, ex: explain(g) };
}

const isElem = (e: FlowEdge, name: string, index: number[]) =>
  'node' in e.from && e.from.node.name === name && e.from.index.join(',') === index.join(',');

describe('explain on the lm_head graph', () => {
  it('h_k to the product cell has local derivative W^T[k][i]', () => {
    const { ex, z, W: Wn } = build();
    const edges = ex.edges(z);
    expect(edges).toHaveLength(4 * 5 * 3);
    const e = edges.find((e) => e.arg === 0 && isElem(e, 'h', [0, 2]) && 'cell' in e.to && e.to.cell.index.join(',') === '0,2,3')!;
    expect(ex.local(e)).toBe(Wn.T.value.get([2, 3]));
    expect(ex.local(e)).toBe(W[3][2]);
    expect(ex.cellValue(e.to as { cell: { op: typeof z; index: [number, number, number] } })).toBeCloseTo(H[2] * W[3][2], 12);
    expect(ex.elemTex(e.from)).toBe('h_{3}');
    expect(ex.elemTex(e.to)).toBe('h_{3}W^{\\top}_{3,4}');
    const card = ex.edgeCard(e);
    expect(card.title).toBe('h_{3}\\to h_{3}W^{\\top}_{3,4}');
    expect(card.local.tex).toBe('\\frac{\\partial (h_{3}W^{\\top}_{3,4})}{\\partial h_{3}}=W^{\\top}_{3,4}');
    expect(card.upstream.value).toBeCloseTo(z.grad!.get([0, 3]), 12);
    expect(card.note).toContain('5 条出边');
    const cellOut = edges.find((e) => e.arg === -1 && 'cell' in e.from && e.from.cell.index.join(',') === '0,2,3')!;
    expect(ex.local(cellOut)).toBe(1);
    expect(ex.edgeCard(cellOut).local.tex).toBe('\\frac{\\partial z_{4}}{\\partial (h_{3}W^{\\top}_{3,4})}=1');
  });

  it('LSE to log p_i has zero upstream for non-target classes', () => {
    const { ex, logp, L, t } = build();
    const lseEdges = ex.edges(logp).filter((e) => e.arg === 1);
    expect(lseEdges).toHaveLength(5);
    for (const e of lseEdges) {
      const i = (e.to as { index: number[] }).index[1];
      expect(ex.local(e)).toBe(-1);
      expect(ex.upstream(e)).toBe(i === 0 ? -1 : 0);
      expect(ex.zeroGrad(e)).toBe(i !== 0);
      expect(ex.elemTex(e.from)).toBe('\\mathrm{LSE}');
    }
    expect(ex.outEdges({ node: L, index: [0, 0] })).toHaveLength(5);
    const ids = lseEdges.map((e) => e.id);
    t.set([3]);
    const after = ex.edges(logp).filter((e) => e.arg === 1);
    expect(after.map((e) => e.id)).toEqual(ids);
    expect(after.map((e) => ex.zeroGrad(e))).toEqual([true, true, true, false, true]);
    const lossEdges = ex.edges(ex.edges().find((e) => e.op.name === 'loss')!.op);
    expect(lossEdges).toHaveLength(1);
    expect((lossEdges[0].from as { index: number[] }).index).toEqual([0, 3]);
    expect(lossEdges[0].id).toBe('loss:0:#0');
  });

  it('z_i has two out edges whose contributions add up to p_i - delta', () => {
    const { ex, z, L } = build();
    const p = (i: number) => Math.exp(z.value.get([0, i]) - L.value.get([0, 0]));
    for (let i = 0; i < 5; i++) {
      const out = ex.outEdges({ node: z, index: [0, i] });
      expect(out).toHaveLength(2);
      const total = out.reduce((s, e) => s + ex.contribution(e), 0);
      expect(total).toBeCloseTo(p(i) - (i === 0 ? 1 : 0), 12);
      const toLse = out.find((e) => e.op === L)!;
      const card = ex.edgeCard(toLse);
      expect(card.title).toBe(`z_{${i + 1}}\\to \\mathrm{LSE}`);
      expect(card.local.tex).toBe(`\\frac{\\partial \\mathrm{LSE}}{\\partial z_{${i + 1}}}=\\exp(z_{${i + 1}}-\\mathrm{LSE})`);
      expect(card.local.value).toBeCloseTo(p(i), 12);
      expect(card.upstream.tex).toBe('\\frac{\\partial\\ell}{\\partial \\mathrm{LSE}}');
      expect(card.upstream.value).toBeCloseTo(1, 12);
      expect(card.product.value).toBeCloseTo(p(i), 12);
      expect(card.note).toContain('2 条出边');
      expect(card.note).toContain('相加');
      if (i !== 0) {
        const direct = out.find((e) => e.op !== L)!;
        expect(ex.zeroGrad(direct)).toBe(true);
        expect(ex.edgeCard(direct).note).toContain('上游梯度恒为 0');
      }
    }
    const backward = ex.nodeCard(z, 'backward');
    expect(backward.title).toBe('∂ℓ/∂z');
    expect(backward.tex).toEqual([
      '\\frac{\\partial\\ell}{\\partial \\mathbf{z}}=\\Big(\\frac{\\partial \\mathrm{LSE}}{\\partial \\mathbf{z}}\\Big)^{\\top}\\frac{\\partial\\ell}{\\partial \\mathrm{LSE}}+\\Big(\\frac{\\partial \\log\\mathbf{p}}{\\partial \\mathbf{z}}\\Big)^{\\top}\\frac{\\partial\\ell}{\\partial \\log\\mathbf{p}}',
    ]);
    expect(backward.rows.map((r) => r.label)).toEqual(['shape', '真实模型', 'Jacobian', 'Jacobian', '当前梯度']);
    expect(backward.rows[1].value).toBe('Qwen3-8B：(1, 151936)');
    expect(backward.rows[2].value).toBe('∂LSE/∂z：(5,)');
    expect(backward.rows[3].value).toBe('∂log p/∂z：(5, 5)');
  });

  it('builds forward cards with shape, real model and values', () => {
    const { ex, h, z, loss, logp } = build();
    const card = ex.nodeCard(z, 'forward');
    expect(card.title).toBe('z');
    expect(card.tex).toEqual(['\\mathbf{z}=\\mathbf{h}\\mathbf{W}^{\\top}']);
    expect(card.rows[0]).toEqual({ label: 'shape', value: '(1, 5)' });
    expect(card.rows[1]).toEqual({ label: '真实模型', value: 'Qwen3-8B：(1, 151936)' });
    expect(card.rows[2].label).toBe('当前值');
    expect(card.rows[2].value.startsWith('[')).toBe(true);
    expect(ex.nodeCard(h, 'forward').tex).toEqual(['\\mathbf{h}']);
    expect(ex.nodeCard(loss, 'backward').tex).toEqual(['\\frac{\\partial\\ell}{\\partial\\ell}=1']);
    expect(ex.nodeCard(logp, 'forward').tex).toEqual(['\\log\\mathbf{p}=\\mathbf{z}-\\mathrm{LSE}']);
    const hBack = ex.nodeCard(h, 'backward');
    expect(hBack.rows.find((r) => r.label === 'Jacobian')!.value).toBe('∂z/∂h：(5, 4)');
    expect(hBack.rows.at(-1)!.value).toBe('[' + Array.from(h.grad!.data, (v) => (v < 0 ? '−' : '') + Math.abs(v).toFixed(3)).join(', ') + ']');
  });

  it('ignores consumers that do not reach the loss', () => {
    const g = graph();
    const x = g.input('x', [3], [1, 2, 3]);
    const side = g.exp(x, { name: 'side' });
    const y = g.neg(x, { name: 'y' });
    g.loss(g.sum(y, { name: 'loss' }));
    g.log(x, { name: 'after' });
    const ex = explain(g);
    const card = ex.nodeCard(x, 'backward');
    expect(card.tex).toEqual(['\\frac{\\partial\\ell}{\\partial \\mathbf{x}}=\\Big(\\frac{\\partial \\mathbf{y}}{\\partial \\mathbf{x}}\\Big)^{\\top}\\frac{\\partial\\ell}{\\partial \\mathbf{y}}']);
    expect(card.rows.map((r) => r.label)).toEqual(['shape', 'Jacobian', '当前梯度']);
    expect(card.rows[1].value).toBe('∂y/∂x：(3, 3)');
    expect(ex.nodeCard(side, 'backward').tex).toEqual(['\\frac{\\partial\\ell}{\\partial \\mathrm{side}}=0']);
  });

  it('lets card options override the generated text', () => {
    const g = graph();
    const x = g.input('x', [2], [1, 2], { card: { forward: { title: '输入', note: '说明' }, backward: { tex: ['\\mathbf{g}'] } } });
    g.loss(g.sum(x));
    const ex = explain(g);
    const forward = ex.nodeCard(x, 'forward');
    expect(forward.title).toBe('输入');
    expect(forward.note).toBe('说明');
    expect(forward.tex).toEqual(['\\mathbf{x}']);
    const backward = ex.nodeCard(x, 'backward');
    expect(backward.title).toBe('∂ℓ/∂x');
    expect(backward.tex).toEqual(['\\mathbf{g}']);
    expect(backward.rows.map((r) => r.label)).toEqual(['shape', 'Jacobian', '当前梯度']);
  });
});
