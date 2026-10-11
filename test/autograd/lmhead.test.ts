import { describe, expect, it } from 'vitest';
import { graph } from '../../src/core';

const H = [0.8, -1.1, 0.5, 1.4];
const W = [
  [0.9, -0.6, 0.3, 0.6],
  [0.2, -0.8, 0.6, 0.2],
  [-0.4, 0.2, 0.9, 0.3],
  [0.5, 0.7, -0.6, -0.2],
  [-0.7, 0.4, -0.3, -0.5],
];

function reference(h: number[], w: number[][], t: number) {
  const z = w.map((row) => row.reduce((s, v, k) => s + v * h[k], 0));
  const m = Math.max(...z);
  const L = m + Math.log(z.reduce((s, v) => s + Math.exp(v - m), 0));
  const p = z.map((v) => Math.exp(v - L));
  const loss = L - z[t];
  const dz = p.map((v, i) => v - (i === t ? 1 : 0));
  const dWT = h.map((hk) => dz.map((gi) => hk * gi));
  const dW = w.map((_, i) => h.map((hk) => hk * dz[i]));
  return { z, L, p, loss, dz, dWT, dW };
}

function closeTo(actual: unknown, expected: unknown, digits = 10): void {
  const flat = (v: unknown): number[] => (Array.isArray(v) ? v.flatMap(flat) : [v as number]);
  const a = flat(actual);
  const e = flat(expected);
  expect(a.length).toBe(e.length);
  a.forEach((v, i) => expect(v).toBeCloseTo(e[i], digits));
}

describe('lm_head with NLL loss', () => {
  function build() {
    const g = graph({ dims: { d: 4, V: 5 }, check: true });
    const h = g.input('h', [1, 'd'], H);
    const Wn = g.param('W', ['V', 'd'], W);
    const t = g.label('t', [0], { tokens: ['a', 'b', 'c', 'd', 'e'] });
    const z = g.matmul(h, Wn.T, { name: 'z' });
    const L = g.logsumexp(z, -1, { keepdim: true, name: 'LSE' });
    const logp = g.sub(z, L, { name: 'log p' });
    const loss = g.nll_loss(logp, t, { name: 'loss' });
    return { g, h, W: Wn, t, z, L, logp, loss };
  }

  it('matches the hand-computed loss and gradients', () => {
    const { g, h, W: Wn, t, z, L, logp, loss } = build();
    const ref = reference(H, W, 0);
    expect(z.shape).toEqual([1, 5]);
    expect(L.shape).toEqual([1, 1]);
    expect(logp.shape).toEqual([1, 5]);
    expect(loss.shape).toEqual([]);
    expect(g.loss()).toBe(loss);
    expect(g.node('logp')).toBe(logp);
    closeTo(z.value.toNested(), [ref.z]);
    closeTo(L.value.toNested(), [[ref.L]]);
    expect(loss.value.get([])).toBeCloseTo(ref.loss, 12);
    closeTo(z.grad!.toNested(), [ref.dz]);
    closeTo(Wn.T.grad!.toNested(), ref.dWT);
    closeTo(Wn.grad!.toNested(), ref.dW);
    const dh = W.reduce((acc, row, i) => acc.map((v, k) => v + ref.dz[i] * row[k]), [0, 0, 0, 0]);
    closeTo(h.grad!.toNested(), [dh]);
    expect(t.grad).toBeNull();
    expect(logp.grad!.toNested()).toEqual([[-1, 0, 0, 0, 0]]);
  });

  it('recomputes when the target or the parameters change', () => {
    const { g, W: Wn, t, z, loss } = build();
    let notified = 0;
    g.subscribe(() => notified++);
    t.set([3]);
    const ref = reference(H, W, 3);
    expect(notified).toBe(1);
    expect(loss.value.get([])).toBeCloseTo(ref.loss, 12);
    closeTo(z.grad!.toNested(), [ref.dz]);
    const W2 = W.map((row, i) => row.map((v, k) => v - 0.5 * ref.dW[i][k]));
    Wn.set(W2);
    const after = reference(H, W2, 3);
    expect(notified).toBe(2);
    expect(loss.value.get([])).toBeCloseTo(after.loss, 12);
    expect(after.loss).toBeLessThan(ref.loss);
    g.reset();
    expect(loss.value.get([])).toBeCloseTo(reference(H, W, 0).loss, 12);
    expect(g.version).toBe(3);
  });
});
