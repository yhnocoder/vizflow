import { describe, expect, it } from 'vitest';
import { graph } from '../../src/core';
import { explain } from '../../src/explain';
import { allVisible, createSteps } from '../../src/steps';
import { formatNumber } from '../../src/tensor';

const H = [0.8, -1.1, 0.5, 1.4];
const W = [
  [0.9, -0.6, 0.3, 0.6],
  [0.2, -0.8, 0.6, 0.2],
  [-0.4, 0.2, 0.9, 0.3],
  [0.5, 0.7, -0.6, -0.2],
  [-0.7, 0.4, -0.3, -0.5],
];

function lmHead() {
  const g = graph({ dims: { d: 4, V: 5 } });
  const h = g.input('h', [1, 'd'], H);
  const Wn = g.param('W', ['V', 'd'], W);
  const t = g.label('t', [0], { tokens: ['the', 'cat', 'sat', 'on', 'mat'] });
  const z = g.matmul(h, Wn.T, { name: 'z' });
  const L = g.logsumexp(z, -1, { keepdim: true, name: 'LSE' });
  const logp = g.sub(z, L, { name: 'log p', tex: '\\log\\mathbf{p}', elem: '\\log p' });
  const loss = g.nll_loss(logp, t, { name: 'loss', tex: '\\ell', elem: '\\ell' });
  const steps = createSteps(g, {
    overview: { title: '总览', text: '目标 token 是 {{t:token}}（第 {{t+1}} 个）。' },
    forward: [
      { show: [h, Wn.T], title: 'h, Wᵀ', text: '$h_{1} = {{h[0]}}$，正文里 {{h[1]}}。' },
      { show: z, active: [h, Wn.T, z], title: 'z = hWᵀ', text: '$z_{ {{t+1}} } = {{z[t]}}$' },
      { show: L, title: 'LSE', text: 'LSE = {{LSE:2}}' },
      { show: logp, title: 'log p', text: '{{log p[t]}} 与 {{logp.grad[0, t]}}' },
      { show: loss, title: 'ℓ', text: '{{loss:3}}' },
    ],
  });
  return { g, h, W: Wn, t, z, L, logp, loss, steps };
}

const X = [
  [2.0, 1.0, 0.2, -0.6, -1.4],
  [-0.5, 1.8, 0.3, 0.9, -1.0],
  [0.4, -0.2, 1.5, 0.1, 0.6],
];

function softmaxBatch() {
  const g = graph({ dims: { B: 3, C: 5 } });
  const x = g.input('X', ['B', 'C'], X);
  const t = g.label('t', [[0], [1], [2]]);
  const U = g.exp(x, { name: 'U' });
  const s = g.sum(U, -1, { keepdim: true, name: 's' });
  const P = g.div(U, s, { name: 'P' });
  const y = g.gather(P, -1, t, { name: 'y' });
  const lb = g.neg(g.log(y), { name: 'lb' });
  const loss = g.mean(lb, { name: 'loss' });
  return { g, x, t, U, s, P, y, lb, loss };
}

describe('automatic backward steps', () => {
  it('gives the five lm_head steps', () => {
    const { steps, h, W: Wn, z, L, logp, loss } = lmHead();
    const backward = steps.list.filter((s) => s.dir === 'backward');
    expect(backward.map((s) => s.nodes)).toEqual([[loss, logp], [L], [z], [Wn.T], [h]]);
    expect(backward.map((s) => s.index)).toEqual([1, 2, 3, 4, 5]);
    expect(backward.map((s) => s.title)).toEqual(['∂ℓ/∂log p', '∂ℓ/∂LSE', '∂ℓ/∂z', '∂ℓ/∂W.T', '∂ℓ/∂h']);
    expect(backward.map((s) => s.label)).toEqual(['∂log p', '∂LSE', '∂z', '∂W.T', '∂h']);
    expect(backward[2].text).toBe(
      '$$\\frac{\\partial\\ell}{\\partial \\mathbf{z}}=\\Big(\\frac{\\partial \\mathrm{LSE}}{\\partial \\mathbf{z}}\\Big)^{\\top}\\frac{\\partial\\ell}{\\partial \\mathrm{LSE}}+\\Big(\\frac{\\partial \\log\\mathbf{p}}{\\partial \\mathbf{z}}\\Big)^{\\top}\\frac{\\partial\\ell}{\\partial \\log\\mathbf{p}}$$',
    );
    expect(steps.visible).toEqual([h, Wn.T, z, L, logp, loss]);
    expect(steps.list).toHaveLength(11);
    expect(steps.list[0]).toMatchObject({ dir: 'forward', index: 0, title: '总览', label: '总览' });
    expect(steps.list[2].label).toBe('z = hWᵀ');
  });

  it('merges lb and y into the first softmax batch step', () => {
    const { g, x, U, s, P, y, lb, loss } = softmaxBatch();
    const steps = createSteps(g, {
      forward: [
        { show: x, title: 'X', text: '' },
        { show: U, title: 'U', text: '' },
        { show: s, title: 's', text: '' },
        { show: P, title: 'P', text: '' },
        { show: [y, lb, loss], title: 'ℓ', text: '' },
      ],
      backward: { auto: true, merge: [[lb, y]], override: { 1: { label: '∂y', text: '第一步。' }, 5: { title: '到 logits' } } },
    });
    const backward = steps.list.filter((st) => st.dir === 'backward');
    expect(backward.map((st) => st.nodes)).toEqual([[loss, lb, y], [P], [s], [U], [x]]);
    expect(backward[0]).toMatchObject({ title: '∂ℓ/∂lb', label: '∂y', text: '第一步。' });
    expect(backward[4]).toMatchObject({ title: '到 logits', label: '∂X' });
    const plain = createSteps(g, {
      forward: [{ show: [x, U, s, P, y, lb, loss], title: '全部', text: '' }],
    });
    expect(plain.list.filter((st) => st.dir === 'backward').map((st) => st.nodes)).toEqual([[loss, lb], [y], [P], [s], [U], [x]]);
  });

  it('generates no backward steps for a graph without a loss node', () => {
    const g = graph();
    const x = g.input('x', [5], [1, 2, 3, 4, 5]);
    const p = g.softmax(x, -1, { name: 'p' });
    const steps = createSteps(g, { forward: [{ show: [x, p], title: 'p', text: '{{p[0]:2}}' }] });
    expect(steps.list.map((st) => st.dir)).toEqual(['forward', 'forward']);
    expect(steps.text(1).body).toBe('0.01');
    const merged = { forward: [{ show: [x, p], title: 'p', text: '' }], backward: { auto: true as const, merge: [[x, p]] } };
    expect(() => createSteps(g, merged)).toThrow('backward: the graph has no loss node, so merge and override cannot apply');
    const overridden = { forward: [{ show: [x, p], title: 'p', text: '' }], backward: { auto: true as const, override: { 1: { title: 'x' } } } };
    expect(() => createSteps(g, overridden)).toThrow('backward: the graph has no loss node, so merge and override cannot apply');
  });

  it('uses an empty body when a forward step or the overview has no text', () => {
    const g = graph();
    const x = g.input('x', [5], [1, 2, 3, 4, 5]);
    const p = g.softmax(x, -1, { name: 'p' });
    const steps = createSteps(g, { overview: { title: 'all' }, forward: [{ show: [x, p], title: 'p' }] });
    expect(steps.text(0).body).toBe('');
    expect(steps.text(1).body).toBe('');
  });

  it('rejects merge groups and overrides that point outside the steps', () => {
    const { g, x, U, lb, y } = softmaxBatch();
    expect(() => createSteps(g, { forward: [{ show: [x, U], title: 'x', text: '' }], backward: { auto: true, merge: [[lb, y]] } })).toThrow(
      "merge: node 'lb' is not in any backward step",
    );
    expect(() => createSteps(g, { forward: [{ show: [x, U], title: 'x', text: '' }], backward: { auto: true, override: { 3: { title: 'x' } } } })).toThrow(
      'override: backward step 3 does not exist (there are 2 steps)',
    );
  });
});

describe('step state', () => {
  it('reveals and activates nodes per step and notifies subscribers', () => {
    const { steps, h, W: Wn, z, L, logp, loss } = lmHead();
    let calls = 0;
    const off = steps.subscribe(() => calls++);
    const overview = steps.state();
    expect(overview).toMatchObject({ dir: 'forward', index: 0, focus: 0 });
    expect([h, z, loss].map((n) => overview.revealed(n) && overview.active(n))).toEqual([true, true, true]);
    steps.go(2);
    let st = steps.state();
    expect(st.index).toBe(2);
    expect([h, Wn.T, z, L].map((n) => st.revealed(n))).toEqual([true, true, true, false]);
    expect([h, Wn.T, z, L].map((n) => st.active(n))).toEqual([true, true, true, false]);
    steps.go(3);
    st = steps.state();
    expect([h, z, L].map((n) => st.active(n))).toEqual([false, false, true]);
    steps.go(8);
    st = steps.state();
    expect(st).toMatchObject({ dir: 'backward', index: 3 });
    expect([loss, logp, L, z, Wn.T, h].map((n) => st.revealed(n))).toEqual([true, true, true, true, false, false]);
    expect([loss, logp, L, z].map((n) => st.active(n))).toEqual([false, false, false, true]);
    steps.next();
    steps.next();
    steps.next();
    expect(steps.current).toBe(10);
    steps.prev();
    expect(steps.current).toBe(9);
    steps.go(-5);
    expect(steps.current).toBe(0);
    expect(calls).toBe(7);
    steps.setFocus(2);
    expect(steps.state().focus).toBe(2);
    expect(calls).toBe(8);
    off();
    steps.go(1);
    expect(calls).toBe(8);
    steps.dispose();
  });

  it('setFocus rejects rows outside the visible nodes', () => {
    const { g, x, P, loss } = softmaxBatch();
    const steps = createSteps(g, { forward: [{ show: [x, P, loss], title: 'x', text: '' }] });
    steps.setFocus(2);
    expect(steps.focus).toBe(2);
    expect(() => steps.setFocus(3)).toThrow('setFocus: 3 is out of range; the visible nodes have 3 row(s), so b must be between 0 and 2');
    expect(() => steps.setFocus(-1)).toThrow('setFocus: -1 is out of range');
    expect(() => steps.setFocus(1.5)).toThrow('setFocus: 1.5 is out of range');
    const single = createSteps(g, { forward: [{ show: loss, title: 'l', text: '' }] });
    expect(() => single.setFocus(1)).toThrow('the visible nodes have 1 row(s), so b must be between 0 and 0');
  });

  it('allVisible shows and highlights everything', () => {
    const { g, z } = lmHead();
    const st = allVisible(g);
    expect(st).toMatchObject({ dir: 'forward', index: 0, focus: 0 });
    expect(st.revealed(z)).toBe(true);
    expect(st.active(z)).toBe(true);
  });

  it('zeroGrad edges follow the target token', () => {
    const { g, t, logp } = lmHead();
    const ex = explain(g);
    const zero = () => ex.edges(logp).filter((e) => e.arg === 0).map((e) => ex.zeroGrad(e));
    expect(zero()).toEqual([false, true, true, true, true]);
    t.set([3]);
    expect(zero()).toEqual([true, true, true, false, true]);
  });
});

describe('text templates', () => {
  it('evaluates node values, labels, tokens and offsets, and follows changes', () => {
    const { steps, t, z } = lmHead();
    expect(steps.text(0).body).toBe('目标 token 是 the（第 1 个）。');
    expect(steps.text(2).body).toBe(`$z_{ 1 } = ${formatNumber(z.value.get([0, 0]), { tex: true })}$`);
    t.set([3]);
    expect(steps.text(0).body).toBe('目标 token 是 on（第 4 个）。');
    expect(steps.text(2).body).toBe(`$z_{ 4 } = ${formatNumber(z.value.get([0, 3]), { tex: true })}$`);
    expect(steps.text(2).title).toBe('z = hWᵀ');
    expect(steps.text(3).body).toBe(`LSE = ${steps.graph.node('LSE')!.value.get([0, 0]).toFixed(2)}`);
    expect(steps.text(4).body).toBe(`${formatNumber(steps.graph.node('log p')!.value.get([0, 3]))} 与 −1.00`);
    expect(steps.text(5).body).toBe(steps.graph.loss().value.get([]).toFixed(3));
  });

  it('writes negative numbers with ASCII minus inside formulas and U+2212 outside', () => {
    const { steps } = lmHead();
    expect(steps.text(1).body).toBe('$h_{1} = 0.800$，正文里 −1.10。');
  });

  it('uses the focus row for b and label[b]', () => {
    const { g, x, t, P, loss } = softmaxBatch();
    const steps = createSteps(g, {
      forward: [{ show: [x, P, loss], title: 'x', text: '样本 {{b+1}} 的目标是 {{t[b]}}，$X = {{X[b, t[b]]}}$，P = {{P[b, t[b]]:2}}，$g = {{X.grad[b, 0]}}$' }],
    });
    expect(steps.text(1).body).toBe('样本 1 的目标是 0，$X = 2.00$，P = 0.61，$g = ' + formatNumber(x.grad!.get([0, 0]), { tex: true }) + '$');
    steps.setFocus(2);
    expect(steps.text(1).body).toBe('样本 3 的目标是 2，$X = 1.50$，P = 0.46，$g = ' + formatNumber(x.grad!.get([2, 0]), { tex: true }) + '$');
    t.set([[4], [4], [4]]);
    expect(steps.text(1).body.startsWith('样本 3 的目标是 4，$X = 0.600$')).toBe(true);
  });

  it('evaluates a bare b as the current focus', () => {
    const { g, x } = softmaxBatch();
    const steps = createSteps(g, { forward: [{ show: x, title: '样本 {{b+1}}', text: '$b = {{b}}$，{{b-1}}，{{b:2}}' }] });
    expect(steps.text(1)).toEqual({ title: '样本 1', body: '$b = 0$，−1，0.00' });
    steps.setFocus(2);
    expect(steps.text(1)).toEqual({ title: '样本 3', body: '$b = 2$，1，2.00' });
    expect(() => createSteps(g, { forward: [{ show: x, title: 'x', text: '{{b:token}}' }] })).toThrow("cannot parse template '{{b:token}}': 'b' has no tokens");
    expect(() => createSteps(g, { forward: [{ show: x, title: 'x', text: '{{b[0]}}' }] })).toThrow("cannot parse template '{{b[0]}}': no node name matches 'b[0]'");
  });

  it('resolves node names that contain dots and spaces by longest prefix', () => {
    const { g, W: Wn, logp, t } = lmHead();
    const steps = createSteps(g, {
      forward: [{ show: Wn.T, title: 'W', text: '{{W.T[0,1]}}，{{W.T.grad[0, t]}}，{{ W . T [1, 0] }}，{{log p[0, t]}}，{{W[1, 0]}}' }],
    });
    const expected = [Wn.T.value.get([0, 1]), Wn.T.grad!.get([0, 0]), Wn.T.value.get([1, 0]), logp.value.get([0, 0]), Wn.value.get([1, 0])];
    expect(steps.text(1).body).toBe(expected.map((v) => formatNumber(v)).join('，'));
    t.set([2]);
    expect(steps.text(1).body.split('，')[1]).toBe(formatNumber(Wn.T.grad!.get([0, 2])));
  });

  it('reserves b for the current sample index', () => {
    const g = graph();
    const b = g.input('b', [2], [1, 2]);
    g.loss(g.sum(b));
    const make = (text: string) => () => createSteps(g, { forward: [{ show: b, title: 'b', text }] });
    const reserved = "'b' is the current sample index; rename node 'b'";
    expect(make('{{b}}')).toThrow(reserved);
    expect(make('{{b+1}}')).toThrow(reserved);
    expect(make('{{b[b]}}')).toThrow(reserved);
    expect(make('{{b[0]}}')).not.toThrow();
    const other = graph();
    const x = other.input('x', [2], [1, 2]);
    other.label('t', [[0], [1]]);
    other.loss(other.sum(x));
    expect(() => createSteps(other, { forward: [{ show: x, title: 'x', text: '{{x[t[b]]}}' }] })).not.toThrow();
    expect(() => createSteps(other, { forward: [{ show: x, title: 'x', text: '{{t[b]}}' }] })).not.toThrow();
  });

  it('rejects templates that cannot be resolved when the steps are created', () => {
    const { g, z } = lmHead();
    const make = (text: string) => () => createSteps(g, { forward: [{ show: z, title: 'z', text }] });
    expect(make('{{nope}}')).toThrow("cannot parse template '{{nope}}': no node name matches 'nope'");
    expect(make('{{z}}')).toThrow("cannot parse template '{{z}}': node 'z' with shape (1, 5) needs a subscript");
    expect(make('{{z[0, 1, 2]}}')).toThrow('takes 2 subscripts, got 3');
    expect(make('{{t.grad}}')).toThrow("label 't' has no gradient");
    expect(make('{{z[0]:token}}')).toThrow("node 'z' has no tokens");
    expect(make('{{z[0] * 2}}')).toThrow("cannot parse template '{{z[0] * 2}}': unexpected '*2'");
    expect(make('$z_{{{t}}}$')).toThrow("starts with '{{{'");
    expect(make('$z_{ {{t}}}$')).toThrow("ends with '}}}'");
    expect(make('{{z[0]')).toThrow('is not closed');
    expect(make('{{z[0}}')).toThrow('unclosed subscript');
    expect(make('{{z[0].grad}}')).toThrow("unexpected '.grad'");
    expect(make('$\\mathbf{h}$ 与 {{z[0]}} 可以共存')).not.toThrow();
    const steps = make('{{z[9]}}')();
    expect(() => steps.text(1)).toThrow("cannot parse template '{{z[9]}}': index (0, 9) is out of range for shape (1, 5)");
  });
});
