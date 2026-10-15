// @vitest-environment jsdom
import { afterEach, describe, expect, it } from 'vitest';
import { graph } from '../../src/core';
import { createSteps } from '../../src/steps';
import { mountPanels, mountSections, type Mounted, type PageSequence, type SectionItem } from '../../src/ui';
import { shown } from './shown';

const TOKENS = ['a', 'b', 'c'];
const Q = [
  [0.5, -1, 0.3, 0.8],
  [1.2, 0.1, -0.4, 0.6],
  [-0.7, 0.9, 1.1, -0.2],
];
const K = [
  [0.9, 0.2, -0.5, 0.4],
  [-0.3, 0.7, 0.6, -1],
  [0.1, -0.8, 0.2, 0.5],
];

function attentionGraph() {
  const g = graph({ dims: { T: 3, d: 4, V: 3 } });
  const ids = g.label('ids', [0, 2, 1], { tokens: TOKENS, classes: 3 });
  const y = g.label('y', [2, 1, 0], { tokens: TOKENS, classes: 3 });
  const q = g.param('q', ['T', 'd'], Q);
  const k = g.param('k', ['T', 'd'], K);
  const W = g.param('W', ['V', 'd'], [
    [0.5, 0.7, -0.6, -0.2],
    [-0.7, 0.4, -0.3, -0.5],
    [0.9, -0.6, 0.3, 0.6],
  ]);
  const S = g.matmul(q, k.T, { name: 'attn.S' });
  const P = g.softmax(S, -1, { name: 'P' });
  const z = g.matmul(g.matmul(P, q, { name: 'o' }), W.T, { name: 'z' });
  const logp = g.log_softmax(z, -1, { name: 'logp' });
  const lt = g.nll_loss(logp, y, { name: 'lt', reduction: 'none' });
  const loss = g.mean(lt, null, { name: 'loss' });
  const steps = createSteps(g, {
    forward: [
      { show: [q, k], title: 'qk' },
      { show: S, title: 'S' },
      { show: [P, z, logp, lt, loss], title: 'rest' },
    ],
  });
  return { g, ids, y, q, k, W, S, P, z, lt, loss, steps };
}

const mounted: Mounted[] = [];

function container(): HTMLElement {
  const el = document.createElement('div');
  document.body.append(el);
  return el;
}

afterEach(() => {
  mounted.splice(0).forEach((m) => m.dispose());
  document.body.replaceChildren();
});

const text = (el: Element | null) => (el === null ? '' : shown(el));

function button(el: HTMLElement, name: string | RegExp): HTMLButtonElement {
  const found = [...el.querySelectorAll('button')].find((b) => (typeof name === 'string' ? shown(b) === name : name.test(shown(b))));
  if (found === undefined) throw new Error(`no button ${name}`);
  return found;
}

function sequenceOf(ids: ReturnType<typeof attentionGraph>['ids'], y: ReturnType<typeof attentionGraph>['y'], g: ReturnType<typeof attentionGraph>['g']): PageSequence {
  return {
    tokens: TOKENS,
    length: 4,
    get: () => [...Array.from(ids.value.data), y.value.data[2]],
    set: (v) => g.batch(() => {
      ids.set(v.slice(0, 3));
      y.set(v.slice(1));
    }),
  };
}

describe('new panels', () => {
  it('shows per-position losses and the hint until two sgd steps from any card exist', () => {
    const { g, ids, y, lt, steps } = attentionGraph();
    const el = container();
    const all = g.nodes.filter((n) => n.kind === 'param');
    mounted.push(
      mountPanels(el, g, [
        { loss: { parts: lt, hint: '走几步看 $\\ell$ 下降' } },
        { sequence: true, labels: ['$x_1$', '$x_2 \\cdot y_1$', '$x_3 \\cdot y_2$', '$y_3$'], sgd: { params: all, lr: 0.5, label: '更新一步全部参数' }, random: true, reset: true, title: '序列' },
      ], { steps, sequence: sequenceOf(ids, y, g) }),
    );
    const rows = [...el.querySelectorAll('.vf-lossrows > span')].map(text);
    expect(rows).toEqual(Array.from(lt.value.data, (v, t) => `$\\ell_{${t + 1}}$ = ${v.toFixed(3)}`));
    expect(text(el.querySelector('.vf-spark-hint'))).toBe('走几步看 $\\ell$ 下降');
    expect(el.querySelector('.vf-spark-hint')!.namespaceURI).toBe('http://www.w3.org/1999/xhtml');
    const selects = [...el.querySelectorAll<HTMLSelectElement>('.vf-seqrow select')];
    expect(selects.map((s) => s.value)).toEqual(['0', '2', '1', '0']);
    expect([...el.querySelectorAll('.vf-seqrow label')].map((l) => text(l.firstElementChild))).toEqual(['$x_1$', '$x_2 \\cdot y_1$', '$x_3 \\cdot y_2$', '$y_3$']);

    const before = g.loss().value.data[0];
    button(el, '更新一步全部参数（$\\eta = 0.5$）').click();
    button(el, '更新一步全部参数（$\\eta = 0.5$）').click();
    expect(g.loss().value.data[0]).toBeLessThan(before);
    expect(el.querySelector('.vf-spark path')).not.toBeNull();
    expect(el.querySelector('.vf-spark-hint')).toBeNull();

    selects[1].value = '1';
    selects[1].dispatchEvent(new Event('change', { bubbles: true }));
    expect(Array.from(ids.value.data)).toEqual([0, 1, 1]);
    expect(Array.from(y.value.data)).toEqual([1, 1, 0]);
    expect(el.querySelector('.vf-spark path')).toBeNull();

    button(el, '随机序列').click();
    expect(Array.from(ids.value.data).every((v) => Number.isInteger(v) && v >= 0 && v < 3)).toBe(true);

    button(el, '重置').click();
    expect(Array.from(ids.value.data)).toEqual([0, 2, 1]);
    expect(Array.from(y.value.data)).toEqual([2, 1, 0]);
    expect(g.loss().value.data[0]).toBeCloseTo(before, 12);
  });

  it('refuses a sequence card without a sequence', () => {
    const { g } = attentionGraph();
    expect(() => mountPanels(container(), g, [{ sequence: true }])).toThrow('sequence panel: mountPanels needs the sequence option');
  });

  it('draws one attention grid per node with token names and dots above the diagonal', () => {
    const { g, ids, P } = attentionGraph();
    const el = container();
    mounted.push(mountPanels(el, g, [{ attention: [P, P], ids, title: '注意力权重 P' }]));
    const grids = [...el.querySelectorAll<HTMLElement>('.vf-attn')];
    expect(grids).toHaveLength(2);
    expect(el.querySelector<HTMLElement>('.vf-attg')!.style.gridTemplateColumns).toBe('repeat(2, minmax(0, 1fr))');
    const cells = [...grids[1].children].map(text);
    expect(cells.slice(0, 4)).toEqual(['$h_{2}$', 'a', 'c', 'b']);
    expect(cells.slice(4, 8)).toEqual(['a', P.value.get([0, 0]).toFixed(2), '·', '·']);
    expect(cells[15]).toBe(P.value.get([2, 2]).toFixed(2));
    expect(() => mountPanels(container(), g, [{ attention: [g.nodes.find((n) => n.name === 'q')!], ids }])).toThrow(/must be a square matrix/);
  });

  it('adds the extra key columns after the token columns, without dots', () => {
    const { g, ids, q } = attentionGraph();
    const el = container();
    mounted.push(mountPanels(el, g, [{ attention: [q], ids, extra_keys: ['$c_{1}$'] }]));
    const cells = [...el.querySelector<HTMLElement>('.vf-attn')!.children].map(text);
    expect(cells.slice(0, 5)).toEqual(['$h_{1}$', 'a', 'c', 'b', '$c_{1}$']);
    expect(cells.slice(5, 10)).toEqual(['a', q.value.get([0, 0]).toFixed(2), '·', '·', q.value.get([0, 3]).toFixed(2)]);
    expect(() => mountPanels(container(), g, [{ attention: [q], ids, extra_keys: ['$c_{1}$', '$c_{2}$'] }])).toThrow(/must be a \(T, T \+ 2\) matrix/);
  });

  it('puts several matrices in one card with captions, row names and frames on every label column', () => {
    const { g, y, W } = attentionGraph();
    const el = container();
    mounted.push(
      mountPanels(el, g, [
        {
          title: 'lm_head 权重',
          matrices: [
            { matrix: W.T, caption: '$W^\\top$', limit: 1.2 },
            { matrix: W.T, grad: true, caption: '$\\partial\\ell/\\partial W^\\top$', limit: 0.3 },
          ],
          highlight: { node: y, all: true },
          rows: 'h',
          note: '说明',
        },
      ]),
    );
    expect([...el.querySelectorAll('.vf-wcap')].map(text)).toEqual(['$W^\\top$', '$\\partial\\ell/\\partial W^\\top$']);
    const grid = el.querySelectorAll('.vf-wmat')[0];
    expect([...grid.querySelectorAll('.vf-rh')].map(text)).toEqual(['$h_{1}$', '$h_{2}$', '$h_{3}$', '$h_{4}$']);
    expect(grid.querySelector('.vf-frame')).toBeNull();
    const framed = [...grid.querySelectorAll<HTMLElement>('.vf-ct')].map((c) => c.style.gridColumn);
    expect(new Set(framed)).toEqual(new Set(['2', '3', '4']));
    expect(framed).toHaveLength(12);
    const cell = [...grid.children].find((c) => (c as HTMLElement).style.gridRow === '2' && (c as HTMLElement).style.gridColumn === '2') as HTMLElement;
    expect(cell.style.background).toContain(`${(8 + Math.min(0.5 / 1.2, 1) * 62).toFixed(0)}%`);
  });
});

function sectionItem(fixture: ReturnType<typeof attentionGraph>): SectionItem {
  const { q, k, S, ids } = fixture;
  return {
    title: 'RoPE',
    rope: { q: [q], k: [k], rows: [-1, 1], head: 1, theta: 10, dh: 4, positions: 24 },
    rows: [
      { columns: [300, '1fr'], cells: [['rope_dial'], [{ lead: '开头' }, { text: '### 一、问题\n\n正文 $x$' }]] },
      { columns: [340, '1fr'], cells: [['rope_curve'], [{ text: '### 四' }]] },
      { cells: [[{ text: '### 七\n\n对比' }, { buttons: [{ label: '看打分', go: S }] }, { rope_compare: { ids } }]] },
    ],
  };
}

describe('mountSections', () => {
  it('lays out rows, columns, markdown and the lead paragraph', () => {
    const fixture = attentionGraph();
    const el = container();
    mounted.push(mountSections(el, fixture.g, [sectionItem(fixture)], { steps: fixture.steps }));
    expect(text(el.querySelector('.vf-section h2'))).toBe('RoPE');
    const rows = [...el.querySelectorAll<HTMLElement>('.vf-row')];
    expect(rows.map((r) => r.style.gridTemplateColumns)).toEqual(['minmax(0, 300px) minmax(0, 1fr)', 'minmax(0, 340px) minmax(0, 1fr)', '']);
    expect(rows[2].classList.contains('vf-cols')).toBe(false);
    expect(text(rows[0].querySelector('.vf-lead p'))).toBe('开头');
    expect(text(rows[0].querySelector('.vf-md:not(.vf-lead) h3'))).toBe('一、问题');
    expect(rows[0].querySelector('.vf-md:not(.vf-lead) p .vf-tex')).not.toBeNull();
  });

  it('rotates q and k in the dial, keeps the dot product when both positions move and updates the curve note', () => {
    const fixture = attentionGraph();
    const el = container();
    mounted.push(mountSections(el, fixture.g, [sectionItem(fixture)], { steps: fixture.steps }));
    const kv = () => [...el.querySelectorAll('.vf-ropekv dd')].map(text);
    expect([...el.querySelectorAll('.vf-seg')[0].querySelectorAll('button')].map(text)).toEqual(['$\\omega = 1.000$', '$\\omega = 0.316$']);
    expect(text(el.querySelector('.vf-ropecap'))).toBe('选一对分量：$\\omega = 1$ 是第 1、3 分量，$\\omega = 0.316$ 是第 2、4 分量');
    const q = Q[2];
    const k = K[0];
    const angle = 2;
    const qr = [q[0] * Math.cos(angle) - q[2] * Math.sin(angle), q[2] * Math.cos(angle) + q[0] * Math.sin(angle)];
    const pair = qr[0] * k[0] + qr[1] * k[2];
    const first = kv();
    expect(first.slice(0, 3)).toEqual(['2.000 rad', '0.000 rad', '2']);
    expect(first[3]).toBe(`${pair.toFixed(3).replace('-', '−')}（转之前 ${(q[0] * k[0] + q[2] * k[2]).toFixed(3).replace('-', '−')}）`);
    button(el, '$t$ 和 $s$ 同时 +1').click();
    expect(kv().slice(0, 2)).toEqual(['3.000 rad', '1.000 rad']);
    expect(kv()[3]).toBe(first[3]);
    expect(kv()[4]).toBe(first[4]);
    expect(text(el.querySelector('.vf-rope-curve .vf-note'))).toContain('$t-s=2$。');
    expect(text(el.querySelector('.vf-rope-curve .vf-note'))).toContain('$\\bar{\\mathbf{q}}$ 取自最后一个位置的 head 1，$\\bar{\\mathbf{k}}$ 取自第一个位置');
    const s = el.querySelectorAll<HTMLInputElement>('.vf-ropectl input')[1];
    expect(s.max).toBe('23');
    s.value = '5';
    s.dispatchEvent(new Event('input', { bubbles: true }));
    expect(text(el.querySelector('.vf-rope-curve .vf-note'))).toBe('$t-s=-2<0$，在 causal attention 里这一对会被 mask 挡住；曲线只画了 $t-s\\ge 0$ 的部分。');
    button(el, '回到 $t = 2$，$s = 0$').click();
    expect(kv().slice(0, 2)).toEqual(['2.000 rad', '0.000 rad']);
    button(el, '$\\omega = 0.316$').click();
    expect(kv()[0]).toBe(`${(2 / Math.sqrt(10)).toFixed(3)} rad`);
    expect(el.querySelectorAll('.vf-rope-curve svg circle').length).toBe(25);
  });

  it('compares scores with and without rotation and jumps to the step of a node', () => {
    const fixture = attentionGraph();
    const el = container();
    let jumped = 0;
    mounted.push(mountSections(el, fixture.g, [sectionItem(fixture)], { steps: fixture.steps, onJump: () => jumped++ }));
    const caps = [...el.querySelectorAll('.vf-ropecmp .vf-cap')].map(text);
    expect(caps).toEqual([
      '不加 RoPE：$\\bar{\\mathbf{q}}\\bar{\\mathbf{k}}^{\\top}/\\sqrt{d_h}$',
      "加了 RoPE：$\\mathbf{q}'\\mathbf{k}'^{\\top}/\\sqrt{d_h}$",
      '不加 RoPE 的注意力权重',
      '加了 RoPE 的注意力权重',
    ]);
    const grids = [...el.querySelectorAll('.vf-cmpg')];
    const s0 = (Q[2].reduce((a, v, c) => a + v * K[0][c], 0) / 2).toFixed(2).replace('-', '−');
    expect(text(grids[0].children[13])).toBe(s0);
    const rotated = [1, 1 / Math.sqrt(10)].reduce((a, w, i) => {
      const [c, s] = [Math.cos(2 * w), Math.sin(2 * w)];
      return a + (Q[2][i] * c - Q[2][i + 2] * s) * K[0][i] + (Q[2][i + 2] * c + Q[2][i] * s) * K[0][i + 2];
    }, 0);
    expect(text(grids[1].children[13])).toBe((rotated / 2).toFixed(2).replace('-', '−'));
    expect(grids[0].querySelectorAll('.vf-dg')).toHaveLength(3);
    expect(text(grids[2].children[5])).toBe('1.00');
    expect(text(el.querySelector('.vf-cmpnote'))).toContain('比如 “b” 看 “a”，距离是 2');
    button(el, '看打分').click();
    expect(fixture.steps.current).toBe(2);
    expect(jumped).toBe(1);
  });

  it('jumps to the folded module step when no step shows the node', () => {
    const fixture = attentionGraph();
    const list = fixture.steps.list as unknown as { kind?: string; module?: string; nodes: unknown[] }[];
    list[2].nodes = [];
    Object.assign(list[3], { kind: 'module', module: 'attn' });
    const el = container();
    mounted.push(mountSections(el, fixture.g, [sectionItem(fixture)], { steps: fixture.steps }));
    button(el, '看打分').click();
    expect(fixture.steps.current).toBe(3);
  });

  it('reads q and k laid out as (T, H, dh) the same as (T, H dh)', () => {
    const shown = (split: boolean) => {
      const g = graph({ dims: { T: 3 } });
      const shape = split ? ['T', 2, 2] : ['T', 4];
      const data = (m: number[][]) => (split ? m.map((r) => [r.slice(0, 2), r.slice(2)]) : m);
      const q = g.param('q', shape, data(Q));
      const k = g.param('k', shape, data(K));
      g.sum(g.mul(q, k), null, { name: 'loss' });
      const el = container();
      const item: SectionItem = { rope: { q: [q], k: [k], rows: [-1, 1], head: 2, theta: 10, dh: 2, positions: 24 }, rows: [{ cells: [['rope_dial', 'rope_curve']] }] };
      mounted.push(mountSections(el, g, [item]));
      return [...el.querySelectorAll('.vf-ropekv dd, .vf-rope-curve .vf-note')].map(text);
    };
    const flat = shown(false);
    expect(flat[3]).toContain(`（转之前 ${(Q[2][2] * K[0][2] + Q[2][3] * K[0][3]).toFixed(3).replace('-', '−')}）`);
    expect(shown(true)).toEqual(flat);
  });

  it('refuses rope blocks without the rope option', () => {
    const fixture = attentionGraph();
    expect(() => mountSections(container(), fixture.g, [{ rows: [{ cells: [['rope_dial']] }] }])).toThrow("section: rope_dial needs the section's rope option");
  });
});
