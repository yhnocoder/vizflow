// @vitest-environment jsdom
import { afterEach, describe, expect, it, vi } from 'vitest';
import { graph } from '../../src/core';
import { createSteps, renderTemplate } from '../../src/steps';
import { mountPanels, mountSteps, type Mounted } from '../../src/ui';

const X = [2.0, 1.0, 0.2, -0.6, -1.4];
const TOKENS = [' mat', ' floor', ' sofa', ' dog', ' sat'];

function softmaxGraph() {
  const g = graph({ dims: { V: 5 } });
  const x = g.input('x', ['V'], X);
  const t = g.label('t', 0, { tokens: TOKENS });
  const p = g.softmax(x, -1, { name: 'p' });
  const logp = g.log(p, { name: 'log p' });
  const loss = g.nll_loss(logp, t, { name: 'loss' });
  const steps = createSteps(g, {
    overview: { title: '总览', text: '$\\ell = {{loss:3}}$，目标 {{t:token}}' },
    forward: [
      { show: x, title: 'logits', text: 'x = {{x[t]:2}}' },
      { show: p, title: 'softmax', text: 'p = {{p[t]:3}}' },
      { show: [logp, loss], title: 'loss', text: '{{loss:3}}' },
    ],
  });
  return { g, x, t, p, logp, loss, steps };
}

function batchGraph() {
  const g = graph({ dims: { B: 2, C: 3 } });
  const x = g.input('X', ['B', 'C'], [
    [1, 0, -1],
    [0.5, 2, 0],
  ]);
  const t = g.label('t', [0, 1], { tokens: ['a', 'b', 'c'] });
  const logp = g.log_softmax(x, -1, { name: 'log p' });
  const loss = g.nll_loss(logp, t, { name: 'loss' });
  const steps = createSteps(g, { forward: [{ show: [x, logp, loss], title: 'all', text: '{{t[b]}}' }] });
  return { g, x, t, logp, loss, steps };
}

const mounted: Mounted[] = [];

function container(): HTMLElement {
  const el = document.createElement('div');
  document.body.append(el);
  return el;
}

function panels(...args: Parameters<typeof mountPanels>): HTMLElement {
  mounted.push(mountPanels(...args));
  return typeof args[0] === 'string' ? document.querySelector<HTMLElement>(args[0])! : args[0];
}

afterEach(() => {
  mounted.splice(0).forEach((m) => m.dispose());
  document.body.replaceChildren();
});

const text = (el: Element | null) => el?.textContent ?? '';

function slide(input: HTMLInputElement, value: number) {
  input.value = String(value);
  input.dispatchEvent(new Event('input', { bubbles: true }));
}

function button(el: HTMLElement, name: string | RegExp): HTMLButtonElement {
  const found = [...el.querySelectorAll('button')].find((b) => (typeof name === 'string' ? b.textContent === name : name.test(b.textContent ?? '')));
  if (found === undefined) throw new Error(`no button ${name}`);
  return found;
}

describe('renderTemplate', () => {
  it('evaluates the same templates as steps.text', () => {
    const { g, steps } = softmaxGraph();
    expect(renderTemplate(g, '$\\ell = {{loss:3}}$，目标 {{t:token}}', 0)).toBe(steps.text(0).body);
    expect(renderTemplate(g, 'p = {{p[t]:3}} 与 {{p[t]:3}}', 0)).toBe('p = 0.609 与 0.609');
  });

  it('uses the focus row in batch graphs', () => {
    const { g } = batchGraph();
    expect(renderTemplate(g, '{{t[b]:token}} {{b+1}}', 1)).toBe('b 2');
  });

  it('rejects unknown nodes with the template text', () => {
    const { g } = softmaxGraph();
    expect(() => renderTemplate(g, '{{nope}}', 0)).toThrow('{{nope}}');
  });
});

describe('loss panel', () => {
  it('shows the loss, the text line and the empty curve hint', () => {
    const { g } = softmaxGraph();
    const el = panels(container(), g, [{ loss: { text: '目标 "{{t:token}}"，p = {{p[t]:3}}' } }]);
    expect(text(el.querySelector('.vf-big'))).toBe('ℓ =0.495');
    expect(text(el.querySelector('.vf-sub'))).toBe('目标 " mat"，p = 0.609');
    expect(text(el.querySelector('.vf-spark'))).toBe('走几步看损失下降');
    expect(el.querySelector('.vf-btns')).toBeNull();
  });

  it('accepts the plain string form', () => {
    const { g } = softmaxGraph();
    const el = panels(container(), g, ['loss']);
    expect(text(el.querySelector('.vf-loss-value'))).toBe('0.495');
    expect(el.querySelector('.vf-sub')).toBeNull();
  });

  it('runs gradient steps, records the curve and clears it on other changes', () => {
    const { g, x } = softmaxGraph();
    const el = panels(container(), g, [
      { loss: { sgd: { params: [x], lr: 1 }, random: { nodes: [x], range: [-3, 3] }, reset: true } },
      { sliders: x },
    ]);
    const sgd = button(el, '更新一步 x（η = 1）');
    expect(sgd.classList.contains('vf-pri')).toBe(true);
    const grad = Array.from(x.grad!.data);
    sgd.click();
    expect(Array.from(x.value.data)).toEqual(X.map((v, i) => v - grad[i]));
    sgd.click();
    const loss = Number(text(el.querySelector('.vf-loss-value')));
    expect(loss).toBeLessThan(0.495);
    expect(el.querySelectorAll('.vf-spark path')).toHaveLength(1);
    expect(el.querySelector('.vf-spark path')!.getAttribute('d')!.split('L')).toHaveLength(3);

    slide(el.querySelector<HTMLInputElement>('input[type="range"]')!, 0);
    expect(el.querySelectorAll('.vf-spark path')).toHaveLength(0);

    vi.spyOn(Math, 'random').mockReturnValue(0.123456);
    button(el, '随机 x').click();
    vi.restoreAllMocks();
    expect(Array.from(x.value.data)).toEqual(X.map(() => -2.26));

    button(el, '重置').click();
    expect(Array.from(x.value.data)).toEqual(X);
    expect(text(el.querySelector('.vf-loss-value'))).toBe('0.495');
  });

  it('uses the given button labels', () => {
    const { g, x } = softmaxGraph();
    const el = panels(container(), g, [{ loss: { sgd: { params: [x], lr: 0.5, label: '走一步' }, random: { nodes: [x], range: [0, 1], label: '换一组' } } }]);
    button(el, '走一步（η = 0.5）');
    button(el, '换一组');
  });
});

describe('sliders panel', () => {
  it('gives one slider per element with subscripted names', () => {
    const { g, x } = softmaxGraph();
    const el = panels(container(), g, [{ sliders: x }]);
    expect(text(el.querySelector('h2'))).toBe('x，(5,)');
    expect([...el.querySelectorAll('.vf-hrow b')].map(text)).toEqual(['x₁', 'x₂', 'x₃', 'x₄', 'x₅']);
    const input = el.querySelector<HTMLInputElement>('input[type="range"]')!;
    expect([input.min, input.max, input.step]).toEqual(['-2.5', '2.5', '0.01']);
    expect([...el.querySelectorAll('.vf-sl span')].map(text)).toEqual(['2.00', '1.00', '0.20', '−0.60', '−1.40']);
    slide(el.querySelectorAll<HTMLInputElement>('input[type="range"]')[3], 1.25);
    expect(x.value.get([3])).toBe(1.25);
    expect(text(el.querySelectorAll('.vf-sl span')[3])).toBe('1.25');
  });

  it('shows the focus row of a batch node', () => {
    const { g, x, steps } = batchGraph();
    const el = panels(container(), g, [{ sliders: x, min: -4, max: 4, title: 'logits' }], { steps });
    expect(text(el.querySelector('h2'))).toBe('logits');
    expect([...el.querySelectorAll('.vf-sl span')].map(text)).toEqual(['1.00', '0.00', '−1.00']);
    steps.setFocus(1);
    expect([...el.querySelectorAll('.vf-sl span')].map(text)).toEqual(['0.50', '2.00', '0.00']);
    slide(el.querySelector<HTMLInputElement>('input[type="range"]')!, -3);
    expect(x.value.toNested()).toEqual([
      [1, 0, -1],
      [-3, 2, 0],
    ]);
  });
});

describe('tokens panel', () => {
  it('lists tokens, sliders and columns and sets the target', () => {
    const { g, x, t, p } = softmaxGraph();
    const el = panels(container(), g, [{ tokens: t, sliders: x, min: -6, max: 6, columns: [p, { node: x, grad: true }] }]);
    expect(text(el.querySelector('h2'))).toBe('词表与目标 token');
    expect([...el.querySelectorAll('th')].map(text)).toEqual(['目标', 'x', 'p', '∂ℓ/∂x']);
    const rows = [...el.querySelectorAll('tbody tr')];
    expect(rows.map((r) => text(r.querySelector('.vf-tok')))).toEqual(TOKENS.map((tk, i) => `${i + 1} "${tk}"`));
    expect(rows[0].classList.contains('vf-on')).toBe(true);
    const grads = rows.map((r) => r.querySelectorAll<HTMLElement>('td.vf-n')[1]);
    expect(text(grads[0])).toBe('−0.391');
    expect(grads[0].style.color).toBe('var(--neg)');
    expect(grads[1].style.color).toBe('var(--pos)');
    expect(text(rows[0].querySelectorAll('td.vf-n')[0])).toBe('0.61');
    expect(rows[0].querySelector<HTMLInputElement>('input[type="range"]')!.max).toBe('6');

    const radio = rows[2].querySelector<HTMLInputElement>('input[type="radio"]')!;
    radio.checked = true;
    radio.dispatchEvent(new Event('change', { bubbles: true }));
    expect(t.index()).toBe(2);
    expect(rows[2].classList.contains('vf-on')).toBe(true);
    expect(rows[0].classList.contains('vf-on')).toBe(false);
  });

  it('changes only the focus row of a batch label', () => {
    const { g, t, logp, steps } = batchGraph();
    const el = panels(container(), g, [{ tokens: t, columns: [{ node: logp, title: 'log p' }] }], { steps });
    steps.setFocus(1);
    const rows = [...el.querySelectorAll('tbody tr')];
    expect(rows[1].classList.contains('vf-on')).toBe(true);
    const radio = rows[2].querySelector<HTMLInputElement>('input[type="radio"]')!;
    radio.checked = true;
    radio.dispatchEvent(new Event('change', { bubbles: true }));
    expect(t.value.toNested()).toEqual([0, 2]);
  });
});

describe('matrix panel', () => {
  it('shades cells against the limit fixed at mount and frames the target column', () => {
    const g = graph();
    const W = g.param('W', [2, 3], [
      [1, -0.5, 0],
      [0.25, 2, -2],
    ]);
    const t = g.label('t', [1], { tokens: ['a', ' b', 'c'] });
    g.nll_loss(g.log_softmax(g.sum(W, 0, { keepdim: true }), -1), t);
    const el = panels(container(), g, [{ matrix: W, highlight: t }]);
    expect(text(el.querySelector('h2'))).toBe('W，(2, 3)');
    const grid = el.querySelector<HTMLElement>('.vf-wmat')!;
    expect(grid.style.gridTemplateColumns).toBe('26px repeat(3, minmax(0, 1fr))');
    const headers = [...grid.querySelectorAll('.vf-h')].map(text);
    expect(headers).toEqual(['', 'a', 'b', 'c', '1', '2']);
    const cells = [...grid.children].filter((c) => !c.classList.contains('vf-h')) as HTMLElement[];
    expect(cells.map(text)).toEqual(['1.00', '−0.50', '0.00', '0.25', '2.00', '−2.00']);
    expect(cells[1].style.background).toContain('var(--neg) 24%');
    expect(cells[2].style.background).toContain('var(--pos) 0%');
    expect(cells[4].style.background).toContain('var(--pos) 70%');
    expect(cells.map((c) => c.classList.contains('vf-ct'))).toEqual([false, true, false, false, true, false]);
    W.set([
      [4, 0, 0],
      [0, 0, 0],
    ]);
    expect(cells[0].style.background).toContain('var(--pos) 70%');
    t.set([2]);
    expect(cells.map((c) => c.classList.contains('vf-ct'))).toEqual([false, false, true, false, false, true]);
  });

  it('frames the target row of a square matrix from a function and follows the focus', () => {
    const { g, x, logp, t, steps } = batchGraph();
    const el = panels(container(), g, [{ matrix: (graph, state) => graph.jacobian(logp, x, { row: state.focus }), highlight: t, title: 'J' }], { steps });
    const grid = el.querySelector('.vf-wmat')!;
    const framed = () => [...grid.querySelectorAll('.vf-rt')].length;
    expect(framed()).toBe(3);
    expect(grid.querySelectorAll('.vf-ct')).toHaveLength(0);
    const first = text(grid.querySelector('.vf-rt'));
    steps.setFocus(1);
    expect(framed()).toBe(3);
    expect(text(grid.querySelector('.vf-rt'))).not.toBe(first);
  });

  it('rejects tensors that are not 2-D', () => {
    const { g, x } = softmaxGraph();
    expect(() => mountPanels(container(), g, [{ matrix: x }])).toThrow("matrix panel: node 'x' must be 2-D, got shape (5,)");
  });
});

describe('code and note panels', () => {
  it('marks comments and renders notes', () => {
    const { g } = softmaxGraph();
    const el = panels(container(), g, [
      { code: 'a = 1  # one\nb = 2', title: 'PyTorch', note: '说明' },
      { note: '公式 $x$', title: '注' },
    ]);
    const pre = el.querySelector('pre.vf-code')!;
    expect(pre.textContent).toBe('a = 1  # one\nb = 2');
    expect(text(pre.querySelector('.vf-c'))).toBe('# one');
    const notes = el.querySelectorAll('.vf-note');
    expect(text(notes[0])).toBe('说明');
    expect(notes[1].querySelector('.vf-tex')).not.toBeNull();
    expect([...el.querySelectorAll('h2')].map(text)).toEqual(['PyTorch', '注']);
  });
});

describe('mountSteps', () => {
  function nextFrame() {
    return new Promise((resolve) => requestAnimationFrame(() => resolve(undefined)));
  }

  it('builds the track and switches steps with buttons and keys', async () => {
    const { steps } = softmaxGraph();
    const el = container();
    el.id = 'steps-host';
    mounted.push(mountSteps('#steps-host', steps));
    const groups = [...el.querySelectorAll('.vf-grp')].map(text);
    expect(groups).toEqual(['前向', '反向']);
    const pills = [...el.querySelectorAll<HTMLButtonElement>('.vf-st')];
    expect(pills.map(text)).toEqual(steps.list.map((s) => s.label));
    expect(pills[0].getAttribute('aria-current')).toBe('step');
    const prev = el.querySelector<HTMLButtonElement>('[aria-label="上一步"]')!;
    const next = el.querySelector<HTMLButtonElement>('[aria-label="下一步"]')!;
    expect(prev.disabled).toBe(true);
    await Promise.resolve();
    expect(text(el.querySelector('h3'))).toBe('总览');
    expect(el.querySelector('.vf-caption-body .vf-tex')).not.toBeNull();

    next.click();
    expect(steps.current).toBe(1);
    expect(pills[1].getAttribute('aria-current')).toBe('step');
    expect(pills[0].hasAttribute('aria-current')).toBe(false);
    await nextFrame();
    expect(text(el.querySelector('h3'))).toBe('logits');
    expect(text(el.querySelector('.vf-caption-body'))).toBe('x = 2.00');

    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowRight' }));
    expect(steps.current).toBe(2);
    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowLeft' }));
    expect(steps.current).toBe(1);

    const input = document.createElement('input');
    document.body.append(input);
    input.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowRight', bubbles: true }));
    expect(steps.current).toBe(1);

    pills[pills.length - 1].click();
    expect(next.disabled).toBe(true);
  });

  it('redraws the caption once per frame after graph changes', async () => {
    const { steps, x } = softmaxGraph();
    const el = container();
    mounted.push(mountSteps(el, steps));
    steps.go(1);
    await nextFrame();
    const spy = vi.spyOn(steps, 'text');
    x.set([1, 1, 1, 1, 1]);
    x.set([3, 1, 1, 1, 1]);
    steps.go(1);
    expect(spy).not.toHaveBeenCalled();
    await nextFrame();
    expect(spy).toHaveBeenCalledTimes(1);
    expect(text(el.querySelector('.vf-caption-body'))).toBe('x = 3.00');
  });

  it('removes its DOM and key listener on dispose', () => {
    const { steps } = softmaxGraph();
    const el = container();
    mountSteps(el, steps).dispose();
    expect(el.children).toHaveLength(0);
    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowRight' }));
    expect(steps.current).toBe(0);
  });

  it('rejects selectors that match nothing', () => {
    const { steps } = softmaxGraph();
    expect(() => mountSteps('#missing', steps)).toThrow("mountSteps: no element matches '#missing'");
  });
});
