// @vitest-environment jsdom
import { afterEach, describe, expect, it, vi } from 'vitest';
import { createStepList } from '../../src/steps';
import { mountStepPanel, mountSteps, mountTimeline, type Mounted, type TimelineSpec } from '../../src/ui';
import { shown } from './shown';

const mounted: Mounted[] = [];

afterEach(() => {
  mounted.splice(0).forEach((m) => m.dispose());
  document.body.replaceChildren();
});

function container(): HTMLElement {
  const el = document.createElement('div');
  document.body.append(el);
  return el;
}

const text = (el: Element | null) => (el === null ? '' : shown(el));

function threeSteps() {
  return createStepList([
    { title: '调度第 $1$ 批', label: '调度', text: 'CPU 准备 $N=1$' },
    { title: 'GPU 运行', id: 'run' },
    { title: '采样', segment: '收尾', sub: '取 token' },
  ]);
}

describe('createStepList', () => {
  it('builds StepInfo from the items', () => {
    const steps = threeSteps();
    expect(steps.list.map((s) => [s.index, s.id, s.dir, s.kind, s.label, s.segment, s.sub, s.nodes.length])).toEqual([
      [0, 's:0', 'forward', 'op', '调度', '调度', undefined, 0],
      [1, 'run', 'forward', 'op', 'GPU 运行', 'GPU 运行', undefined, 0],
      [2, 's:2', 'forward', 'op', '采样', '收尾', '取 token', 0],
    ]);
    expect(steps.graph.nodes).toEqual([]);
    expect(steps.visible).toEqual([]);
    expect(steps.modules).toEqual([]);
    expect(steps.folded('any')).toBe(false);
    expect(steps.state().revealed).toBeTypeOf('function');
    expect(() => createStepList([])).toThrow(/at least one step/);
    expect(() => createStepList([{ title: 'a', id: 'x' }, { title: 'b', id: 'x' }])).toThrow(/'x' is used more than once/);
  });

  it('moves between steps, notifies subscribers and returns text', () => {
    const steps = threeSteps();
    const seen: number[] = [];
    const off = steps.subscribe(() => seen.push(steps.current));
    expect(steps.text()).toEqual({ title: '调度第 $1$ 批', body: 'CPU 准备 $N=1$' });
    steps.next();
    steps.next();
    steps.next();
    steps.prev();
    steps.go(0);
    steps.go(0);
    expect(seen).toEqual([1, 2, 1, 0]);
    expect(steps.text(1)).toEqual({ title: 'GPU 运行', body: '' });
    expect(() => steps.text(5)).toThrow(/does not exist/);
    off();
    steps.next();
    expect(seen).toEqual([1, 2, 1, 0]);
    expect(steps.state().index).toBe(1);
  });

  it('works with mountSteps', () => {
    const steps = threeSteps();
    const el = container();
    mounted.push(mountSteps(el, steps));
    expect([...el.querySelectorAll('.vf-st')].map(text)).toEqual(['调度', 'GPU 运行', '采样']);
    expect(text(el.querySelector('.vf-caption h3'))).toBe('调度第 $1$ 批');
    el.querySelectorAll<HTMLButtonElement>('.vf-st')[2].click();
    expect(steps.current).toBe(2);
    expect(el.querySelector('[aria-current="step"]')!.textContent).toBe('采样');
    el.querySelector<HTMLButtonElement>('[aria-label="上一步"]')!.click();
    expect(steps.current).toBe(1);
  });
});

describe('mountStepPanel', () => {
  it('renders on mount and after every step change until disposed', () => {
    const steps = threeSteps();
    const el = container();
    const render = vi.fn((root: HTMLElement, step: { index: number; info: { id: string } }) => {
      root.textContent = `${step.index}:${step.info.id}`;
    });
    const panel = mountStepPanel(el, steps, { title: '设备上的批次 $N$', render });
    expect(render).toHaveBeenCalledTimes(1);
    expect(text(el.querySelector('.vf-card h2'))).toBe('设备上的批次 $N$');
    expect(el.querySelector('.vf-step-panel-body')!.textContent).toBe('0:s:0');
    steps.go(1);
    expect(render).toHaveBeenCalledTimes(2);
    expect(render.mock.calls[1][1].index).toBe(1);
    expect(el.querySelector('.vf-step-panel-body')!.textContent).toBe('1:run');
    panel.dispose();
    expect(el.childElementCount).toBe(0);
    steps.go(2);
    expect(render).toHaveBeenCalledTimes(2);
  });

  it('omits the heading without a title', () => {
    const el = container();
    mounted.push(mountStepPanel(el, threeSteps(), { render: () => {} }));
    expect(el.querySelector('h2')).toBeNull();
  });
});

function overlapSpec(): TimelineSpec {
  return {
    title: '重叠',
    unit: 'ms',
    lanes: [
      { id: 'cpu', label: 'CPU' },
      { id: 'gpu', label: 'GPU' },
    ],
    kinds: { schedule: { label: '调度' }, forward: { label: '前向' }, idle: { label: '空闲' } },
    items: [
      { lane: 'cpu', start: 0, end: 2, label: '调度 1', kind: 'schedule', step: 0 },
      { lane: 'gpu', start: 2, end: 10, label: '前向 1', kind: 'forward', step: 'run' },
      { lane: 'cpu', start: 2, end: 4, label: '调度 2', kind: 'schedule', step: 1 },
      { lane: 'cpu', start: 4, end: 10 },
    ],
    times: { 's:2': [8, 12] },
  };
}

const items = (el: HTMLElement) => [...el.querySelectorAll<HTMLElement>('.vf-gantt-item')];
const box = (node: HTMLElement) => [parseFloat(node.style.left), parseFloat(node.style.width)];

describe('mountTimeline', () => {
  it('places items by percentage of the time range and labels the axis', () => {
    const el = container();
    mounted.push(mountTimeline(el, threeSteps(), overlapSpec()));
    expect(items(el).map(box)).toEqual([
      [0, 20],
      [20, 80],
      [20, 20],
      [40, 60],
    ]);
    expect(items(el).map((n) => n.style.top)).toEqual(['calc(var(--vf-gantt-row) * 0)', 'calc(var(--vf-gantt-row) * 1)', 'calc(var(--vf-gantt-row) * 0)', 'calc(var(--vf-gantt-row) * 0)']);
    expect(items(el).map((n) => n.tagName)).toEqual(['BUTTON', 'BUTTON', 'BUTTON', 'SPAN']);
    expect(items(el)[1].title).toBe('前向 1：2 到 10 ms');
    expect(items(el)[3].title).toBe('4 到 10 ms');
    expect([...el.querySelectorAll('.vf-gantt-tick')].map(text)).toEqual(['0 ms', '2 ms', '4 ms', '6 ms', '8 ms', '10 ms']);
    expect([...el.querySelectorAll('.vf-gantt-lane-label')].map(text)).toEqual(['CPU', 'GPU']);
    expect(text(el.querySelector('.vf-gantt h2'))).toBe('重叠');
  });

  it('draws the cursor from times or from the items of the current step', () => {
    const steps = threeSteps();
    const el = container();
    mounted.push(mountTimeline(el, steps, overlapSpec()));
    const cursor = () => el.querySelector<HTMLElement>('.vf-gantt-cursor');
    expect(box(cursor()!)).toEqual([0, 20]);
    expect(items(el).map((n) => [n.classList.contains('vf-on'), n.classList.contains('vf-dim')])).toEqual([
      [true, false],
      [false, true],
      [false, true],
      [false, true],
    ]);
    steps.go(1);
    expect(box(cursor()!)).toEqual([20, 80]);
    expect(items(el).filter((n) => n.classList.contains('vf-on')).map((n) => n.textContent)).toEqual(['前向 1', '调度 2']);
    steps.go(2);
    expect(box(cursor()!)).toEqual([80, 20]);
    expect(items(el).some((n) => n.classList.contains('vf-on') || n.classList.contains('vf-dim'))).toBe(false);
    const spec = overlapSpec();
    delete spec.times;
    const other = container();
    mounted.push(mountTimeline(other, steps, spec));
    expect(other.querySelector('.vf-gantt-cursor')).toBeNull();
  });

  it('jumps to the step of a clicked item', () => {
    const steps = threeSteps();
    const el = container();
    mounted.push(mountTimeline(el, steps, overlapSpec()));
    items(el)[1].click();
    expect(steps.current).toBe(1);
    expect(items(el)[1].getAttribute('aria-current')).toBe('step');
    items(el)[0].click();
    expect(steps.current).toBe(0);
  });

  it('asks a function spec again on every step change', () => {
    const steps = threeSteps();
    const el = container();
    const source = vi.fn(({ index }: { index: number }): TimelineSpec => ({
      lanes: [{ id: 'gpu', label: 'GPU' }],
      items: Array.from({ length: index + 1 }, (_, k) => ({ lane: 'gpu', start: k, end: k + 1, label: `批 ${k + 1}`, step: k })),
    }));
    mounted.push(mountTimeline(el, steps, source));
    expect(items(el).map((n) => n.textContent)).toEqual(['批 1']);
    expect(items(el).map(box)).toEqual([[0, 100]]);
    steps.go(2);
    expect(source).toHaveBeenLastCalledWith({ index: 2, info: steps.list[2] });
    expect(items(el).map((n) => n.textContent)).toEqual(['批 1', '批 2', '批 3']);
    expect(box(items(el)[2])).toEqual([100 * (2 / 3), 100 / 3].map((v) => parseFloat(v.toFixed(4))));
  });

  it('lists only the kinds that items use and colors them from the palette', () => {
    const el = container();
    mounted.push(mountTimeline(el, threeSteps(), { ...overlapSpec(), kinds: { idle: { label: '空闲' }, schedule: { label: '调度' }, forward: { label: '前向', color: 'red' } } }));
    const legend = [...el.querySelectorAll<HTMLElement>('.vf-gantt-legend li')];
    expect(legend.map(text)).toEqual(['调度', '前向']);
    expect(legend.map((li) => li.style.getPropertyValue('--vf-kind'))).toEqual(['var(--vf-series-2)', 'red']);
    expect(items(el)[3].style.getPropertyValue('--vf-kind')).toBe('var(--muted)');
  });

  it('rejects items on unknown lanes and steps that do not exist', () => {
    const base = overlapSpec();
    expect(() => mountTimeline(container(), threeSteps(), { ...base, items: [{ lane: 'tpu', start: 0, end: 1 }] })).toThrow(/lane 'tpu'/);
    expect(() => mountTimeline(container(), threeSteps(), { ...base, items: [{ lane: 'cpu', start: 0, end: 1, step: 'nope' }] })).toThrow(/step 'nope'/);
    expect(() => mountTimeline(container(), threeSteps(), { ...base, items: [{ lane: 'cpu', start: 2, end: 1 }] })).toThrow(/before its start/);
  });
});
