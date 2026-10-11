import { For, Show, createEffect, createMemo, createSignal, createUniqueId, onCleanup, type Accessor, type JSX } from 'solid-js';
import { render } from 'solid-js/web';
import { Node, type Graph } from '../autograd';
import { renderText } from '../formula';
import { allVisible, compileTemplate, type StepState, type Steps } from '../steps';
import { formatNumber, formatShape, type Tensor } from '../tensor';
import { resolveElement, trackGraph, trackSteps } from './mount';
import type { Mounted } from './mounted';

export interface LossOptions {
  sgd?: { params: Node[]; lr: number; label?: string };
  random?: { nodes: Node[]; range: [number, number]; label?: string };
  reset?: boolean;
  text?: string;
}

export type MatrixSource = Node | ((g: Graph, state: StepState) => Tensor);
export type MatrixHighlight = Node | { node: Node; axis: 'row' | 'col' };
export type TokenColumn = Node | { node: Node; grad?: boolean; title?: string };

export type PanelItem =
  | 'loss'
  | { loss: LossOptions }
  | { sliders: Node; min?: number; max?: number; step?: number; title?: string }
  | { tokens: Node; columns?: TokenColumn[]; title?: string; sliders?: Node; min?: number; max?: number }
  | { matrix: MatrixSource; grad?: boolean; title?: string; highlight?: MatrixHighlight; note?: string }
  | { code: string; title?: string; note?: string }
  | { note: string; title?: string };

const SLIDER_MIN = -2.5;
const SLIDER_MAX = 2.5;
const SLIDER_STEP = 0.01;
const ZERO_CELL = 1e-4;
const CELL_DIGITS = 2;
const MIN_MATRIX_LIMIT = 1e-6;
const SUBSCRIPT_DIGITS = '₀₁₂₃₄₅₆₇₈₉';

interface Context {
  g: Graph;
  version: Accessor<number>;
  focus: Accessor<number>;
  state: Accessor<StepState>;
}

const subscript = (n: number) => String(n).replace(/\d/g, (d) => SUBSCRIPT_DIGITS[Number(d)]);

const display = (v: number, digits: number) => formatNumber(v, { digits });

const signColor = (v: number) => `var(${v >= 0 ? '--pos' : '--neg'})`;

function rowElement(node: Node, focus: number, i: number): number[] {
  const shape = node.shape;
  if (shape.length === 0) return [];
  if (shape.length === 1) return [i];
  if (shape.length === 2) return [Math.min(focus, shape[0] - 1), i];
  throw new Error(`ui: node '${node.name}' with shape ${formatShape(shape)} must have at most 2 dimensions`);
}

const rowLength = (node: Node) => (node.shape.length === 0 ? 1 : node.shape[node.shape.length - 1]);

function setElement(node: Node, index: number[], value: number): void {
  const next = node.value.clone();
  next.set(index, value);
  node.set(Array.from(next.data));
}

function targetIndex(label: Node, focus: number): number[] {
  const shape = label.shape;
  if (shape.length === 0) return [];
  if (shape.length === 1) return [Math.min(focus, shape[0] - 1)];
  if (shape.length === 2 && shape[1] === 1) return [Math.min(focus, shape[0] - 1), 0];
  throw new Error(`ui: label '${label.name}' with shape ${formatShape(shape)} must have one target per row`);
}

function classCount(label: Node): number {
  const classes = (label.attrs.classes as number | undefined) ?? label.tokens?.length;
  if (classes === undefined) throw new Error(`ui: label '${label.name}' needs tokens or classes`);
  return classes;
}

function gradOf(node: Node): Tensor {
  const grad = node.grad;
  if (grad === null) throw new Error(`ui: label '${node.name}' has no gradient`);
  return grad;
}

function Card(props: { title?: string; children: JSX.Element }) {
  return (
    <div class="vf-card">
      <Show when={props.title}>
        <h2>{props.title}</h2>
      </Show>
      {props.children}
    </div>
  );
}

function TemplateText(props: { ctx: Context; text: string; class: string }) {
  const evaluate = compileTemplate(props.ctx.g, props.text);
  let target!: HTMLDivElement;
  createEffect(() => {
    props.ctx.version();
    renderText(evaluate(props.ctx.focus()), target);
  });
  return <div class={props.class} ref={(el) => (target = el)} />;
}

function Spark(props: { history: number[] }) {
  const points = createMemo(() => {
    const history = props.history;
    const max = Math.max(...history) || 1;
    return history.map((v, k) => [4 + (k * 112) / (history.length - 1), 37 - (v / max) * 31]);
  });
  const path = () => points().map(([x, y], k) => `${k ? 'L' : 'M'}${x.toFixed(1)} ${y.toFixed(1)}`).join('');
  const lastPoint = () => points()[points().length - 1];
  return (
    <svg class="vf-spark" viewBox="0 0 120 42" aria-label="损失变化">
      <line x1="4" x2="116" y1="37" y2="37" stroke="var(--line-strong)" />
      <path d={path()} fill="none" stroke="var(--pos)" stroke-width="1.6" />
      <circle cx={lastPoint()[0]} cy={lastPoint()[1]} r="2.6" fill="var(--pos)" />
    </svg>
  );
}

function LossPanel(props: { ctx: Context; options: LossOptions }) {
  const { g, version } = props.ctx;
  const options = props.options;
  const lossValue = () => g.loss().value.data[0];
  const loss = createMemo(() => {
    version();
    return lossValue();
  });
  const [history, setHistory] = createSignal<number[]>([]);
  let stepping = false;
  onCleanup(
    g.subscribe(() => {
      if (!stepping) setHistory([]);
    }),
  );

  const sgdStep = (sgd: NonNullable<LossOptions['sgd']>) => {
    const before = lossValue();
    const updates = sgd.params.map((p) => {
      const grad = gradOf(p);
      return { p, values: Array.from(p.value.data, (v, i) => v - sgd.lr * grad.data[i]) };
    });
    stepping = true;
    try {
      g.batch(() => updates.forEach(({ p, values }) => p.set(values)));
    } finally {
      stepping = false;
    }
    setHistory((h) => [...(h.length > 0 ? h : [before]), lossValue()]);
  };

  const randomize = (random: NonNullable<LossOptions['random']>) => {
    const [low, high] = random.range;
    g.batch(() => random.nodes.forEach((n) => n.set(Array.from(n.value.data, () => Math.round((low + Math.random() * (high - low)) * 100) / 100))));
  };

  for (const [field, nodes] of [['sgd.params', options.sgd?.params], ['random.nodes', options.random?.nodes]] as const) {
    for (const node of nodes ?? []) {
      if (node.kind === 'buffer') throw new Error(`loss panel: buffer '${node.name}' is a constant without gradient and cannot be in ${field}`);
    }
  }
  const names = (nodes: Node[]) => nodes.map((n) => n.name).join('、');
  const hasButtons = options.sgd !== undefined || options.random !== undefined || options.reset === true;

  return (
    <Card>
      <div class="vf-readout">
        <div>
          <div class="vf-big">
            <i>ℓ =</i>
            <span class="vf-loss-value">{display(loss(), 3)}</span>
          </div>
          <Show when={options.text}>{(text) => <TemplateText ctx={props.ctx} class="vf-sub" text={text()} />}</Show>
        </div>
        <Show when={history().length >= 2}>
          <Spark history={history()} />
        </Show>
      </div>
      <Show when={hasButtons}>
        <div class="vf-btns">
          <Show when={options.sgd}>
            {(sgd) => (
              <button type="button" class="vf-btn vf-pri" onClick={() => sgdStep(sgd())}>
                {`${sgd().label ?? `更新一步 ${names(sgd().params)}`}（η = ${sgd().lr}）`}
              </button>
            )}
          </Show>
          <Show when={options.random}>
            {(random) => (
              <button type="button" class="vf-btn" onClick={() => randomize(random())}>
                {random().label ?? `随机 ${names(random().nodes)}`}
              </button>
            )}
          </Show>
          <Show when={options.reset}>
            <button type="button" class="vf-btn" onClick={() => g.reset()}>
              重置
            </button>
          </Show>
        </div>
      </Show>
    </Card>
  );
}

function Slider(props: { ctx: Context; node: Node; i: number; label: string; min: number; max: number; step: number }) {
  const value = createMemo(() => {
    props.ctx.version();
    return props.node.value.get(rowElement(props.node, props.ctx.focus(), props.i));
  });
  return (
    <div class="vf-sl">
      <input
        type="range"
        min={props.min}
        max={props.max}
        step={props.step}
        value={value()}
        aria-label={props.label}
        style={{ '--vf-fill': `${(Math.min(Math.max((value() - props.min) / (props.max - props.min), 0), 1) * 100).toFixed(1)}%` }}
        onInput={(e) => setElement(props.node, rowElement(props.node, props.ctx.focus(), props.i), Number(e.currentTarget.value))}
      />
      <span>{display(value(), 2)}</span>
    </div>
  );
}

function SlidersPanel(props: { ctx: Context; item: Extract<PanelItem, { sliders: Node }> }) {
  const { sliders: node, min = SLIDER_MIN, max = SLIDER_MAX, step = SLIDER_STEP } = props.item;
  const indices = Array.from({ length: rowLength(node) }, (_, i) => i);
  return (
    <Card title={props.item.title ?? `${node.name}，${formatShape(node.shape)}`}>
      <For each={indices}>
        {(i) => {
          const label = `${node.name}${subscript(i + 1)}`;
          return (
            <div class="vf-hrow">
              <b>{label}</b>
              <Slider ctx={props.ctx} node={node} i={i} label={label} min={min} max={max} step={step} />
            </div>
          );
        }}
      </For>
    </Card>
  );
}

function TokensPanel(props: { ctx: Context; item: Extract<PanelItem, { tokens: Node }> }) {
  const { ctx, item } = props;
  const label = item.tokens;
  const classes = Array.from({ length: classCount(label) }, (_, i) => i);
  const group = createUniqueId();
  const columns = (item.columns ?? []).map((c) => {
    const spec = c instanceof Node ? { node: c } : c;
    const grad = spec.grad ?? false;
    return { node: spec.node, grad, title: spec.title ?? (grad ? `∂ℓ/∂${spec.node.name}` : spec.node.name) };
  });
  const target = createMemo(() => {
    ctx.version();
    return Math.round(label.value.get(targetIndex(label, ctx.focus())));
  });
  const choose = (i: number) => setElement(label, targetIndex(label, ctx.focus()), i);
  const tokenText = (i: number) => label.tokens?.[i].replace(/^ +/, (spaces) => '␣'.repeat(spaces.length)) ?? '';

  return (
    <Card title={item.title ?? '词表与目标 token'}>
      <div class="vf-scroll">
      <table class="vf-cls">
        <thead>
          <tr>
            <th>目标</th>
            <Show when={item.sliders}>{(node) => <th>{node().name}</th>}</Show>
            <For each={columns}>{(c) => <th class="vf-n">{c.title}</th>}</For>
          </tr>
        </thead>
        <tbody>
          <For each={classes}>
            {(i) => (
              <tr classList={{ 'vf-on': target() === i }}>
                <td>
                  <label class="vf-tg">
                    <input type="radio" name={group} checked={target() === i} onChange={() => choose(i)} />
                    <span class="vf-tok-id">{i + 1}</span>
                    <span class="vf-tok">{tokenText(i)}</span>
                  </label>
                </td>
                <Show when={item.sliders}>
                  {(node) => (
                    <td class="vf-slcell">
                      <Slider
                        ctx={ctx}
                        node={node()}
                        i={i}
                        label={`${node().name}${subscript(i + 1)}`}
                        min={item.min ?? SLIDER_MIN}
                        max={item.max ?? SLIDER_MAX}
                        step={SLIDER_STEP}
                      />
                    </td>
                  )}
                </Show>
                <For each={columns}>
                  {(c) => {
                    const value = createMemo(() => {
                      ctx.version();
                      const tensor = c.grad ? gradOf(c.node) : c.node.value;
                      return tensor.get(rowElement(c.node, ctx.focus(), i));
                    });
                    return (
                      <td class="vf-n" style={c.grad ? { color: signColor(value()) } : undefined}>
                        {display(value(), 3)}
                      </td>
                    );
                  }}
                </For>
              </tr>
            )}
          </For>
        </tbody>
      </table>
      </div>
    </Card>
  );
}

function MatrixPanel(props: { ctx: Context; item: Extract<PanelItem, { matrix: MatrixSource }> }) {
  const { ctx, item } = props;
  const source = item.matrix;
  const read = (): Tensor => {
    const tensor = source instanceof Node ? (item.grad ? gradOf(source) : source.value) : source(ctx.g, ctx.state());
    if (tensor.ndim !== 2) {
      const what = source instanceof Node ? `node '${source.name}'` : 'matrix function result';
      throw new Error(`matrix panel: ${what} must be 2-D, got shape ${formatShape(tensor.shape)}`);
    }
    return tensor;
  };
  const initial = read();
  const limit = Math.max(MIN_MATRIX_LIMIT, ...Array.from(initial.data, Math.abs));
  const [rows, cols] = initial.shape;
  const tensor = createMemo(() => {
    ctx.version();
    return read();
  });
  const highlight = item.highlight instanceof Node ? item.highlight : item.highlight?.node;
  const frameRow = !(item.highlight instanceof Node) && item.highlight?.axis === 'row';
  const tokenHeaders = highlight?.tokens !== undefined && classCount(highlight) === cols ? highlight.tokens : undefined;
  const frameArea = (t: number) =>
    frameRow ? { 'grid-row': `${t + 2}`, 'grid-column': `2 / span ${cols}` } : { 'grid-row': `2 / span ${rows}`, 'grid-column': `${t + 2}` };
  const target = createMemo(() => {
    ctx.version();
    return highlight === undefined ? undefined : Math.round(highlight.value.get(targetIndex(highlight, ctx.focus())));
  });
  const shade = (v: number) => {
    if (Math.abs(v) < ZERO_CELL || Number(display(Math.abs(v), CELL_DIGITS)) === 0) return 'transparent';
    const percent = 8 + Math.min(Math.abs(v) / limit, 1) * 62;
    return `color-mix(in srgb, ${signColor(v)} ${percent.toFixed(0)}%, transparent)`;
  };
  const rowIds = Array.from({ length: rows }, (_, r) => r);
  const colIds = Array.from({ length: cols }, (_, c) => c);
  const title =
    item.title ?? (source instanceof Node ? `${item.grad ? `∂ℓ/∂${source.name}` : source.name}，${formatShape(source.shape)}` : undefined);

  return (
    <Card title={title}>
      <div class="vf-wmat" style={{ 'grid-template-columns': `26px repeat(${cols}, minmax(0, 1fr))` }}>
        <div class="vf-h" style={{ 'grid-row': '1', 'grid-column': '1' }} />
        <For each={colIds}>
          {(c) => (
            <div class="vf-h" title={tokenHeaders?.[c]} style={{ 'grid-row': '1', 'grid-column': `${c + 2}` }}>
              {tokenHeaders === undefined ? c + 1 : tokenHeaders[c].trim()}
            </div>
          )}
        </For>
        <For each={rowIds}>
          {(r) => (
            <>
              <div class="vf-h vf-rh" style={{ 'grid-row': `${r + 2}`, 'grid-column': '1' }}>
                {r + 1}
              </div>
              <For each={colIds}>
                {(c) => {
                  const value = () => tensor().get([r, c]);
                  return (
                    <div style={{ background: shade(value()), 'grid-row': `${r + 2}`, 'grid-column': `${c + 2}` }}>
                      {display(value(), CELL_DIGITS)}
                    </div>
                  );
                }}
              </For>
            </>
          )}
        </For>
        <Show when={target() !== undefined}>
          <div class={frameRow ? 'vf-frame vf-rt' : 'vf-frame vf-ct'} style={frameArea(target() ?? 0)} />
        </Show>
      </div>
      <Show when={item.note}>{(note) => <TemplateText ctx={ctx} class="vf-note" text={note()} />}</Show>
    </Card>
  );
}

function CodePanel(props: { ctx: Context; item: Extract<PanelItem, { code: string }> }) {
  const lines = props.item.code.split('\n').map((line) => {
    const hash = line.indexOf('#');
    return hash < 0 ? { code: line, comment: '' } : { code: line.slice(0, hash), comment: line.slice(hash) };
  });
  return (
    <Card title={props.item.title}>
      <pre class="vf-code">
        <For each={lines}>
          {(line, k) => (
            <>
              {line.code}
              <Show when={line.comment}>
                <span class="vf-c">{line.comment}</span>
              </Show>
              {k() < lines.length - 1 ? '\n' : ''}
            </>
          )}
        </For>
      </pre>
      <Show when={props.item.note}>{(note) => <TemplateText ctx={props.ctx} class="vf-note" text={note()} />}</Show>
    </Card>
  );
}

function Panel(props: { ctx: Context; item: PanelItem }) {
  const item = props.item;
  if (item === 'loss') return <LossPanel ctx={props.ctx} options={{}} />;
  if ('loss' in item) return <LossPanel ctx={props.ctx} options={item.loss} />;
  if ('sliders' in item && !('tokens' in item)) return <SlidersPanel ctx={props.ctx} item={item} />;
  if ('tokens' in item) return <TokensPanel ctx={props.ctx} item={item} />;
  if ('matrix' in item) return <MatrixPanel ctx={props.ctx} item={item} />;
  if ('code' in item) return <CodePanel ctx={props.ctx} item={item} />;
  return (
    <Card title={item.title}>
      <TemplateText ctx={props.ctx} class="vf-note" text={item.note} />
    </Card>
  );
}

function PanelsView(props: { g: Graph; items: PanelItem[]; steps: Steps | undefined }) {
  const { g, steps } = props;
  const version = trackGraph(g);
  const stepsVersion = trackSteps(steps);
  const ctx: Context = {
    g,
    version,
    focus: () => {
      stepsVersion();
      return steps?.focus ?? 0;
    },
    state: () => {
      stepsVersion();
      return steps?.state() ?? allVisible(g);
    },
  };
  return (
    <div class="vf-panels">
      <For each={props.items}>{(item) => <Panel ctx={ctx} item={item} />}</For>
    </div>
  );
}

export function mountPanels(el: string | HTMLElement, g: Graph, items: PanelItem[], options: { steps?: Steps } = {}): Mounted {
  const target = resolveElement(el, 'mountPanels');
  return { dispose: render(() => <PanelsView g={g} items={items} steps={options.steps} />, target) };
}
