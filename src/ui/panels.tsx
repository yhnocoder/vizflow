import { For, Show, createEffect, createMemo, createSignal, createUniqueId, onCleanup, type Accessor, type JSX } from 'solid-js';
import { render } from 'solid-js/web';
import { Node, type Graph } from '../autograd';
import { renderText } from '../formula';
import { type InferSpec, allVisible, compileTemplate, type StepState, type Steps } from '../steps';
import { formatNumber, formatShape, type Tensor } from '../tensor';
import { resolveElement, trackGraph, trackSteps } from './mount';
import type { Mounted } from './mounted';
import { CostPanel, type CostPanelOptions } from './cost';
import { Rich, indexed, plain } from './rich';
import { KvBlocksPanel } from './kv-blocks';

export interface SgdOptions {
  params: Node[];
  lr: number;
  label?: string;
}

export interface LossOptions {
  sgd?: SgdOptions;
  random?: { nodes: Node[]; range: [number, number]; label?: string };
  reset?: boolean;
  text?: string;
  parts?: Node;
  hint?: string;
}

export interface PageSequence {
  readonly tokens: string[];
  readonly length: number;
  get(): number[];
  set(ids: number[]): void;
}

export interface SequenceOptions {
  sequence: true;
  labels?: string[];
  sgd?: SgdOptions;
  random?: boolean | { label?: string };
  reset?: boolean;
  title?: string;
  note?: string;
}

export interface MatrixEntry {
  matrix: MatrixSource;
  grad?: boolean;
  caption?: string;
  limit?: number;
}

export interface KvBlocksRequest {
  id: string;
  prompt: string[];
  arrive: number;
}

export interface KvBlocksOptions {
  infer: InferSpec;
  blockSize: number;
  numBlocks: number;
  others: KvBlocksRequest[];
}

export type MatrixSource = Node | ((g: Graph, state: StepState) => Tensor);
export type MatrixHighlight = Node | { node: Node; axis?: 'row' | 'col'; all?: boolean };
export type TokenColumn = Node | { node: Node; grad?: boolean; title?: string };

export type PanelItem =
  | 'loss'
  | { loss: LossOptions }
  | 'cost'
  | { cost: CostPanelOptions }
  | { sliders: Node; min?: number; max?: number; step?: number; title?: string }
  | { tokens: Node; columns?: TokenColumn[]; title?: string; sliders?: Node; min?: number; max?: number }
  | { matrix: MatrixSource; grad?: boolean; title?: string; highlight?: MatrixHighlight; note?: string; caption?: string; limit?: number; rows?: string }
  | { matrices: MatrixEntry[]; title?: string; highlight?: MatrixHighlight; note?: string; rows?: string }
  | { attention: Node[]; ids: Node; extra_keys?: string[]; title?: string; note?: string }
  | SequenceOptions
  | { code: string; title?: string; note?: string }
  | { generation: InferSpec; title?: string; note?: string }
  | { kvBlocks: KvBlocksOptions; title?: string; note?: string }
  | { note: string; title?: string };

const SLIDER_MIN = -2.5;
const SLIDER_MAX = 2.5;
const SLIDER_STEP = 0.01;
const ZERO_CELL = 1e-4;
const CELL_DIGITS = 2;
const MIN_MATRIX_LIMIT = 1e-6;
const SUBSCRIPT_DIGITS = '₀₁₂₃₄₅₆₇₈₉';
const TINT_ZERO = 1e-9;
const ATTENTION_LIMIT = 1;
const ATTENTION_SIDE_BY_SIDE = 4;
const PROB_DIGITS = 2;
const CACHE_KIND_TEXT = { rows: '每个 token 一行，随长度增长', window: '只留最近几行，大小固定', state: '只留一个状态，大小固定' } as const;

interface Context {
  g: Graph;
  version: Accessor<number>;
  focus: Accessor<number>;
  state: Accessor<StepState>;
  history: Accessor<number[]>;
  sgdStep(sgd: SgdOptions): void;
  sequence: PageSequence | undefined;
  steps: Steps | undefined;
  stepsVersion: Accessor<number>;
}

const subscript = (n: number) => String(n).replace(/\d/g, (d) => SUBSCRIPT_DIGITS[Number(d)]);

const display = (v: number, digits: number) => formatNumber(v, { digits });

const signColor = (v: number) => `var(${v >= 0 ? '--pos' : '--neg'})`;

const tint = (v: number, limit: number) => {
  const percent = Math.abs(v) < TINT_ZERO ? 0 : 8 + Math.min(Math.abs(v) / limit, 1) * 62;
  return `color-mix(in srgb, ${signColor(v)} ${percent.toFixed(0)}%, transparent)`;
};

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
        <h2>
          <Rich text={props.title ?? ''} />
        </h2>
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

function Spark(props: { history: number[]; hint?: string }) {
  const points = createMemo(() => {
    const history = props.history;
    const max = Math.max(...history) || 1;
    return history.map((v, k) => [4 + (k * 112) / (history.length - 1), 37 - (v / max) * 31]);
  });
  const path = () => points().map(([x, y], k) => `${k ? 'L' : 'M'}${x.toFixed(1)} ${y.toFixed(1)}`).join('');
  const lastPoint = () => points()[points().length - 1];
  return (
    <svg class="vf-spark" viewBox="0 0 120 42" aria-label="损失变化">
      <Show
        when={props.history.length >= 2}
        fallback={
          <foreignObject x="0" y="17" width="120" height="14">
            <div class="vf-fo vf-spark-hint">
              <Rich text={props.hint ?? ''} />
            </div>
          </foreignObject>
        }
      >
        <line x1="4" x2="116" y1="37" y2="37" stroke="var(--line-strong)" />
        <path d={path()} fill="none" stroke="var(--pos)" stroke-width="1.6" />
        <circle cx={lastPoint()[0]} cy={lastPoint()[1]} r="2.6" fill="var(--pos)" />
      </Show>
    </svg>
  );
}

function checkTrainable(nodes: readonly Node[], field: string, panel: string): void {
  for (const node of nodes) {
    if (node.kind === 'buffer') throw new Error(`${panel}: buffer '${node.name}' is a constant without gradient and cannot be in ${field}`);
  }
}

const names = (nodes: Node[]) => nodes.map((n) => `$${n.tex}$`).join('、');

const sgdLabel = (sgd: SgdOptions) => `${sgd.label ?? `更新一步 ${names(sgd.params)}`}（$\\eta = ${sgd.lr}$）`;

function LossPanel(props: { ctx: Context; options: LossOptions }) {
  const { g, version } = props.ctx;
  const options = props.options;
  const loss = createMemo(() => {
    version();
    return g.loss().value.data[0];
  });
  const parts = createMemo(() => {
    version();
    return options.parts === undefined ? [] : Array.from(options.parts.value.data);
  });

  const randomize = (random: NonNullable<LossOptions['random']>) => {
    const [low, high] = random.range;
    g.batch(() => random.nodes.forEach((n) => n.set(Array.from(n.value.data, () => Math.round((low + Math.random() * (high - low)) * 100) / 100))));
  };

  checkTrainable(options.sgd?.params ?? [], 'sgd.params', 'loss panel');
  checkTrainable(options.random?.nodes ?? [], 'random.nodes', 'loss panel');
  const hasButtons = options.sgd !== undefined || options.random !== undefined || options.reset === true;

  return (
    <Card>
      <div class="vf-readout">
        <div>
          <div class="vf-big">
            <i>
              <Rich text={'$\\ell$ ='} />
            </i>
            <span class="vf-loss-value">{display(loss(), 3)}</span>
          </div>
          <Show when={options.parts !== undefined}>
            <div class="vf-lossrows">
              <For each={parts()}>{(v, t) => <Rich text={`${indexed('\\ell', t() + 1)} = ${display(v, 3)}`} />}</For>
            </div>
          </Show>
          <Show when={options.text}>{(text) => <TemplateText ctx={props.ctx} class="vf-sub" text={text()} />}</Show>
        </div>
        <Show when={options.hint !== undefined || props.ctx.history().length >= 2}>
          <Spark history={props.ctx.history()} hint={options.hint} />
        </Show>
      </div>
      <Show when={hasButtons}>
        <div class="vf-btns">
          <Show when={options.sgd}>
            {(sgd) => (
              <button type="button" class="vf-btn vf-pri" onClick={() => props.ctx.sgdStep(sgd())}>
                <Rich text={sgdLabel(sgd())} />
              </button>
            )}
          </Show>
          <Show when={options.random}>
            {(random) => (
              <button type="button" class="vf-btn" onClick={() => randomize(random())}>
                <Rich text={random().label ?? `随机 ${names(random().nodes)}`} />
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

function SequencePanel(props: { ctx: Context; item: SequenceOptions }) {
  const { ctx, item } = props;
  const sequence = ctx.sequence;
  if (sequence === undefined) throw new Error('sequence panel: mountPanels needs the sequence option');
  checkTrainable(item.sgd?.params ?? [], 'sgd.params', 'sequence panel');
  const initial = sequence.get();
  const ids = createMemo(() => {
    ctx.version();
    return sequence.get();
  });
  const positions = Array.from({ length: sequence.length }, (_, p) => p);
  const choose = (p: number, id: number) => {
    const next = [...sequence.get()];
    next[p] = id;
    sequence.set(next);
  };
  const randomLabel = typeof item.random === 'object' ? (item.random.label ?? '随机序列') : '随机序列';
  return (
    <Card title={item.title}>
      <div class="vf-seqrow" style={{ 'grid-template-columns': `repeat(${sequence.length}, minmax(0, 1fr))` }}>
        <For each={positions}>
          {(p) => {
            const label = item.labels?.[p] ?? String(p + 1);
            return (
              <label>
                <Rich text={label} />
                <select aria-label={plain(label)} value={ids()[p]} onChange={(e) => choose(p, Number(e.currentTarget.value))}>
                  <For each={sequence.tokens}>
                    {(token, i) => (
                      <option value={i()} selected={ids()[p] === i()}>
                        {token}
                      </option>
                    )}
                  </For>
                </select>
              </label>
            );
          }}
        </For>
      </div>
      <Show when={item.note}>{(note) => <TemplateText ctx={ctx} class="vf-note" text={note()} />}</Show>
      <div class="vf-btns">
        <Show when={item.sgd}>
          {(sgd) => (
            <button type="button" class="vf-btn vf-pri" onClick={() => ctx.sgdStep(sgd())}>
              <Rich text={sgdLabel(sgd())} />
            </button>
          )}
        </Show>
        <Show when={item.random}>
          <button type="button" class="vf-btn" onClick={() => sequence.set(positions.map(() => Math.floor(Math.random() * sequence.tokens.length)))}>
            {randomLabel}
          </button>
        </Show>
        <Show when={item.reset}>
          <button
            type="button"
            class="vf-btn"
            onClick={() => {
              ctx.g.reset();
              sequence.set(initial);
            }}
          >
            重置
          </button>
        </Show>
      </div>
    </Card>
  );
}

function AttentionPanel(props: { ctx: Context; item: Extract<PanelItem, { attention: Node[] }> }) {
  const { ctx, item } = props;
  const extra = item.extra_keys ?? [];
  for (const node of item.attention) {
    if (node.shape.length !== 2 || node.shape[1] !== node.shape[0] + extra.length) {
      const expected = extra.length === 0 ? 'a square matrix' : `a (T, T + ${extra.length}) matrix for ${extra.length} extra_keys`;
      throw new Error(`attention panel: node '${node.name}' must be ${expected}, got shape ${formatShape(node.shape)}`);
    }
  }
  const tokens = createMemo(() => {
    ctx.version();
    return Array.from(item.ids.value.data, (id) => item.ids.tokens?.[Math.round(id)] ?? String(Math.round(id)));
  });
  return (
    <Card title={item.title}>
      <div class="vf-attg" style={{ 'grid-template-columns': `repeat(${item.attention[0].shape[0] > ATTENTION_SIDE_BY_SIDE ? 1 : item.attention.length}, minmax(0, 1fr))` }}>
        <For each={item.attention}>
          {(node, h) => {
            const size = node.shape[0];
            const values = createMemo(() => {
              ctx.version();
              return node.value;
            });
            const rows = Array.from({ length: size }, (_, i) => i);
            const columns = Array.from({ length: node.shape[1] }, (_, i) => i);
            return (
              <div class="vf-attn" style={{ 'grid-template-columns': `34px repeat(${columns.length}, minmax(0, 1fr))` }}>
                <div class="vf-h">
                  <Rich text={indexed('h', h() + 1)} />
                </div>
                <For each={columns}>{(c) => <div class="vf-h">{c < size ? tokens()[c] : <Rich text={extra[c - size]} />}</div>}</For>
                <For each={rows}>
                  {(r) => (
                    <>
                      <div class="vf-h">{tokens()[r]}</div>
                      <For each={columns}>
                        {(c) => {
                          const v = () => values().get([r, c]);
                          return <div style={{ background: tint(v(), ATTENTION_LIMIT) }}>{(c > r && c < size) || v() === 0 ? '·' : display(v(), CELL_DIGITS)}</div>;
                        }}
                      </For>
                    </>
                  )}
                </For>
              </div>
            );
          }}
        </For>
      </div>
      <Show when={item.note}>{(note) => <TemplateText ctx={ctx} class="vf-note" text={note()} />}</Show>
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
    <Card title={props.item.title ?? `$${node.tex}$，${formatShape(node.shape)}`}>
      <For each={indices}>
        {(i) => {
          const label = `${node.name}${subscript(i + 1)}`;
          return (
            <div class="vf-hrow">
              <b>
                <Rich text={indexed(node.tex, i + 1)} />
              </b>
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
    return { node: spec.node, grad, title: spec.title ?? (grad ? `$\\partial\\ell/\\partial ${spec.node.tex}$` : `$${spec.node.tex}$`) };
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
            <Show when={item.sliders}>{(node) => <th><Rich text={`$${node().tex}$`} /></th>}</Show>
            <For each={columns}>{(c) => <th class="vf-n"><Rich text={c.title} /></th>}</For>
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

type MatrixItem = Extract<PanelItem, { matrix: MatrixSource }> | Extract<PanelItem, { matrices: MatrixEntry[] }>;

function MatrixGrid(props: { ctx: Context; entry: MatrixEntry; highlight?: MatrixHighlight; rows?: string }) {
  const { ctx, entry } = props;
  const source = entry.matrix;
  const read = (): Tensor => {
    const tensor = source instanceof Node ? (entry.grad ? gradOf(source) : source.value) : source(ctx.g, ctx.state());
    if (tensor.ndim !== 2) {
      const what = source instanceof Node ? `node '${source.name}'` : 'matrix function result';
      throw new Error(`matrix panel: ${what} must be 2-D, got shape ${formatShape(tensor.shape)}`);
    }
    return tensor;
  };
  const initial = read();
  const fixed = entry.limit;
  const limit = fixed ?? Math.max(MIN_MATRIX_LIMIT, ...Array.from(initial.data, Math.abs));
  const [rows, cols] = initial.shape;
  const tensor = createMemo(() => {
    ctx.version();
    return read();
  });
  const spec = props.highlight;
  const highlight = spec instanceof Node ? spec : spec?.node;
  const frameRow = !(spec instanceof Node) && spec?.axis === 'row';
  const frameAll = !(spec instanceof Node) && spec?.all === true;
  const tokenHeaders = highlight?.tokens !== undefined && classCount(highlight) === cols ? highlight.tokens : undefined;
  const frameArea = (t: number) =>
    frameRow ? { 'grid-row': `${t + 2}`, 'grid-column': `2 / span ${cols}` } : { 'grid-row': `2 / span ${rows}`, 'grid-column': `${t + 2}` };
  const target = createMemo(() => {
    ctx.version();
    return highlight === undefined || frameAll ? undefined : Math.round(highlight.value.get(targetIndex(highlight, ctx.focus())));
  });
  const targets = createMemo(() => {
    ctx.version();
    return frameAll && highlight !== undefined ? new Set(Array.from(highlight.value.data, Math.round)) : new Set<number>();
  });
  const shade = (v: number) => {
    if (Math.abs(v) < ZERO_CELL || Number(display(Math.abs(v), CELL_DIGITS)) === 0) return 'transparent';
    return tint(v, limit);
  };
  const rowIds = Array.from({ length: rows }, (_, r) => r);
  const colIds = Array.from({ length: cols }, (_, c) => c);
  const rowName = (r: number) => (props.rows === undefined ? String(r + 1) : indexed(props.rows, r + 1));

  return (
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
              <Rich text={rowName(r)} />
            </div>
            <For each={colIds}>
              {(c) => {
                const value = () => tensor().get([r, c]);
                return (
                  <div classList={{ 'vf-ct': targets().has(c) }} style={{ background: shade(value()), 'grid-row': `${r + 2}`, 'grid-column': `${c + 2}` }}>
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
  );
}

function MatrixPanel(props: { ctx: Context; item: MatrixItem }) {
  const { ctx, item } = props;
  const entries: MatrixEntry[] = 'matrices' in item ? item.matrices : [item];
  const first = entries[0];
  const source = first?.matrix;
  const title =
    item.title ??
    ('matrix' in item && source instanceof Node ? `$${first.grad ? `\\partial\\ell/\\partial ${source.tex}` : source.tex}$，${formatShape(source.shape)}` : undefined);
  const grid = (entry: MatrixEntry) => <MatrixGrid ctx={ctx} entry={entry} highlight={item.highlight} rows={item.rows} />;
  const captioned = entries.some((entry) => entry.caption !== undefined) || entries.length > 1;

  return (
    <Card title={title}>
      <Show when={captioned} fallback={grid(first)}>
        <div class="vf-wg">
          <For each={entries}>
            {(entry) => (
              <div>
                <Show when={entry.caption}>
                  <div class="vf-wcap">
                    <Rich text={entry.caption ?? ''} />
                  </div>
                </Show>
                {grid(entry)}
              </div>
            )}
          </For>
        </div>
      </Show>
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

function GenerationPanel(props: { ctx: Context; item: Extract<PanelItem, { generation: InferSpec }> }) {
  const { ctx, item } = props;
  const infer = item.generation;
  const vocab = infer.label.tokens ?? infer.input.tokens ?? [];
  const length = infer.input.shape[0] + 1;
  const ids = createMemo(() => {
    ctx.version();
    return [...Array.from(infer.input.value.data.subarray(0, 1)), ...Array.from(infer.label.value.data)].map(Math.round);
  });
  const generation = () => ctx.state().generation;
  const sampler = infer.sampler;
  const settings = sampler === undefined ? 'greedy：每步取概率最大的 token' : [`$\\tau=${sampler.temperature}$`, sampler.topK === undefined ? '' : `top-k $=${sampler.topK}$`, sampler.topP === undefined ? '' : `top-p $=${sampler.topP}$`, `seed ${sampler.seed}`].filter((x) => x !== '').join('，');
  const positions = Array.from({ length }, (_, p) => p);
  return (
    <Card title={item.title ?? '生成过程'}>
      <Show when={generation()} fallback={<p class="vf-note">选一个 prefill 或 decode 步骤查看生成过程。</p>}>
        {(state) => (
          <>
            <div class="vf-gen-phase">
              {state().position < infer.prompt ? `prefill：一次算 prompt 的 ${infer.prompt} 个位置` : `decode：只算第 ${state().position + 1} 个位置`}
            </div>
            <div class="vf-gen-seq" style={{ 'grid-template-columns': `repeat(${length}, minmax(0, 1fr))` }}>
              <For each={positions}>
                {(p) => {
                  const known = () => p < state().known;
                  const generated = p >= infer.prompt;
                  const draw = generated ? infer.draws.get(p - 1) : undefined;
                  const cls = () => ['vf-gen-tok', generated ? 'vf-gen-new' : 'vf-gen-prompt', known() ? '' : 'vf-gen-pending', p === state().position ? 'vf-gen-now' : ''].join(' ');
                  return (
                    <div class={cls()}>
                      <span class="vf-gen-pos">{p + 1}</span>
                      <span class="vf-gen-word">{known() ? (vocab[ids()[p]] ?? String(ids()[p])) : '?'}</span>
                      <span class="vf-gen-p">{draw !== undefined && known() ? `p ${Math.exp(draw.logp[draw.token]).toFixed(PROB_DIGITS)}` : generated ? '' : 'prompt'}</span>
                    </div>
                  );
                }}
              </For>
            </div>
            <p class="vf-note">
              <Rich text={`采样：${settings}。每个生成的 token 下面是模型给它的概率 $p$。`} />
            </p>
            <Show when={state().caches.length > 0}>
              <table class="vf-gen-cache">
                <thead>
                  <tr>
                    <th>缓存</th>
                    <th>保存的数</th>
                    <th>每个 token</th>
                  </tr>
                </thead>
                <tbody>
                  <For each={state().caches}>
                    {(row) => (
                      <tr>
                        <td>
                          <Rich text={row.nodes.map((n) => `$${n.tex}$`).join(' ')} />
                          <div class="vf-gen-kind">{CACHE_KIND_TEXT[row.kind]}</div>
                        </td>
                        <td>{row.values}</td>
                        <td>{row.kind === 'rows' ? `+${row.perToken}` : '+0'}</td>
                      </tr>
                    )}
                  </For>
                </tbody>
              </table>
              <p class="vf-note">处理完第 {state().position + 1} 个位置后，图里每份缓存保存的数，按页面的玩具尺寸计算。真实显存还要乘每个数的字节数，没画出的层也各有一份。</p>
            </Show>
          </>
        )}
      </Show>
      <Show when={item.note}>{(note) => <TemplateText ctx={ctx} class="vf-note" text={note()} />}</Show>
    </Card>
  );
}

function Panel(props: { ctx: Context; item: PanelItem }) {
  const item = props.item;
  if (item === 'loss') return <LossPanel ctx={props.ctx} options={{}} />;
  if (item === 'cost') return <CostPanel g={props.ctx.g} steps={props.ctx.steps} stepsVersion={props.ctx.stepsVersion} options={{}} />;
  if ('cost' in item) return <CostPanel g={props.ctx.g} steps={props.ctx.steps} stepsVersion={props.ctx.stepsVersion} options={item.cost} />;
  if ('loss' in item) return <LossPanel ctx={props.ctx} options={item.loss} />;
  if ('sequence' in item) return <SequencePanel ctx={props.ctx} item={item} />;
  if ('attention' in item) return <AttentionPanel ctx={props.ctx} item={item} />;
  if ('sliders' in item && !('tokens' in item)) return <SlidersPanel ctx={props.ctx} item={item} />;
  if ('tokens' in item) return <TokensPanel ctx={props.ctx} item={item} />;
  if ('matrix' in item || 'matrices' in item) return <MatrixPanel ctx={props.ctx} item={item} />;
  if ('code' in item) return <CodePanel ctx={props.ctx} item={item} />;
  if ('generation' in item) return <GenerationPanel ctx={props.ctx} item={item} />;
  if ('kvBlocks' in item) {
    return (
      <Card title={item.title ?? 'KV cache 的块'}>
        <KvBlocksPanel options={item.kvBlocks} version={props.ctx.version} state={props.ctx.state} />
        <Show when={item.note}>{(note) => <TemplateText ctx={props.ctx} class="vf-note" text={note()} />}</Show>
      </Card>
    );
  }
  return (
    <Card title={item.title}>
      <TemplateText ctx={props.ctx} class="vf-note" text={item.note} />
    </Card>
  );
}

export interface PanelsOptions {
  steps?: Steps;
  sequence?: PageSequence;
}

function PanelsView(props: { g: Graph; items: PanelItem[]; options: PanelsOptions }) {
  const { g } = props;
  const { steps, sequence } = props.options;
  const version = trackGraph(g);
  const stepsVersion = trackSteps(steps);
  const [history, setHistory] = createSignal<number[]>([]);
  let stepping = false;
  onCleanup(
    g.subscribe(() => {
      if (!stepping) setHistory([]);
    }),
  );
  const lossValue = () => g.loss().value.data[0];
  const sgdStep = (sgd: SgdOptions) => {
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
    history,
    sgdStep,
    sequence,
    steps,
    stepsVersion,
  };
  return (
    <div class="vf-panels">
      <For each={props.items}>{(item) => <Panel ctx={ctx} item={item} />}</For>
    </div>
  );
}

export function mountPanels(el: string | HTMLElement, g: Graph, items: PanelItem[], options: PanelsOptions = {}): Mounted {
  const target = resolveElement(el, 'mountPanels');
  return { dispose: render(() => <PanelsView g={g} items={items} options={options} />, target) };
}
