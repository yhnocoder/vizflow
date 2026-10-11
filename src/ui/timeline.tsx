import { For, Show, createMemo, untrack } from 'solid-js';
import { render } from 'solid-js/web';
import type { StepInfo, Steps } from '../steps';
import { formatNumber } from '../tensor';
import { resolveElement, trackSteps } from './mount';
import type { Mounted } from './mounted';
import { Rich, plain } from './rich';

export interface TimelineLane {
  id: string;
  label: string;
}

export interface TimelineItem {
  lane: string;
  start: number;
  end: number;
  label?: string;
  kind?: string;
  step?: string | number;
}

export interface TimelineKind {
  label: string;
  color?: string;
}

export interface TimelineSpec {
  title?: string;
  unit?: string;
  lanes: TimelineLane[];
  items: TimelineItem[];
  kinds?: Record<string, TimelineKind>;
  times?: Record<string, [number, number]>;
}

export type TimelineSource = TimelineSpec | ((step: { index: number; info: StepInfo }) => TimelineSpec);

export const TIMELINE_TICKS = 5;
export const TIMELINE_PALETTE_SIZE = 6;
const NICE_STEPS = [1, 2, 5, 10];

interface Placed {
  item: TimelineItem;
  row: number;
  left: number;
  width: number;
  step: number | undefined;
  color: string;
  label: string;
  tip: string;
}

interface Layout {
  spec: TimelineSpec;
  lo: number;
  span: number;
  ticks: { value: number; at: number; text: string }[];
  items: Placed[];
  kinds: { id: string; label: string; color: string }[];
}

function resolveStep(steps: Steps, step: string | number, where: string): number {
  if (typeof step === 'number') {
    if (!Number.isInteger(step) || step < 0 || step >= steps.list.length) throw new Error(`mountTimeline: ${where} refers to step ${step}, but there are ${steps.list.length} steps`);
    return step;
  }
  const index = steps.list.findIndex((info) => info.id === step);
  if (index < 0) throw new Error(`mountTimeline: ${where} refers to step '${step}', which does not exist`);
  return index;
}

function niceTicks(lo: number, hi: number, count: number = TIMELINE_TICKS): { values: number[]; digits: number } {
  if (!(hi > lo)) return { values: [], digits: 0 };
  const raw = (hi - lo) / count;
  const power = Math.floor(Math.log10(raw));
  const magnitude = 10 ** power;
  const step = NICE_STEPS.find((n) => n * magnitude >= raw)! * magnitude;
  const digits = Math.max(0, -Math.floor(Math.log10(step) + 1e-9));
  const values: number[] = [];
  for (let k = Math.ceil(lo / step - 1e-9); k * step <= hi + step * 1e-9; k++) values.push(Number((k * step).toFixed(digits)));
  return { values, digits };
}

function layout(steps: Steps, spec: TimelineSpec): Layout {
  const rows = new Map(spec.lanes.map((lane, row) => [lane.id, row]));
  if (rows.size !== spec.lanes.length) throw new Error('mountTimeline: lane ids must be unique');
  for (const key of Object.keys(spec.times ?? {})) resolveStep(steps, key, `times['${key}']`);
  spec.items.forEach((item, i) => {
    if (!rows.has(item.lane)) throw new Error(`mountTimeline: item ${i} is on lane '${item.lane}', which is not in lanes`);
    if (!(item.end >= item.start)) throw new Error(`mountTimeline: item ${i} ends at ${item.end}, before its start ${item.start}`);
  });
  const lo = Math.min(...spec.items.map((item) => item.start));
  const hi = Math.max(...spec.items.map((item) => item.end));
  const span = spec.items.length > 0 && hi > lo ? hi - lo : 0;
  const at = (value: number) => (span === 0 ? 0 : ((value - lo) / span) * 100);
  const declared = Object.keys(spec.kinds ?? {});
  const used = [...new Set(spec.items.flatMap((item) => (item.kind === undefined ? [] : [item.kind])))];
  const order = [...declared, ...used.filter((kind) => !declared.includes(kind))];
  const colorOf = (kind: string) => spec.kinds?.[kind]?.color ?? `var(--vf-series-${(order.indexOf(kind) % TIMELINE_PALETTE_SIZE) + 1})`;
  const unit = spec.unit === undefined ? '' : ` ${spec.unit}`;
  const { values, digits } = niceTicks(lo, lo + span);
  const items = spec.items.map((item, i): Placed => {
    const label = item.label ?? '';
    const range = `${item.start} 到 ${item.end}${unit}`;
    return {
      item,
      row: rows.get(item.lane)!,
      left: at(item.start),
      width: at(item.end) - at(item.start),
      step: item.step === undefined ? undefined : resolveStep(steps, item.step, `item ${i}`),
      color: item.kind === undefined ? 'var(--muted)' : colorOf(item.kind),
      label,
      tip: label === '' ? range : `${plain(label)}：${range}`,
    };
  });
  return {
    spec,
    lo,
    span,
    ticks: values.map((value) => ({ value, at: at(value), text: `${formatNumber(value, { digits })}${unit}` })),
    items,
    kinds: order.filter((kind) => used.includes(kind)).map((kind) => ({ id: kind, label: spec.kinds?.[kind]?.label ?? kind, color: colorOf(kind) })),
  };
}

function TimelineView(props: { steps: Steps; source: TimelineSource }) {
  const steps = props.steps;
  const version = trackSteps(steps);
  const current = createMemo(() => {
    version();
    return steps.current;
  });
  const view = createMemo(() => {
    const index = current();
    const source = props.source;
    const spec = typeof source === 'function' ? untrack(() => source({ index, info: steps.list[index] })) : source;
    return layout(steps, spec);
  });
  const matched = createMemo(() => view().items.filter((placed) => placed.step === current()));
  const cursor = createMemo(() => {
    const v = view();
    const clamp = (value: number) => Math.min(100, Math.max(0, v.span === 0 ? 0 : ((value - v.lo) / v.span) * 100));
    const time = v.spec.times?.[steps.list[current()].id];
    const range = time ?? (matched().length > 0 ? [Math.min(...matched().map((p) => p.item.start)), Math.max(...matched().map((p) => p.item.end))] : undefined);
    if (range === undefined) return undefined;
    const left = clamp(range[0]);
    return { left, width: clamp(range[1]) - left };
  });
  const percent = (value: number) => `${value.toFixed(4)}%`;

  return (
    <div class="vf-panels">
      <section class="vf-card vf-gantt">
        <Show when={view().spec.title}>{(title) => <h2><Rich text={title()} /></h2>}</Show>
        <div class="vf-gantt-scroll">
          <div class="vf-gantt-grid" style={{ 'grid-template-rows': `repeat(${view().spec.lanes.length}, var(--vf-gantt-row)) auto` }}>
            <For each={view().spec.lanes}>
              {(lane, row) => (
                <div class="vf-gantt-lane-label" style={{ 'grid-row': `${row() + 1}` }}>
                  <Rich text={lane.label} />
                </div>
              )}
            </For>
            <For each={view().spec.lanes}>{(_, row) => <div class="vf-gantt-track" style={{ 'grid-row': `${row() + 1}` }} />}</For>
            <div class="vf-gantt-overlay" style={{ 'grid-row': `1 / ${view().spec.lanes.length + 1}` }}>
              <For each={view().ticks}>{(tick) => <span class="vf-gantt-gridline" style={{ left: percent(tick.at) }} />}</For>
              <Show when={cursor()}>{(c) => <span class="vf-gantt-cursor" style={{ left: percent(c().left), width: percent(c().width) }} />}</Show>
            </div>
            <div class="vf-gantt-items" style={{ 'grid-row': `1 / ${view().spec.lanes.length + 1}` }}>
              <For each={view().items}>
                {(placed) => {
                  const style = {
                    left: percent(placed.left),
                    width: percent(placed.width),
                    top: `calc(var(--vf-gantt-row) * ${placed.row})`,
                    '--vf-kind': placed.color,
                  };
                  const classList = () => ({ 'vf-on': placed.step !== undefined && placed.step === current(), 'vf-dim': matched().length > 0 && placed.step !== current() });
                  const label = placed.label === '' ? undefined : <Rich text={placed.label} />;
                  return placed.step === undefined ? (
                    <span class="vf-gantt-item" classList={classList()} style={style} title={placed.tip}>
                      {label}
                    </span>
                  ) : (
                    <button
                      type="button"
                      class="vf-gantt-item"
                      classList={classList()}
                      style={style}
                      title={placed.tip}
                      aria-current={placed.step === current() ? 'step' : undefined}
                      onClick={() => steps.go(placed.step!)}
                    >
                      {label}
                    </button>
                  );
                }}
              </For>
            </div>
            <div class="vf-gantt-axis" style={{ 'grid-row': `${view().spec.lanes.length + 1}` }}>
              <For each={view().ticks}>
                {(tick) => (
                  <span class="vf-gantt-tick" style={{ left: percent(tick.at) }}>
                    <span style={{ transform: `translateX(-${tick.at.toFixed(4)}%)` }}>{tick.text}</span>
                  </span>
                )}
              </For>
            </div>
          </div>
        </div>
        <Show when={view().kinds.length > 0}>
          <ul class="vf-gantt-legend">
            <For each={view().kinds}>
              {(kind) => (
                <li style={{ '--vf-kind': kind.color }}>
                  <Rich text={kind.label} />
                </li>
              )}
            </For>
          </ul>
        </Show>
      </section>
    </div>
  );
}

export function mountTimeline(el: string | HTMLElement, steps: Steps, spec: TimelineSource): Mounted {
  const target = resolveElement(el, 'mountTimeline');
  return { dispose: render(() => <TimelineView steps={steps} source={spec} />, target) };
}
