import { For, Show, createEffect, createMemo, createSignal, on, onCleanup, onMount } from 'solid-js';
import { render } from 'solid-js/web';
import { renderText } from '../formula';
import { Rich, plain } from './rich';
import type { StepInfo, Steps } from '../steps';
import { renderFigures } from './figures';
import { resolveElement, trackSteps } from './mount';
import { counterpart } from './counterpart';
import type { Mounted } from './mounted';

function StepsView(props: { steps: Steps }) {
  const steps = props.steps;
  const stepsVersion = trackSteps(steps);
  const current = createMemo(() => {
    stepsVersion();
    return steps.current;
  });
  const entries = steps.list.map((info, position) => ({ info, position }));
  const forward = entries.filter((e) => e.info.dir === 'forward');
  const backward = entries.filter((e) => e.info.dir === 'backward');

  let title!: HTMLHeadingElement;
  let body!: HTMLDivElement;
  let frame: number | undefined;
  const draw = () => {
    frame = undefined;
    const text = steps.text();
    renderText(text.title, title);
    renderText(text.body, body);
    renderFigures(body, steps.graph);
  };
  const schedule = () => {
    if (frame === undefined) frame = requestAnimationFrame(draw);
  };
  onMount(draw);
  onCleanup(steps.graph.subscribe(schedule));
  onCleanup(steps.subscribe(schedule));
  onCleanup(() => {
    if (frame !== undefined) cancelAnimationFrame(frame);
  });

  let root!: HTMLDivElement;
  let engaged = false;
  const onInteract = (event: Event) => {
    const target = event.target;
    if (!(target instanceof Element) || !target.closest('.vf-stepper')) return;
    engaged = root.contains(target);
  };
  const onKey = (event: KeyboardEvent) => {
    if (event.defaultPrevented || event.altKey || event.ctrlKey || event.metaKey) return;
    if (event.key !== 'ArrowRight' && event.key !== 'ArrowLeft') return;
    if (event.target instanceof Element) {
      if (event.target.closest('input, textarea, select, [contenteditable]')) return;
      if (event.target.closest('button') && !root.contains(event.target)) return;
    }
    if (!engaged && document.querySelectorAll('.vf-stepper').length > 1) return;
    if (event.key === 'ArrowRight') steps.next();
    else steps.prev();
  };
  document.addEventListener('pointerdown', onInteract);
  document.addEventListener('focusin', onInteract);
  document.addEventListener('keydown', onKey);
  onCleanup(() => {
    document.removeEventListener('pointerdown', onInteract);
    document.removeEventListener('focusin', onInteract);
    document.removeEventListener('keydown', onKey);
  });

  const label = (info: StepInfo) => (el: HTMLElement) => {
    if (info.label.includes('$')) renderText(info.label, el);
  };
  const plainLabel = (info: StepInfo) => (info.label.includes('$') ? undefined : info.label);

  const button = (entry: { info: StepInfo; position: number }) => (
    <button
      type="button"
      class="vf-st"
      aria-current={current() === entry.position ? 'step' : undefined}
      onClick={() => steps.go(entry.position)}
      ref={label(entry.info)}
    >
      {plainLabel(entry.info)}
    </button>
  );

  const timeline = createMemo(() => {
    stepsVersion();
    return steps.modules.length > 0;
  });
  const lastIndex = createMemo(() => {
    stepsVersion();
    return steps.list.length - 1;
  });

  return (
    <div class="vf-stepper" classList={{ 'vf-timeline': timeline() }} ref={(el) => (root = el)}>
      <div class="vf-steprow">
        <button type="button" class="vf-nav" aria-label="上一步" disabled={current() === 0} onClick={() => steps.prev()}>
          ←
        </button>
        <Show
          when={timeline()}
          fallback={
            <div class="vf-track">
              <span class="vf-grp">前向</span>
              <For each={forward}>{button}</For>
              <Show when={backward.length > 0}>
                <span class="vf-sep" />
                <span class="vf-grp">反向</span>
                <For each={backward}>{button}</For>
              </Show>
            </div>
          }
        >
          <Timeline steps={steps} version={stepsVersion} current={current} label={label} plainLabel={plainLabel} />
        </Show>
        <button type="button" class="vf-nav" aria-label="下一步" disabled={current() === lastIndex()} onClick={() => steps.next()}>
          →
        </button>
      </div>
      <div class="vf-caption">
        <h3 ref={(el) => (title = el)} />
        <div class="vf-caption-body" ref={(el) => (body = el)} />
      </div>
    </div>
  );
}

interface Segment {
  name: string;
  positions: number[];
  folded: boolean;
}

type Item = { kind: 'step'; info: StepInfo; position: number; inner?: boolean } | { kind: 'group'; id: string; name: string; position: number; run: number[]; inner?: boolean };

export const SEGMENT_PREVIEW_STEPS = 5;
export const PEEK_ARROW_INSET = 18;

const DIRECTIONS = [
  { dir: 'forward', name: '前向' },
  { dir: 'backward', name: '反向' },
] as const;

function Timeline(props: {
  steps: Steps;
  version: () => number;
  current: () => number;
  label: (info: StepInfo) => (el: HTMLElement) => void;
  plainLabel: (info: StepInfo) => string | undefined;
}) {
  const steps = props.steps;
  const view = createMemo(() => {
    props.version();
    const list = steps.list;
    const current = steps.current;
    const dir = list[current].dir;
    const titles = new Map(steps.groups.map((g) => [g.id, g.title]));
    const segments: Segment[] = [];
    let lastKey: string | undefined;
    list.forEach((info, position) => {
      if (info.dir !== dir) return;
      const top = info.groups[0];
      const key = top ?? `step:${info.forwardId ?? info.id}`;
      const name = top === undefined ? info.label : (titles.get(top) ?? '');
      const last = segments[segments.length - 1];
      if (last !== undefined && lastKey === key) last.positions.push(position);
      else segments.push({ name, positions: [position], folded: info.kind === 'module' });
      lastKey = key;
    });
    const all = segments.flatMap((s) => s.positions);
    const active = segments.find((s) => s.positions.includes(current)) ?? segments[0];
    const directions = DIRECTIONS.filter((d) => list.some((info) => info.dir === d.dir));
    const chain = list[current].groups;
    const firstIn = (depth: number) => list.findIndex((info) => info.dir === dir && info.groups.length > depth && info.groups[depth] === chain[depth]);
    const path = chain.map((id, depth) => ({ name: titles.get(id) ?? id, position: firstIn(depth) }));
    const inside = (depth: number) => (info: StepInfo) => info.dir === dir && chain.slice(0, depth).every((id, at) => info.groups[at] === id);
    const itemsAt = (depth: number, pool: number[]): Item[] => {
      const out: Item[] = [];
      for (const position of pool) {
        const info = list[position];
        const child = info.groups[depth];
        if (child === undefined) out.push({ kind: 'step', info, position });
        else {
          const last = out[out.length - 1];
          if (last?.kind === 'group' && last.id === child) last.run.push(position);
          else out.push({ kind: 'group', id: child, name: titles.get(child) ?? child, position, run: [position] });
        }
      }
      return out;
    };
    const positionsInside = (depth: number) => list.flatMap((info, position) => (inside(depth)(info) ? [position] : []));
    const items =
      chain.length < 2
        ? itemsAt(chain.length, chain.length === 0 ? active.positions : positionsInside(chain.length))
        : itemsAt(chain.length - 1, positionsInside(chain.length - 1)).flatMap((item) => (item.kind === 'group' && item.run.includes(current) ? itemsAt(chain.length, item.run).map((own) => ({ ...own, inner: true })) : [item]));
    return { dir, segments, all, active, directions, path, items, position: all.indexOf(current) + 1 };
  });
  const dirName = () => (view().dir === 'forward' ? '前向' : '反向');
  const [peek, setPeek] = createSignal<{ segment: Segment; center: number } | undefined>();
  let bar!: HTMLDivElement;
  const openPeek = (segment: Segment, el: HTMLElement) => {
    const box = bar.getBoundingClientRect();
    const own = el.getBoundingClientRect();
    setPeek({ segment, center: own.left + own.width / 2 - box.left });
  };
  const closePeek = () => setPeek(undefined);

  return (
    <div class="vf-track">
      <div class="vf-tl-row">
        <div class="vf-dirseg" role="tablist">
          <For each={view().directions}>
            {(d) => (
              <button
                type="button"
                role="tab"
                aria-selected={view().dir === d.dir ? 'true' : 'false'}
                onClick={() => {
                  if (d.dir === view().dir) return;
                  steps.go(counterpart(steps.list, steps.current));
                }}
              >
                {d.name}
              </button>
            )}
          </For>
        </div>
        <div class="vf-tl-bar" ref={(el) => (bar = el)} onPointerLeave={closePeek}>
          <For each={view().segments}>
            {(segment) => {
              const on = () => segment === view().active;
              const done = () => segment.positions[segment.positions.length - 1] < props.current();
              const fraction = () => ((segment.positions.indexOf(props.current()) + 1) / segment.positions.length) * 100;
              return (
                <button
                  type="button"
                  class="vf-tb"
                  classList={{ 'vf-on': on(), 'vf-done': done() }}
                  aria-label={plain(segment.name)}
                  style={{ flex: `${segment.positions.length} 1 0`, ...(on() ? { '--vf-fr': `${fraction().toFixed(0)}%` } : {}) }}
                  onClick={() => steps.go(segment.positions[0])}
                  onPointerEnter={(ev) => openPeek(segment, ev.currentTarget)}
                  onFocus={(ev) => openPeek(segment, ev.currentTarget)}
                  onBlur={closePeek}
                />
              );
            }}
          </For>
          <Show when={peek()}>{(p) => <SegmentPeek steps={steps} segment={p().segment} center={p().center} current={props.current()} bar={bar} />}</Show>
        </div>
        <span class="vf-tl-pos" title={`${dirName()}第 ${view().position} 步，共 ${view().all.length} 步`}>
          {view().position}
          <i>/</i>
          {view().all.length}
        </span>
      </div>
      <div class="vf-tl-path">
        <For each={view().path}>
          {(crumb, i) => (
            <>
              <Show when={i() > 0}>
                <span class="vf-tl-crumbsep">›</span>
              </Show>
              <button type="button" class="vf-tl-crumb" onClick={() => steps.go(crumb.position)}>
                <Rich text={crumb.name} />
              </button>
            </>
          )}
        </For>
      </div>
      <div class="vf-tl-sub">
        <For each={view().items}>
          {(item) =>
            item.kind === 'step' ? (
              <button
                type="button"
                class="vf-st"
                classList={{ 'vf-st-in': item.inner === true }}
                aria-current={props.current() === item.position ? 'step' : undefined}
                onClick={() => steps.go(item.position)}
                ref={props.label(item.info)}
              >
                {props.plainLabel(item.info)}
              </button>
            ) : (
              <button type="button" class="vf-st vf-st-group" classList={{ 'vf-st-in': item.inner === true }} onClick={() => steps.go(item.position)}>
                <Rich text={item.name} />
                <span class="vf-st-more">›</span>
              </button>
            )
          }
        </For>
      </div>
    </div>
  );
}

function SegmentPeek(props: { steps: Steps; segment: Segment; center: number; current: number; bar: HTMLElement }) {
  const list = props.steps.list;
  const positions = () => props.segment.positions;
  const index = () => positions().indexOf(props.current);
  const start = () => Math.max(0, Math.min(index() - Math.floor(SEGMENT_PREVIEW_STEPS / 2), positions().length - SEGMENT_PREVIEW_STEPS));
  const shown = () => positions().slice(start(), start() + SEGMENT_PREVIEW_STEPS);
  const rest = () => positions().length - start() - shown().length;
  const state = () => {
    const count = positions().length;
    if (index() >= 0) return `第 ${index() + 1} / ${count} 步`;
    return positions()[0] > props.current ? `${count} 步` : `${count} 步 · 已完成`;
  };
  const name = () => (props.segment.name === '' ? '步骤' : props.segment.name) + (props.segment.folded ? '（已折叠）' : '');
  const single = () => positions().length === 1 && list[positions()[0]].label === props.segment.name;
  let el!: HTMLDivElement;
  createEffect(
    on([() => props.center, () => props.segment, () => props.current], ([center]) => {
      const width = el.offsetWidth;
      const max = props.bar.clientWidth;
      const fitted = Math.min(Math.max(center - width / 2, 0), Math.max(max - width, 0));
      const left = Math.max(Math.min(fitted, center - PEEK_ARROW_INSET), center - width + PEEK_ARROW_INSET);
      el.style.left = `${left}px`;
      el.style.setProperty('--vf-peek-arrow', `${center - left}px`);
    }),
  );
  return (
    <div class="vf-peek" role="tooltip" ref={(e) => (el = e)}>
      <div class="vf-peek-head">
        <Rich class="vf-peek-name" text={name()} />
        <span class="vf-peek-count" classList={{ 'vf-on': index() >= 0 }}>
          {state()}
        </span>
      </div>
      <Show when={!single()}>
        <ol class="vf-peek-steps">
          <Show when={start() > 0}>
            <li class="vf-peek-more">前 {start()} 步</li>
          </Show>
          <For each={shown()}>
            {(position) => (
              <li classList={{ 'vf-on': position === props.current, 'vf-done': position < props.current }}>
                <Rich text={list[position].label} />
              </li>
            )}
          </For>
          <Show when={rest() > 0}>
            <li class="vf-peek-more">后 {rest()} 步</li>
          </Show>
        </ol>
      </Show>
    </div>
  );
}

export function mountSteps(el: string | HTMLElement, steps: Steps): Mounted {
  const target = resolveElement(el, 'mountSteps');
  return { dispose: render(() => <StepsView steps={steps} />, target) };
}
