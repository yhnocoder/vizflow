import { For, Show, createMemo, onCleanup, onMount } from 'solid-js';
import { render } from 'solid-js/web';
import { renderText } from '../formula';
import type { StepInfo, Steps } from '../steps';
import { renderFigures } from './figures';
import { resolveElement, trackSteps } from './mount';
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

type TimelineInfo = StepInfo & { kind?: 'overview' | 'inputs' | 'op' | 'module'; segment?: string; sub?: string; module?: string };


interface Segment {
  name: string;
  positions: number[];
  folded: boolean;
}

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
    const list = steps.list as readonly TimelineInfo[];
    const current = steps.current;
    const dir = list[current].dir;
    const segments: Segment[] = [];
    list.forEach((info, position) => {
      if (info.dir !== dir) return;
      const name = info.segment ?? '';
      const last = segments[segments.length - 1];
      if (last !== undefined && last.name === name) last.positions.push(position);
      else segments.push({ name, positions: [position], folded: info.kind === 'module' });
    });
    const all = segments.flatMap((s) => s.positions);
    const active = segments.find((s) => s.positions.includes(current)) ?? segments[0];
    const directions = DIRECTIONS.filter((d) => list.some((info) => info.dir === d.dir));
    const only = active.positions.length === 1 ? list[active.positions[0]] : undefined;
    const showName = active.name !== '' && !(only !== undefined && only.label === active.name);
    let previous: string | undefined;
    const items = active.positions.map((position) => {
      const info = list[position];
      const sub = info.sub !== undefined && info.sub !== active.name ? info.sub : undefined;
      const heading = sub !== undefined && sub !== previous ? sub : undefined;
      previous = sub;
      return { info, position, heading };
    });
    return { dir, segments, all, active, directions, showName, items, position: all.indexOf(current) + 1 };
  });
  const dirName = () => (view().dir === 'forward' ? '前向' : '反向');

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
                  steps.go(steps.list.findIndex((info) => info.dir === d.dir));
                }}
              >
                {d.name}
              </button>
            )}
          </For>
        </div>
        <div class="vf-tl-bar">
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
                  title={segment.name}
                  aria-label={segment.name}
                  style={{ flex: `${segment.positions.length} 1 0`, ...(on() ? { '--vf-fr': `${fraction().toFixed(0)}%` } : {}) }}
                  onClick={() => steps.go(segment.positions[0])}
                />
              );
            }}
          </For>
        </div>
        <span class="vf-tl-pos" title={`${dirName()}第 ${view().position} 步，共 ${view().all.length} 步`}>
          {view().position}
          <i>/</i>
          {view().all.length}
        </span>
      </div>
      <div class="vf-tl-sub">
        <Show when={view().showName}>
          <span class="vf-tl-inner">{view().active.name + (view().active.folded ? '（已折叠）' : '')}</span>
        </Show>
        <For each={view().items}>
          {(item) => (
            <>
              <Show when={item.heading}>{(heading) => <span class="vf-tl-inner vf-sub">{heading()}</span>}</Show>
              <button
                type="button"
                class="vf-st"
                aria-current={props.current() === item.position ? 'step' : undefined}
                onClick={() => steps.go(item.position)}
                ref={props.label(item.info)}
              >
                {props.plainLabel(item.info)}
              </button>
            </>
          )}
        </For>
      </div>
    </div>
  );
}

export function mountSteps(el: string | HTMLElement, steps: Steps): Mounted {
  const target = resolveElement(el, 'mountSteps');
  return { dispose: render(() => <StepsView steps={steps} />, target) };
}
