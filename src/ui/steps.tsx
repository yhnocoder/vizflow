import { For, Show, createMemo, onCleanup, onMount } from 'solid-js';
import { render } from 'solid-js/web';
import { renderText } from '../formula';
import type { Steps } from '../steps';
import { resolveElement, trackSteps, type Mounted } from './mount';

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
  const last = steps.list.length - 1;

  let title!: HTMLHeadingElement;
  let body!: HTMLDivElement;
  let frame: number | undefined;
  const draw = () => {
    frame = undefined;
    const text = steps.text();
    renderText(text.title, title);
    renderText(text.body, body);
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
    engaged = event.target instanceof globalThis.Node && root.contains(event.target);
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

  const button = (entry: { info: Steps['list'][number]; position: number }) => (
    <button
      type="button"
      class="vf-st"
      aria-current={current() === entry.position ? 'step' : undefined}
      onClick={() => steps.go(entry.position)}
    >
      {entry.info.label}
    </button>
  );

  return (
    <div class="vf-stepper" ref={(el) => (root = el)}>
      <div class="vf-steprow">
        <button type="button" class="vf-nav" aria-label="上一步" disabled={current() === 0} onClick={() => steps.prev()}>
          ←
        </button>
        <div class="vf-track">
          <span class="vf-grp">前向</span>
          <For each={forward}>{button}</For>
          <Show when={backward.length > 0}>
            <span class="vf-sep" />
            <span class="vf-grp">反向</span>
            <For each={backward}>{button}</For>
          </Show>
        </div>
        <button type="button" class="vf-nav" aria-label="下一步" disabled={current() === last} onClick={() => steps.next()}>
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

export function mountSteps(el: string | HTMLElement, steps: Steps): Mounted {
  const target = resolveElement(el, 'mountSteps');
  return { dispose: render(() => <StepsView steps={steps} />, target) };
}
