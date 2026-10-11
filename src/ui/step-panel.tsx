import { Show, createEffect, untrack } from 'solid-js';
import { render } from 'solid-js/web';
import type { StepInfo, Steps } from '../steps';
import { resolveElement, trackSteps } from './mount';
import type { Mounted } from './mounted';
import { Rich } from './rich';

export interface StepPanelStep {
  index: number;
  info: StepInfo;
  steps: Steps;
}

export interface StepPanelOptions {
  title?: string;
  render(root: HTMLElement, step: StepPanelStep): void;
}

function StepPanelView(props: { steps: Steps; options: StepPanelOptions }) {
  const steps = props.steps;
  const version = trackSteps(steps);
  let body!: HTMLDivElement;
  createEffect(() => {
    version();
    untrack(() => props.options.render(body, { index: steps.current, info: steps.list[steps.current], steps }));
  });
  return (
    <div class="vf-panels">
      <section class="vf-card vf-step-panel">
        <Show when={props.options.title}>{(title) => <h2><Rich text={title()} /></h2>}</Show>
        <div class="vf-step-panel-body" ref={(el) => (body = el)} />
      </section>
    </div>
  );
}

export function mountStepPanel(el: string | HTMLElement, steps: Steps, options: StepPanelOptions): Mounted {
  const target = resolveElement(el, 'mountStepPanel');
  return { dispose: render(() => <StepPanelView steps={steps} options={options} />, target) };
}
