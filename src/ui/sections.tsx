import { For, Show, createEffect, type JSX } from 'solid-js';
import { render } from 'solid-js/web';
import type { Graph, Node } from '../autograd';
import { renderMarkdown } from '../formula';
import { compileTemplate, type Steps } from '../steps';
import { renderFigures } from './figures';
import { resolveElement, trackGraph } from './mount';
import type { Mounted } from './mounted';
import { Rich } from './rich';
import { RopeCompare, RopeCurve, RopeDial, createRopeState } from './rope';

export interface RopeSpec {
  q: Node[];
  k: Node[];
  rows: [number, number];
  head: number;
  theta: number;
  dh: number;
  positions: number;
}

export interface RopeCompareSpec {
  scores: Node;
  weights: Node;
  ids: Node;
}

export interface JumpButton {
  label: string;
  go: Node;
}

export type SectionBlock =
  | { text: string }
  | { lead: string }
  | { buttons: JumpButton[] }
  | 'rope_dial'
  | 'rope_curve'
  | { rope_compare: RopeCompareSpec };

export interface SectionRow {
  columns?: (number | string)[];
  cells: SectionBlock[][];
}

export interface SectionItem {
  title?: string;
  rope?: RopeSpec;
  rows: SectionRow[];
}

export interface SectionsOptions {
  steps?: Steps;
  onJump?: () => void;
}

const columnTrack = (width: number | string) => `minmax(0, ${typeof width === 'number' ? `${width}px` : width})`;

function Markdown(props: { g: Graph; version: () => number; text: string; class: string }) {
  const evaluate = compileTemplate(props.g, props.text);
  let target!: HTMLDivElement;
  createEffect(() => {
    props.version();
    renderMarkdown(evaluate(0), target);
    renderFigures(target, props.g);
  });
  return <div class={props.class} ref={(el) => (target = el)} />;
}

function jumpTarget(steps: Steps, node: Node): number {
  const list = steps.list as readonly (Steps['list'][number] & { kind?: string; module?: string })[];
  const direct = list.findIndex((info) => info.dir === 'forward' && info.index > 0 && info.nodes.includes(node));
  if (direct >= 0) return direct;
  return list.findIndex((info) => info.dir === 'forward' && info.kind === 'module' && info.module !== undefined && node.name.startsWith(`${info.module}.`));
}

function SectionView(props: { g: Graph; item: SectionItem; options: SectionsOptions; version: () => number }) {
  const { g, item, options } = props;
  const rope = item.rope === undefined ? undefined : createRopeState(item.rope);
  const needRope = (block: string) => {
    if (rope === undefined || item.rope === undefined) throw new Error(`section: ${block} needs the section's rope option`);
    return { spec: item.rope, state: rope };
  };
  const jump = (button: JumpButton) => {
    const steps = options.steps;
    if (steps === undefined) throw new Error('section: buttons need the steps option');
    const target = jumpTarget(steps, button.go);
    if (target < 0) throw new Error(`section: node '${button.go.name}' is not shown by any forward step`);
    steps.go(target);
    options.onJump?.();
  };
  const block = (b: SectionBlock): JSX.Element => {
    if (b === 'rope_dial') {
      const { spec, state } = needRope(b);
      return <RopeDial g={g} version={props.version} spec={spec} state={state} />;
    }
    if (b === 'rope_curve') {
      const { spec, state } = needRope(b);
      return <RopeCurve g={g} version={props.version} spec={spec} state={state} />;
    }
    if ('rope_compare' in b) {
      const { spec } = needRope('rope_compare');
      return <RopeCompare version={props.version} spec={spec} compare={b.rope_compare} />;
    }
    if ('text' in b) return <Markdown g={g} version={props.version} text={b.text} class="vf-md" />;
    if ('lead' in b) return <Markdown g={g} version={props.version} text={b.lead} class="vf-md vf-lead" />;
    return (
      <div class="vf-btns vf-jumps">
        <For each={b.buttons}>
          {(button) => (
            <button type="button" class="vf-btn" onClick={() => jump(button)}>
              <Rich text={button.label} />
            </button>
          )}
        </For>
      </div>
    );
  };
  return (
    <div class="vf-card vf-section">
      <Show when={item.title}>
        <h2>
          <Rich text={item.title ?? ''} />
        </h2>
      </Show>
      <For each={item.rows}>
        {(row) => (
          <div
            class="vf-row"
            classList={{ 'vf-cols': row.columns !== undefined }}
            style={row.columns === undefined ? undefined : { 'grid-template-columns': row.columns.map(columnTrack).join(' ') }}
          >
            <For each={row.cells}>{(cell) => <div class="vf-cell">{cell.map(block)}</div>}</For>
          </div>
        )}
      </For>
    </div>
  );
}

function SectionsView(props: { g: Graph; items: SectionItem[]; options: SectionsOptions }) {
  const version = trackGraph(props.g);
  return (
    <div class="vf-sections">
      <For each={props.items}>{(item) => <SectionView g={props.g} item={item} options={props.options} version={version} />}</For>
    </div>
  );
}

export function mountSections(el: string | HTMLElement, g: Graph, items: SectionItem[], options: SectionsOptions = {}): Mounted {
  const target = resolveElement(el, 'mountSections');
  return { dispose: render(() => <SectionsView g={g} items={items} options={options} />, target) };
}
