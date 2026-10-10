import type { Graph } from '../autograd';
import type { SceneOptions } from '../scene3d';
import { createSteps, type Steps, type StepsSpec } from '../steps';
import { mountPanels, mountSteps, type Mounted, type PanelItem } from '../ui';
import elementCss from './element.css?inline';

const styleId = 'vf-element-style';

if (!document.getElementById(styleId)) {
  const style = document.createElement('style');
  style.id = styleId;
  style.textContent = elementCss;
  document.head.append(style);
}

export type VizflowPageScene = Omit<SceneOptions, 'steps' | 'edit'>;

const propertyNames = ['graph', 'steps', 'panels', 'scene', 'edit'] as const;

function element(tag: string, className: string): HTMLElement {
  const el = document.createElement(tag);
  el.className = className;
  return el;
}

export class VizflowPage extends HTMLElement {
  #graph: Graph | undefined;
  #steps: StepsSpec | undefined;
  #panels: PanelItem[] | undefined;
  #scene: VizflowPageScene | undefined;
  #edit = false;
  #created: Steps | undefined;
  #mounted: Mounted[] = [];
  #layout: HTMLElement | undefined;
  #generation = 0;
  #pending = false;

  constructor() {
    super();
    for (const name of propertyNames) {
      if (!Object.hasOwn(this, name)) continue;
      const value: unknown = Reflect.get(this, name);
      Reflect.deleteProperty(this, name);
      Reflect.set(this, name, value);
    }
  }

  get graph(): Graph | undefined {
    return this.#graph;
  }

  set graph(value: Graph | undefined) {
    this.#graph = value;
    this.#schedule();
  }

  get steps(): StepsSpec | undefined {
    return this.#steps;
  }

  set steps(value: StepsSpec | undefined) {
    this.#steps = value;
    this.#schedule();
  }

  get panels(): PanelItem[] | undefined {
    return this.#panels;
  }

  set panels(value: PanelItem[] | undefined) {
    this.#panels = value;
    this.#schedule();
  }

  get scene(): VizflowPageScene | undefined {
    return this.#scene;
  }

  set scene(value: VizflowPageScene | undefined) {
    this.#scene = value;
    this.#schedule();
  }

  get edit(): boolean {
    return this.#edit;
  }

  set edit(value: boolean) {
    this.#edit = value;
    this.#schedule();
  }

  get createdSteps(): Steps | undefined {
    this.#flush();
    return this.#created;
  }

  connectedCallback(): void {
    this.#schedule();
  }

  disconnectedCallback(): void {
    this.#pending = false;
    this.#unmount();
  }

  #schedule(): void {
    if (this.#pending) return;
    this.#pending = true;
    queueMicrotask(() => this.#flush());
  }

  #flush(): void {
    if (!this.#pending) return;
    this.#pending = false;
    this.#unmount();
    const g = this.#graph;
    const spec = this.#steps;
    if (!this.isConnected || g === undefined || spec === undefined) return;
    try {
      this.#mount(g, spec);
    } catch (error) {
      this.#unmount();
      throw error;
    }
  }

  #mount(g: Graph, spec: StepsSpec): void {
    const steps = createSteps(g, spec);
    this.#created = steps;
    const main = element('div', 'vf-main');
    const stage = element('div', 'vf-stage');
    const stepsBox = element('div', 'vf-steps');
    if (this.#scene !== undefined) main.append(stage);
    main.append(stepsBox);
    const panels = this.#panels;
    const side = element('aside', 'vf-side');
    const layout = element('div', panels === undefined ? 'vf-stack' : 'vf-split');
    layout.append(main);
    if (panels !== undefined) layout.append(side);
    this.#layout = layout;
    this.replaceChildren(layout);
    this.#mounted.push(mountSteps(stepsBox, steps));
    if (panels !== undefined) this.#mounted.push(mountPanels(side, g, panels, { steps }));
    if (this.#scene !== undefined) void this.#mountScene(stage, g, { ...this.#scene, steps, edit: this.#edit });
  }

  async #mountScene(stage: HTMLElement, g: Graph, options: SceneOptions): Promise<void> {
    const generation = this.#generation;
    const { mountScene } = await import('../scene3d');
    if (generation === this.#generation) this.#mounted.push(mountScene(stage, g, options));
  }

  #unmount(): void {
    this.#generation++;
    for (const mounted of this.#mounted.splice(0)) mounted.dispose();
    this.#created?.dispose();
    this.#created = undefined;
    this.#layout?.remove();
    this.#layout = undefined;
  }
}

if (customElements.get('vizflow-page') === undefined) customElements.define('vizflow-page', VizflowPage);

declare global {
  interface HTMLElementTagNameMap {
    'vizflow-page': VizflowPage;
  }
}
