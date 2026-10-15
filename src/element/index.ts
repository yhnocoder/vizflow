import type { Graph } from '../autograd';
import type { CompiledPage } from '../dsl';
import type { SceneOptions } from '../scene3d';
import { createSteps, type StepInfo, type Steps, type StepsSpec } from '../steps';
import { mountPanels, mountSections, mountSteps, type Mounted, type PageSequence, type PanelItem, type PanelSettings, type SectionItem } from '../ui';
import elementCss from './element.css?inline';

const styleId = 'vf-element-style';

if (!document.getElementById(styleId)) {
  const style = document.createElement('style');
  style.id = styleId;
  style.textContent = elementCss;
  document.head.append(style);
}

export type VizflowPageScene = Omit<SceneOptions, 'steps' | 'edit'>;

interface Snapshot {
  params: Map<string, number[]>;
  sequence: number[] | undefined;
  step: { id: string; dir: StepInfo['dir']; groups: string[] } | undefined;
}

const propertyNames = ['graph', 'steps', 'panels', 'sections', 'sequence', 'scene', 'edit'] as const;

function element(tag: string, className: string): HTMLElement {
  const el = document.createElement(tag);
  el.className = className;
  return el;
}

export class VizflowPage extends HTMLElement {
  #graph: Graph | undefined;
  #steps: StepsSpec | undefined;
  #panels: PanelItem[] | undefined;
  #sections: SectionItem[] | undefined;
  #sequence: PageSequence | undefined;
  #scene: VizflowPageScene | undefined;
  #edit = false;
  #created: Steps | undefined;
  #mounted: Mounted[] = [];
  #layout: HTMLElement | undefined;
  #generation = 0;
  #pending = false;
  #compiled: CompiledPage | undefined;
  #loading = 0;
  #config: Record<string, string | boolean> = {};
  #settings: PanelSettings = {};

  static readonly observedAttributes = ['src'];

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

  get sections(): SectionItem[] | undefined {
    return this.#sections;
  }

  set sections(value: SectionItem[] | undefined) {
    this.#sections = value;
    this.#schedule();
  }

  get sequence(): PageSequence | undefined {
    return this.#sequence;
  }

  set sequence(value: PageSequence | undefined) {
    this.#sequence = value;
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

  get src(): string | null {
    return this.getAttribute('src');
  }

  set src(value: string | null) {
    if (value === null) this.removeAttribute('src');
    else this.setAttribute('src', value);
  }

  get compiled(): CompiledPage | undefined {
    return this.#compiled;
  }

  attributeChangedCallback(name: string, previous: string | null, value: string | null): void {
    if (name !== 'src' || previous === value || value === null) return;
    this.#config = {};
    this.#settings = {};
    void this.#load(value);
  }

  async #configure(change: Record<string, string | boolean>): Promise<void> {
    const src = this.src;
    if (src === null) return;
    this.#config = { ...this.#config, ...change };
    await this.#load(src, this.#snapshot());
  }

  #snapshot(): Snapshot {
    const params = new Map<string, number[]>();
    for (const n of this.#graph?.nodes ?? []) if (n.kind === 'param') params.set(n.name, Array.from(n.value.data));
    const steps = this.#created;
    const step = steps === undefined ? undefined : steps.list[steps.current];
    return { params, sequence: this.#sequence?.get(), step: step === undefined ? undefined : { id: step.id, dir: step.dir, groups: step.groups } };
  }

  #restore(compiled: CompiledPage, snapshot: Snapshot): void {
    const g = compiled.graph;
    g.batch(() => {
      for (const n of g.nodes) {
        const values = n.kind === 'param' ? snapshot.params.get(n.name) : undefined;
        if (values !== undefined && values.length === n.value.data.length) n.set(values);
      }
    });
    const sequence = (compiled as CompiledPage & { sequence?: PageSequence }).sequence;
    if (sequence !== undefined && snapshot.sequence !== undefined && snapshot.sequence.length === sequence.length) sequence.set(snapshot.sequence);
  }

  #restoreStep(snapshot: Snapshot): void {
    const steps = this.#created;
    const step = snapshot.step;
    if (steps === undefined || step === undefined) return;
    const shared = (groups: string[]) => {
      let k = 0;
      while (k < groups.length && groups[k] === step.groups[k]) k++;
      return k;
    };
    let target = steps.list.findIndex((s) => s.id === step.id && s.dir === step.dir);
    if (target < 0) {
      steps.list.forEach((s, i) => {
        if (s.dir === step.dir && shared(s.groups) > (target < 0 ? 0 : shared(steps.list[target].groups))) target = i;
      });
    }
    if (target >= 0) steps.go(target);
  }

  async #load(src: string, snapshot?: Snapshot): Promise<void> {
    const loading = ++this.#loading;
    try {
      const [{ loadPage }, { autoLayout }] = await Promise.all([import('../dsl'), import('../scene3d')]);
      const compiled = await loadPage(new URL(src, document.baseURI).href, { config: this.#config });
      if (loading !== this.#loading) return;
      if (snapshot !== undefined) this.#restore(compiled, snapshot);
      const { layout, modules, levels } = autoLayout(compiled.tree, compiled.steps, compiled.levels);
      this.#compiled = compiled;
      this.#graph = compiled.graph;
      this.#steps = compiled.steps;
      this.#panels = compiled.panels.length > 0 ? compiled.panels : undefined;
      this.#sections = compiled.sections.length > 0 ? compiled.sections : undefined;
      this.#sequence = (compiled as CompiledPage & { sequence?: PageSequence }).sequence;
      this.#scene = { layout, modules, levels, sequence: this.#sequence, camera: compiled.page.camera };
      this.#schedule();
      this.#flush();
      if (snapshot !== undefined) this.#restoreStep(snapshot);
      this.dispatchEvent(new Event('load'));
    } catch (error) {
      if (loading !== this.#loading) return;
      const message = element('p', 'vf-load-error');
      message.textContent = `${src}: ${(error as Error).message}`;
      this.replaceChildren(message);
      reportError(error);
    }
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
    const sections = this.#sections;
    const sectionsBox = element('div', 'vf-article');
    if (sections !== undefined) main.append(sectionsBox);
    const panels = this.#panels;
    const side = element('aside', 'vf-side');
    const layout = element('div', panels === undefined ? 'vf-stack' : 'vf-split');
    layout.append(main);
    if (panels !== undefined) layout.append(side);
    this.#layout = layout;
    const header = this.#header();
    this.replaceChildren(...(header === undefined ? [] : [header]), layout);
    this.#mounted.push(mountSteps(stepsBox, steps));
    const configure = this.src === null || this.#graph !== this.#compiled?.graph ? undefined : (change: Record<string, string | boolean>) => void this.#configure(change);
    if (panels !== undefined) this.#mounted.push(mountPanels(side, g, panels, { steps, sequence: this.#sequence, settings: this.#settings, ...(configure === undefined ? {} : { configure }) }));
    if (sections !== undefined) {
      const onJump = () => stage.scrollIntoView({ behavior: 'smooth', block: 'start' });
      this.#mounted.push(mountSections(sectionsBox, g, sections, { steps, onJump }));
    }
    if (this.#scene !== undefined) void this.#mountScene(stage, g, { ...this.#scene, steps, edit: this.#edit });
  }

  #header(): HTMLElement | undefined {
    const page = this.#compiled?.page;
    if (page === undefined || this.#graph !== this.#compiled?.graph) return undefined;
    const header = element('header', 'vf-header');
    const title = element('h1', 'vf-title');
    title.textContent = page.title;
    header.append(title);
    return header;
  }

  async #mountScene(stage: HTMLElement, g: Graph, options: SceneOptions): Promise<void> {
    const generation = this.#generation;
    const { mountScene } = await import('../scene3d');
    if (generation !== this.#generation) return;
    try {
      this.#mounted.push(mountScene(stage, g, options));
    } catch (error) {
      this.#unmount();
      reportError(error);
    }
  }

  #unmount(): void {
    this.#generation++;
    for (const mounted of this.#mounted.splice(0)) mounted.dispose();
    this.#created?.dispose();
    this.#created = undefined;
    this.#layout?.remove();
    this.#layout = undefined;
    this.querySelector(':scope > .vf-header')?.remove();
  }
}

if (customElements.get('vizflow-page') === undefined) customElements.define('vizflow-page', VizflowPage);

declare global {
  interface HTMLElementTagNameMap {
    'vizflow-page': VizflowPage;
  }
}
