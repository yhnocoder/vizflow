import type { Graph } from '../autograd';
import type { CompiledPage } from '../dsl';
import { renderText } from '../formula';
import type { SceneOptions } from '../scene3d';
import { createSteps, type Steps, type StepsSpec } from '../steps';
import { mountPanels, mountSections, mountSteps, type Mounted, type PageSequence, type PanelItem, type SectionItem } from '../ui';
import elementCss from './element.css?inline';

const styleId = 'vf-element-style';

if (!document.getElementById(styleId)) {
  const style = document.createElement('style');
  style.id = styleId;
  style.textContent = elementCss;
  document.head.append(style);
}

export type VizflowPageScene = Omit<SceneOptions, 'steps' | 'edit'>;

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
  #mode: 'train' | 'infer' = 'infer';

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
    if (name === 'src' && previous !== value && value !== null) void this.#load(value);
  }

  async #load(src: string): Promise<void> {
    const loading = ++this.#loading;
    try {
      const [{ loadPage }, { autoLayout }] = await Promise.all([import('../dsl'), import('../scene3d')]);
      const compiled = await loadPage(new URL(src, document.baseURI).href);
      if (loading !== this.#loading) return;
      const { layout, modules, levels } = autoLayout(compiled.tree, compiled.steps);
      this.#compiled = compiled;
      this.#graph = compiled.graph;
      this.#steps = compiled.steps;
      this.#mode = compiled.modes === undefined ? 'train' : 'infer';
      this.#panels = this.#panelsFor(this.#mode);
      this.#sections = compiled.sections.length > 0 ? compiled.sections : undefined;
      this.#sequence = (compiled as CompiledPage & { sequence?: PageSequence }).sequence;
      this.#scene = { layout, modules, levels, sequence: this.#sequence, hint: compiled.page.hint, camera: compiled.page.camera };
      this.#schedule();
      this.#flush();
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
    if (panels !== undefined) this.#mounted.push(mountPanels(side, g, panels, { steps, sequence: this.#sequence }));
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
    if (page.lede !== undefined) {
      const lede = element('p', 'vf-lede');
      renderText(page.lede, lede);
      header.append(lede);
    }
    const modes = this.#compiled?.modes;
    if (modes !== undefined) header.append(this.#modeSwitch(modes));
    return header;
  }

  get mode(): 'train' | 'infer' {
    return this.#mode;
  }

  set mode(value: 'train' | 'infer') {
    const modes = this.#compiled?.modes;
    if (modes === undefined || value === this.#mode) return;
    this.#mode = value;
    this.#steps = modes[value];
    this.#panels = this.#panelsFor(value);
    this.#schedule();
  }

  #panelsFor(mode: 'train' | 'infer'): PanelItem[] | undefined {
    const compiled = this.#compiled;
    if (compiled === undefined) return undefined;
    const panels = compiled.modePanels?.[mode] ?? compiled.panels;
    return panels.length > 0 ? panels : undefined;
  }

  #modeSwitch(modes: NonNullable<CompiledPage['modes']>): HTMLElement {
    const box = element('div', 'vf-modes');
    box.setAttribute('role', 'group');
    box.setAttribute('aria-label', '模式');
    for (const [mode, text] of [['train', '训练'], ['infer', '推理']] as const) {
      const button = element('button', 'vf-mode') as HTMLButtonElement;
      button.type = 'button';
      button.textContent = text;
      button.setAttribute('aria-pressed', String(mode === this.#mode));
      button.disabled = modes[mode] === undefined;
      button.addEventListener('click', () => {
        this.mode = mode;
      });
      box.append(button);
    }
    return box;
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
