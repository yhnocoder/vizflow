import type { Graph, Node } from '../autograd';
import { explain } from '../explain';
import { compileTemplate } from './template';

export type StepKind = 'overview' | 'inputs' | 'op' | 'module';

export interface BackwardOverride {
  title?: string;
  text?: string;
  label?: string;
}

export interface ForwardStep {
  show: Node | Node[];
  active?: Node | Node[];
  title: string;
  text?: string;
  label?: string;
  kind?: 'inputs' | 'op';
  segment?: string;
  sub?: string;
  module?: string;
  brace?: { title: string; subtitle?: string; formula?: string };
  backward?: BackwardOverride;
  marks?: string[];
}

export type BackwardSpec = 'auto' | 'reverse' | { auto: true; merge?: Node[][]; override?: Record<number, BackwardOverride> };

export interface PageModule {
  id: string;
  parent: string | null;
  title: string;
  subtitle: string;
  formula?: string;
  inputs: Node[];
  outputs: Node[];
  nodes: Node[];
  params: Node[];
  fixed: boolean;
  folded: boolean;
}

export interface StepModule extends PageModule {
  segment: string;
  sub?: string;
  forward: BackwardOverride;
  backward: BackwardOverride;
}

export interface StepsSpec {
  forward: ForwardStep[];
  overview?: { title: string; text?: string; label?: string; backward?: BackwardOverride };
  backward?: BackwardSpec;
  modules?: StepModule[];
}

export interface StepInfo {
  dir: 'forward' | 'backward';
  index: number;
  id: string;
  kind: StepKind;
  label: string;
  title: string;
  text: string;
  nodes: Node[];
  segment: string;
  sub?: string;
  module?: string;
  marks?: string[];
}

export interface StepState {
  dir: 'forward' | 'backward';
  index: number;
  kind: StepKind;
  focus: number;
  revealed(node: Node): boolean;
  active(node: Node): boolean;
}

export interface Steps {
  readonly graph: Graph;
  readonly list: readonly StepInfo[];
  readonly visible: readonly Node[];
  readonly modules: readonly PageModule[];
  readonly current: number;
  readonly focus: number;
  go(i: number): void;
  next(): void;
  prev(): void;
  setFocus(b: number): void;
  state(): StepState;
  subscribe(fn: () => void): () => void;
  text(i?: number): { title: string; body: string };
  folded(id: string): boolean;
  setFolded(id: string, folded: boolean): void;
  setAllFolded(folded: boolean): void;
  dispose(): void;
}

const DEFAULT_OVERVIEW_TITLE = '总览';

const toList = (nodes: Node | Node[] | undefined): Node[] => (nodes === undefined ? [] : Array.isArray(nodes) ? nodes : [nodes]);

function byId(nodes: Iterable<Node>): Node[] {
  return [...new Set(nodes)].sort((a, b) => a.id - b.id);
}

function unique(nodes: Iterable<Node>): Node[] {
  return [...new Set(nodes)];
}

interface Entry {
  info: StepInfo;
  revealed: Set<Node>;
  active: Set<Node>;
}

type Template = (focus: number) => string;

export function backwardGroups(g: Graph, visible: readonly Node[], spec: Exclude<BackwardSpec, 'auto' | 'reverse'>): Node[][] {
  const loss = g.findLoss();
  if (loss === undefined) {
    if ((spec.merge?.length ?? 0) > 0 || Object.keys(spec.override ?? {}).length > 0) {
      throw new Error('backward: the graph has no loss node, so merge and override cannot apply');
    }
    return [];
  }
  const ordered = visible.filter((n) => n.grad !== null && n.id <= loss.id).sort((a, b) => b.id - a.id);
  const groups: Node[][] = [];
  ordered.forEach((node, i) => {
    if (i === 1 && ordered[0] === loss) groups[0].push(node);
    else groups.push([node]);
  });
  for (const group of spec.merge ?? []) {
    const positions = group.map((node) => {
      const at = groups.findIndex((step) => step.includes(node));
      if (at < 0) throw new Error(`merge: node '${node.name}' is not in any backward step`);
      return at;
    });
    const target = Math.min(...positions);
    for (const node of group) {
      const at = groups.findIndex((step) => step.includes(node));
      if (at === target) continue;
      groups[at] = groups[at].filter((n) => n !== node);
      groups[target].push(node);
    }
  }
  return groups.filter((step) => step.length > 0).map((step) => step.sort((a, b) => b.id - a.id));
}

function backwardEntries(g: Graph, visible: readonly Node[], spec: Exclude<BackwardSpec, 'reverse'>): StepInfo[] {
  const resolved = spec === 'auto' ? { auto: true as const } : spec;
  const loss = g.findLoss();
  const explainer = explain(g);
  const infos = backwardGroups(g, visible, resolved).map((nodes, i): StepInfo => {
    const subject = nodes.find((n) => n !== loss) ?? nodes[0];
    const card = explainer.nodeCard(subject, 'backward');
    const override = resolved.override?.[i + 1] ?? {};
    const title = override.title ?? `$\\partial\\ell/\\partial ${subject.tex}$`;
    return {
      dir: 'backward',
      index: i + 1,
      id: `b:${i + 1}`,
      kind: 'op',
      title,
      label: override.label ?? `$\\partial ${subject.tex}$`,
      text: override.text ?? card.tex.map((tex) => `$$${tex}$$`).join(''),
      nodes,
      segment: title,
    };
  });
  for (const key of Object.keys(resolved.override ?? {})) {
    const index = Number(key);
    if (!Number.isInteger(index) || index < 1 || index > infos.length) {
      throw new Error(`override: backward step ${key} does not exist (there are ${infos.length} steps)`);
    }
  }
  return infos;
}

export function isLeafLike(node: Node): boolean {
  return node.kind !== 'op' || (node.op === 'transpose' && node.inputs[0].kind === 'param');
}

function usesOf(prod: readonly Node[]): Node[] {
  return unique(prod.flatMap((n) => n.inputs).flatMap((n) => (n.kind === 'op' && n.op === 'transpose' && n.inputs[0].kind === 'param' ? [n, n.inputs[0]] : [n])));
}

interface Unit {
  index: number;
  step: ForwardStep;
  prod: Node[];
  use: Node[];
  active: Node[];
}

class StepsImpl implements Steps {
  readonly visible: readonly Node[];
  readonly modules: readonly PageModule[];
  current = 0;
  focus = 0;
  private entries: Entry[] = [];
  private infos: readonly StepInfo[] = [];
  private readonly templates = new Map<string, { title: Template; body: Template }>();
  private readonly subscribers: (() => void)[] = [];
  private readonly byModule = new Map<string, StepModule>();
  private readonly foldState = new Map<string, boolean>();
  private readonly units: Unit[] = [];
  private readonly reverse: boolean;

  constructor(
    readonly graph: Graph,
    private readonly spec: StepsSpec,
  ) {
    this.reverse = spec.backward === 'reverse';
    this.modules = Object.freeze([...(spec.modules ?? [])]);
    for (const module of spec.modules ?? []) {
      this.byModule.set(module.id, module);
      this.foldState.set(module.id, module.fixed || module.folded);
    }
    this.visible = Object.freeze(byId(spec.forward.flatMap((step) => toList(step.show))));
    if (this.reverse) {
      spec.forward.forEach((step, index) => {
        if (step.kind === 'inputs') return;
        const show = toList(step.show);
        const computed = show.filter((n) => !isLeafLike(n));
        const prod = computed.length > 0 ? computed : show;
        const use = usesOf(prod).filter((n) => !prod.includes(n));
        this.units.push({ index, step, prod, use, active: step.active === undefined ? unique([...prod, ...use]) : toList(step.active) });
      });
      this.rebuild();
    } else {
      this.buildClassic();
    }
  }

  get list(): readonly StepInfo[] {
    return this.infos;
  }

  private overviewSpec(): { title: string; text: string; label: string } {
    const overview = this.spec.overview ?? { title: DEFAULT_OVERVIEW_TITLE, text: '' };
    return { title: overview.title, text: overview.text ?? '', label: overview.label ?? overview.title };
  }

  private buildClassic(): void {
    const g = this.graph;
    const spec = this.spec;
    const entries: Entry[] = [];
    const overview = this.overviewSpec();
    const all = new Set(this.visible);
    entries.push(this.entry({ dir: 'forward', index: 0, id: 'f:overview', kind: 'overview', title: overview.title, label: overview.label, text: overview.text, nodes: [...this.visible], segment: overview.label }, all, all));
    const shown = new Set<Node>();
    spec.forward.forEach((step, k) => {
      const show = toList(step.show);
      show.forEach((n) => shown.add(n));
      const active = new Set(step.active === undefined ? show : toList(step.active));
      const info: StepInfo = {
        dir: 'forward',
        index: k + 1,
        id: `f:${k}`,
        kind: step.kind ?? 'op',
        title: step.title,
        label: step.label ?? step.title,
        text: step.text ?? '',
        nodes: show,
        segment: step.segment ?? step.label ?? step.title,
      };
      if (step.sub !== undefined) info.sub = step.sub;
      if (step.module !== undefined) info.module = step.module;
      if (step.marks !== undefined) info.marks = step.marks;
      entries.push(this.entry(info, new Set(shown), active));
    });
    const revealed = new Set<Node>();
    for (const info of backwardEntries(g, this.visible, (spec.backward ?? 'auto') as Exclude<BackwardSpec, 'reverse'>)) {
      info.nodes.forEach((n) => revealed.add(n));
      entries.push(this.entry(info, new Set(revealed), new Set(info.nodes)));
    }
    this.setEntries(entries);
  }

  private outerFolded(id: string | undefined): StepModule | undefined {
    let found: StepModule | undefined;
    for (let at = id; at !== undefined && at !== null; at = this.byModule.get(at)?.parent ?? undefined) {
      const module = this.byModule.get(at);
      if (module === undefined) break;
      if (this.foldState.get(at)) found = module;
    }
    return found;
  }

  private rebuild(): void {
    const spec = this.spec;
    const g = this.graph;
    const explainer = explain(g);
    const loss = g.findLoss();
    const all = new Set(this.visible);
    const overview = this.overviewSpec();
    const entries: Entry[] = [];
    entries.push(this.entry({ dir: 'forward', index: 0, id: 'f:overview', kind: 'overview', title: overview.title, label: overview.label, text: overview.text, nodes: [...this.visible], segment: overview.label }, all, all));
    const revealed = new Set<Node>();
    const inputs = spec.forward.find((step) => step.kind === 'inputs');
    if (inputs !== undefined) {
      const show = toList(inputs.show);
      show.forEach((n) => revealed.add(n));
      const label = inputs.label ?? inputs.title;
      entries.push(this.entry({ dir: 'forward', index: 1, id: 'f:inputs', kind: 'inputs', title: inputs.title, label, text: inputs.text ?? '', nodes: show, segment: inputs.segment ?? label }, new Set(revealed), new Set(show)));
    }
    const groups: { unit?: Unit; module?: StepModule }[] = [];
    for (const unit of this.units) {
      const folded = this.outerFolded(unit.step.module);
      if (folded === undefined) groups.push({ unit });
      else if (groups[groups.length - 1]?.module !== folded) groups.push({ module: folded });
    }
    const moduleUse = (m: StepModule) => unique([...m.inputs, ...m.params]);
    for (const group of groups) {
      const index = entries.length;
      let info: StepInfo;
      let prod: Node[];
      let active: Node[];
      if (group.module !== undefined) {
        const m = group.module;
        prod = m.outputs;
        active = unique([...prod, ...moduleUse(m)]);
        info = {
          dir: 'forward',
          index,
          id: `F:${m.id}`,
          kind: 'module',
          title: m.forward.title ?? `${m.title}（已折叠）`,
          label: m.forward.label ?? m.title,
          text: m.forward.text ?? '',
          nodes: prod,
          segment: m.segment,
          module: m.id,
        };
        if (m.sub !== undefined) info.sub = m.sub;
        const marks = this.units.filter((u) => this.outerFolded(u.step.module) === m).flatMap((u) => u.step.marks ?? []);
        if (marks.length > 0) info.marks = marks;
      } else {
        const unit = group.unit as Unit;
        const step = unit.step;
        prod = unit.prod;
        active = unit.active;
        info = {
          dir: 'forward',
          index,
          id: `f:${unit.index}`,
          kind: 'op',
          title: step.title,
          label: step.label ?? step.title,
          text: step.text ?? '',
          nodes: prod,
          segment: step.segment ?? step.label ?? step.title,
        };
        if (step.sub !== undefined) info.sub = step.sub;
        if (step.module !== undefined) info.module = step.module;
        if (step.marks !== undefined) info.marks = step.marks;
      }
      prod.forEach((n) => revealed.add(n));
      entries.push(this.entry(info, new Set(revealed), new Set(active)));
    }
    const received = new Set<Node>();
    let position = 0;
    const pushBackward = (info: Omit<StepInfo, 'index' | 'dir'>, recv: Node[]) => {
      const nodes = position === 0 && loss !== undefined && !recv.includes(loss) ? [loss, ...recv] : recv;
      position++;
      nodes.forEach((n) => received.add(n));
      entries.push(this.entry({ ...info, nodes, dir: 'backward', index: position }, new Set(received), new Set(nodes)));
    };
    const hasGrad = (n: Node) => n.grad !== null && all.has(n);
    for (const group of [...groups].reverse()) {
      if (group.module !== undefined) {
        const m = group.module;
        const info: Omit<StepInfo, 'index' | 'dir'> = {
          id: `B:${m.id}`,
          kind: 'module',
          title: m.backward.title ?? `穿过 ${m.title}（已折叠）`,
          label: m.backward.label ?? `$\\partial$ ${m.title}`,
          text: m.backward.text ?? '',
          nodes: [],
          segment: m.segment,
          module: m.id,
        };
        if (m.sub !== undefined) info.sub = m.sub;
        pushBackward(info, moduleUse(m).filter(hasGrad));
        continue;
      }
      const unit = group.unit as Unit;
      const recv = unit.use.filter(hasGrad);
      if (recv.length === 0 && position > 0) continue;
      const subject = recv[0] ?? unit.prod[0];
      const override = unit.step.backward ?? {};
      const title = override.title ?? `$\\partial\\ell/\\partial ${subject.tex}$`;
      const info: Omit<StepInfo, 'index' | 'dir'> = {
        id: `b:${unit.index}`,
        kind: 'op',
        title,
        label: override.label ?? `$\\partial ${subject.tex}$`,
        text: override.text ?? (override.title === undefined ? explainer.nodeCard(subject, 'backward').tex.map((tex) => `$$${tex}$$`).join('') : ''),
        nodes: [],
        segment: unit.step.segment ?? unit.step.label ?? unit.step.title,
      };
      if (unit.step.sub !== undefined) info.sub = unit.step.sub;
      if (unit.step.module !== undefined) info.module = unit.step.module;
      pushBackward(info, recv);
    }
    const back = spec.overview?.backward;
    if (back !== undefined) {
      const backLabel = back.label ?? overview.label;
      entries.push(
        this.entry(
          { dir: 'backward', index: position + 1, id: 'b:overview', kind: 'overview', title: back.title ?? overview.title, label: backLabel, text: back.text ?? '', nodes: [...this.visible], segment: backLabel },
          all,
          all,
        ),
      );
    }
    this.setEntries(entries);
  }

  private setEntries(entries: Entry[]): void {
    this.entries = entries;
    this.infos = Object.freeze(entries.map((e) => e.info));
    entries.forEach((_, i) => this.template(i));
  }

  private entry(info: StepInfo, revealed: Set<Node>, active: Set<Node>): Entry {
    return { info, revealed, active };
  }

  private template(i: number): { title: Template; body: Template } {
    const info = this.entries[i].info;
    let compiled = this.templates.get(info.id);
    if (compiled === undefined) {
      compiled = { title: compileTemplate(this.graph, info.title), body: compileTemplate(this.graph, info.text) };
      this.templates.set(info.id, compiled);
    }
    return compiled;
  }

  go(i: number): void {
    const next = Math.max(0, Math.min(this.entries.length - 1, i));
    if (next === this.current) return;
    this.current = next;
    this.notify();
  }

  next(): void {
    this.go(this.current + 1);
  }

  prev(): void {
    this.go(this.current - 1);
  }

  setFocus(b: number): void {
    const rows = Math.max(1, ...this.visible.map((n) => (n.shape.length === 0 ? 1 : n.shape[0])));
    if (!Number.isInteger(b) || b < 0 || b >= rows) {
      throw new Error(`setFocus: ${b} is out of range; the visible nodes have ${rows} row(s), so b must be between 0 and ${rows - 1}`);
    }
    if (b === this.focus) return;
    this.focus = b;
    this.notify();
  }

  state(): StepState {
    const entry = this.entries[this.current];
    return {
      dir: entry.info.dir,
      index: entry.info.index,
      kind: entry.info.kind,
      focus: this.focus,
      revealed: (node) => entry.revealed.has(node),
      active: (node) => entry.active.has(node),
    };
  }

  subscribe(fn: () => void): () => void {
    this.subscribers.push(fn);
    return () => {
      const at = this.subscribers.indexOf(fn);
      if (at >= 0) this.subscribers.splice(at, 1);
    };
  }

  text(i: number = this.current): { title: string; body: string } {
    if (this.entries[i] === undefined) throw new Error(`step ${i} does not exist (there are ${this.entries.length} steps)`);
    const compiled = this.template(i);
    return { title: compiled.title(this.focus), body: compiled.body(this.focus) };
  }

  folded(id: string): boolean {
    const state = this.foldState.get(id);
    if (state === undefined) throw new Error(`folded: module '${id}' does not exist`);
    return state;
  }

  setFolded(id: string, folded: boolean): void {
    const module = this.byModule.get(id);
    if (module === undefined) throw new Error(`setFolded: module '${id}' does not exist`);
    if (module.fixed || this.foldState.get(id) === folded) return;
    this.foldState.set(id, folded);
    this.refold();
  }

  setAllFolded(folded: boolean): void {
    let changed = false;
    for (const module of this.byModule.values()) {
      if (module.fixed || this.foldState.get(module.id) === folded) continue;
      this.foldState.set(module.id, folded);
      changed = true;
    }
    if (changed) this.refold();
  }

  private refold(): void {
    if (!this.reverse) return this.notify();
    const before = this.entries[this.current];
    this.rebuild();
    const after = this.entries;
    const prev = before.info;
    const overlaps = (e: Entry) => e.info.nodes.some((n) => prev.nodes.includes(n));
    let j = after.findIndex((e) => e.info.id === prev.id);
    if (j < 0) j = after.findIndex((e) => e.info.dir === prev.dir && e.info.kind !== 'overview' && overlaps(e));
    const enclosing: string[] = [];
    for (let at = prev.module; at !== undefined; at = this.byModule.get(at)?.parent ?? undefined) enclosing.push(at);
    if (j < 0) j = after.findIndex((e) => e.info.dir === prev.dir && e.info.kind === 'module' && enclosing.includes(e.info.module as string));
    if (j < 0) j = after.findIndex((e) => e.info.dir === prev.dir);
    this.current = Math.max(0, j);
    this.notify();
  }

  dispose(): void {
    this.subscribers.length = 0;
  }

  private notify(): void {
    [...this.subscribers].forEach((fn) => fn());
  }
}

export function createSteps(g: Graph, spec: StepsSpec): Steps {
  return new StepsImpl(g, spec);
}

export function allVisible(_g: Graph): StepState {
  return { dir: 'forward', index: 0, kind: 'overview', focus: 0, revealed: () => true, active: () => true };
}
