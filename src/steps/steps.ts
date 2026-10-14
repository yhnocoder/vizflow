import type { Graph, Node } from '../autograd';
import { explain } from '../explain';
import { compileTemplate } from './template';

export type StepKind = 'overview' | 'inputs' | 'op' | 'module';

export interface BackwardOverride {
  title?: string;
  text?: string;
  label?: string;
}

export type NoteSide = 'left' | 'right' | 'above' | 'below';

export interface StepNote {
  nodes: Node[];
  index?: number[];
  text: string;
  side?: NoteSide;
}

export interface ForwardStep {
  show: Node | Node[];
  notes?: StepNote[];
  active?: Node | Node[];
  title: string;
  text?: string;
  label?: string;
  groups?: string[];
  module?: string;
  formula?: { title: string; tex: string };
  backward?: BackwardOverride;
  marks?: string[];
}

export interface StepGroup {
  id: string;
  title: string;
}

export interface StepBrace {
  id: string;
  title: string;
  subtitle?: string;
  nodes: Node[];
  depth: number;
  module?: string;
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
  folded: boolean;
  foldable: boolean;
}

export interface StepModule extends PageModule {
  forward: BackwardOverride;
  backward: BackwardOverride;
}

export interface FunctionStep {
  id: string;
  title: string;
  label?: string;
  text?: string;
  show: Node[];
  receive: Node[];
  groups?: string[];
  notes?: StepNote[];
  formula?: { title: string; tex: string };
  marks?: string[];
}

export interface FunctionBackward {
  module: string;
  after: ForwardStep;
  steps: FunctionStep[];
}

export interface StepsSpec {
  forward: ForwardStep[];
  functions?: FunctionBackward[];
  groups?: StepGroup[];
  braces?: StepBrace[];
  overview?: { title: string; text?: string; label?: string; backward?: BackwardOverride; notes?: StepNote[] };
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
  groups: string[];
  module?: string;
  forwardId?: string;
  computes?: Node[];
  notes?: StepNote[];
  marks?: string[];
  formula?: { title: string; tex: string };
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
  readonly groups: readonly StepGroup[];
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

function withMarks(info: StepInfo, step: ForwardStep): StepInfo {
  if (step.marks !== undefined) info.marks = step.marks;
  if (step.formula !== undefined) info.formula = step.formula;
  return info;
}

function withNotes(info: StepInfo, notes: StepNote[] | undefined): StepInfo {
  if (notes !== undefined && notes.length > 0) info.notes = notes;
  return info;
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
      groups: [],
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
  kind: StepKind;
  prod: Node[];
  use: Node[];
  active: Node[];
}

class StepsImpl implements Steps {
  readonly visible: readonly Node[];
  readonly modules: readonly PageModule[];
  readonly groups: readonly StepGroup[];
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
  private readonly forwardVisible: ReadonlySet<Node>;

  constructor(
    readonly graph: Graph,
    private readonly spec: StepsSpec,
  ) {
    this.reverse = spec.backward === 'reverse';
    this.modules = Object.freeze([...(spec.modules ?? [])]);
    this.groups = Object.freeze([...(spec.groups ?? [])]);
    for (const module of spec.modules ?? []) {
      this.byModule.set(module.id, module);
      this.foldState.set(module.id, module.folded);
    }
    this.visible = Object.freeze(byId([...spec.forward.flatMap((step) => toList(step.show)), ...(spec.functions ?? []).flatMap((f) => f.steps.flatMap((step) => step.show))]));
    this.forwardVisible = new Set(spec.forward.flatMap((step) => toList(step.show)));
    if (this.reverse) {
      spec.forward.forEach((step, index) => {
        const show = toList(step.show);
        const kind: StepKind = show.length > 0 && [...this.forwardVisible].every((n) => show.includes(n)) ? 'overview' : show.length > 0 && show.every(isLeafLike) ? 'inputs' : 'op';
        const computed = show.filter((n) => !isLeafLike(n));
        const prod = kind === 'op' && computed.length > 0 ? computed : show;
        const use = kind === 'op' ? usesOf(prod).filter((n) => !prod.includes(n)) : [];
        this.units.push({ index, step, kind, prod, use, active: step.active === undefined ? unique([...prod, ...use]) : toList(step.active) });
      });
      this.rebuild();
    } else {
      this.buildClassic();
    }
  }

  get list(): readonly StepInfo[] {
    return this.infos;
  }

  private overviewSpec(): { title: string; text: string; label: string; notes?: StepNote[] } {
    const overview = this.spec.overview ?? { title: DEFAULT_OVERVIEW_TITLE, text: '' };
    const out: { title: string; text: string; label: string; notes?: StepNote[] } = { title: overview.title, text: overview.text ?? '', label: overview.label ?? overview.title };
    if (overview.notes !== undefined) out.notes = overview.notes;
    return out;
  }

  private buildClassic(): void {
    const g = this.graph;
    const spec = this.spec;
    const entries: Entry[] = [];
    const overview = this.overviewSpec();
    const all = new Set(this.visible);
    entries.push(this.entry(withNotes({ dir: 'forward', index: 0, id: 'f:overview', kind: 'overview', title: overview.title, label: overview.label, text: overview.text, nodes: [...this.visible], groups: [] }, overview.notes), all, all));
    const shown = new Set<Node>();
    spec.forward.forEach((step, k) => {
      const show = toList(step.show);
      show.forEach((n) => shown.add(n));
      const active = new Set(step.active === undefined ? show : toList(step.active));
      const info: StepInfo = {
        dir: 'forward',
        index: k + 1,
        id: `f:${k}`,
        kind: 'op',
        title: step.title,
        label: step.label ?? step.title,
        text: step.text ?? '',
        nodes: show,
        groups: step.groups ?? [],
      };
      if (step.module !== undefined) info.module = step.module;
      entries.push(this.entry(withNotes(withMarks(info, step), step.notes), new Set(shown), active));
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
    const g = this.graph;
    const explainer = explain(g);
    const loss = g.findLoss();
    const all = new Set(this.visible);
    const forwardAll = new Set(this.forwardVisible);
    const entries: Entry[] = [];
    const revealed = new Set<Node>();
    const groups: { unit?: Unit; module?: StepModule }[] = [];
    for (const unit of this.units) {
      const folded = unit.kind === 'op' ? this.outerFolded(unit.step.module) : undefined;
      if (folded === undefined) groups.push({ unit });
      else if (groups[groups.length - 1]?.module !== folded) groups.push({ module: folded });
    }
    const moduleUse = (m: StepModule) => unique([...m.inputs, ...m.params]);
    const forwardIds = new Map<(typeof groups)[number], string>();
    const groupsOf = (m: StepModule): string[] => {
      const first = this.units.find((u) => this.outerFolded(u.step.module) === m)?.step.groups ?? [];
      const own = first.indexOf(m.id);
      return own < 0 ? first : first.slice(0, own + 1);
    };
    for (const group of groups) {
      const index = entries.length;
      if (group.module !== undefined) {
        const m = group.module;
        const prod = m.outputs;
        const active = unique([...prod, ...moduleUse(m)]);
        const info: StepInfo = {
          dir: 'forward',
          index,
          id: `F:${m.id}`,
          kind: 'module',
          title: m.forward.title ?? `${m.title}（已折叠）`,
          label: m.forward.label ?? m.title,
          text: m.forward.text ?? '',
          nodes: prod,
          groups: groupsOf(m),
          module: m.id,
        };
        const marks = this.units.filter((u) => this.outerFolded(u.step.module) === m).flatMap((u) => u.step.marks ?? []);
        if (marks.length > 0) info.marks = marks;
        prod.forEach((n) => revealed.add(n));
        active.filter(isLeafLike).forEach((n) => revealed.add(n));
        forwardIds.set(group, info.id);
        entries.push(this.entry(info, new Set(revealed), new Set(active)));
        continue;
      }
      const unit = group.unit as Unit;
      const step = unit.step;
      const info: StepInfo = {
        dir: 'forward',
        index,
        id: `f:${unit.index}`,
        kind: unit.kind,
        title: step.title,
        label: step.label ?? step.title,
        text: step.text ?? '',
        nodes: unit.kind === 'overview' ? [...this.forwardVisible] : unit.prod,
        groups: step.groups ?? [],
      };
      if (step.module !== undefined) info.module = step.module;
      withNotes(withMarks(info, step), step.notes);
      forwardIds.set(group, info.id);
      if (unit.kind === 'overview') {
        entries.push(this.entry(info, forwardAll, forwardAll));
        continue;
      }
      unit.prod.forEach((n) => revealed.add(n));
      unit.active.filter(isLeafLike).forEach((n) => revealed.add(n));
      entries.push(this.entry(info, new Set(revealed), new Set(unit.active)));
    }
    const received = new Set<Node>();
    let position = 0;
    const pushBackward = (info: Omit<StepInfo, 'index' | 'dir'>, recv: Node[], lit: Node[] = recv) => {
      const nodes = position === 0 && loss !== undefined && !recv.includes(loss) ? [loss, ...recv] : recv;
      position++;
      nodes.forEach((n) => received.add(n));
      lit.filter((n) => n.phase === 'backward' && all.has(n)).forEach((n) => received.add(n));
      entries.push(this.entry({ ...info, nodes, dir: 'backward', index: position }, new Set(received), new Set([...nodes, ...lit])));
    };
    const hasGrad = (n: Node) => n.grad !== null && all.has(n);
    for (const group of [...groups].reverse()) {
      if (group.module !== undefined) {
        const m = group.module;
        const recv = moduleUse(m).filter(hasGrad);
        if ((recv.length === 0 || !m.outputs.some(hasGrad)) && position > 0) continue;
        pushBackward(
          {
            id: `B:${m.id}`,
            forwardId: forwardIds.get(group),
            kind: 'module',
            title: m.backward.title ?? `穿过 ${m.title}（已折叠）`,
            label: m.backward.label ?? `$\\partial$ ${m.title}`,
            text: m.backward.text ?? '',
            nodes: [],
            groups: groupsOf(m),
            module: m.id,
          },
          recv,
        );
        continue;
      }
      const unit = group.unit as Unit;
      for (const fn of this.spec.functions ?? []) {
        if (fn.after !== unit.step) continue;
        fn.steps.forEach((step, k) => {
          const info: Omit<StepInfo, 'index' | 'dir'> = {
            id: `x:${fn.module}:${k}`,
            kind: 'op',
            title: step.title,
            label: step.label ?? step.title,
            text: step.text ?? '',
            nodes: [],
            groups: step.groups ?? unit.step.groups ?? [],
            module: fn.module,
            computes: step.show,
          };
          if (step.formula !== undefined) info.formula = step.formula;
          if (step.marks !== undefined) info.marks = step.marks;
          if (step.notes !== undefined && step.notes.length > 0) info.notes = step.notes;
          pushBackward(info, unique([...step.show, ...step.receive]), unique([...step.show, ...step.show.flatMap((n) => n.inputs), ...step.receive]));
        });
      }
      if (unit.kind !== 'op') continue;
      const recv = unit.use.filter(hasGrad);
      if ((recv.length === 0 || !unit.prod.some(hasGrad)) && position > 0) continue;
      const subject = recv[0] ?? unit.prod[0];
      if (subject === undefined) continue;
      const override = unit.step.backward ?? {};
      const title = override.title ?? `$\\partial\\ell/\\partial ${subject.tex}$`;
      const info: Omit<StepInfo, 'index' | 'dir'> = {
        id: `b:${unit.index}`,
        forwardId: forwardIds.get(group),
        kind: 'op',
        title,
        label: override.label ?? `$\\partial ${subject.tex}$`,
        text: override.text ?? (override.title === undefined ? explainer.nodeCard(subject, 'backward').tex.map((tex) => `$$${tex}$$`).join('') : ''),
        nodes: [],
        groups: unit.step.groups ?? [],
      };
      if (unit.step.module !== undefined) info.module = unit.step.module;
      pushBackward(info, recv);
    }
    for (const unit of this.units) {
      const back = unit.step.backward;
      if (unit.kind !== 'overview' || back === undefined) continue;
      const label = back.label ?? unit.step.label ?? unit.step.title;
      entries.push(
        this.entry({ dir: 'backward', index: ++position, id: `b:${unit.index}`, forwardId: `f:${unit.index}`, kind: 'overview', title: back.title ?? unit.step.title, label, text: back.text ?? '', nodes: [...this.visible], groups: [] }, all, all),
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
    if (this.foldState.get(id) === folded) return;
    this.foldState.set(id, folded);
    this.refold();
  }

  setAllFolded(folded: boolean): void {
    let changed = false;
    for (const module of this.byModule.values()) {
      if (!module.foldable || this.foldState.get(module.id) === folded) continue;
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
