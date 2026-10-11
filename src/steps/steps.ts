import type { Graph, Node } from '../autograd';
import { type Nested, offsetOf } from '../tensor';
import { type Positions, explain } from '../explain';
import { type CacheDecl, type CacheKind, analyzeCache } from './cache';
import { type SampleDraw, type SamplerSpec } from './sampling';
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
  kind?: 'inputs' | 'op';
  segment?: string;
  sub?: string;
  module?: string;
  brace?: { title: string; subtitle?: string; formula?: string };
  backward?: BackwardOverride;
  marks?: string[];
}

export type BackwardSpec = 'auto' | 'reverse' | 'none' | { auto: true; merge?: Node[][]; override?: Record<number, BackwardOverride> };

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

export interface InferSpec {
  prompt: number;
  input: Node;
  label: Node;
  logits: Node;
  caches: CacheDecl[];
  draws: ReadonlyMap<number, SampleDraw>;
  sampler?: SamplerSpec;
}

export interface UpdateSpec {
  lr: number;
  weightDecay: number;
  title: string;
  label: string;
}

export interface RowSpec {
  shown: number;
  lit: [number, number];
  labels: number;
  ranges?: ReadonlyMap<Node, [number, number]>;
  reads?: ReadonlyMap<Node, ReadonlySet<number>>;
}

export type RowState = 'hidden' | 'inactive';

export interface StepsSpec {
  forward: ForwardStep[];
  infer?: InferSpec;
  update?: UpdateSpec;
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
  segment: string;
  sub?: string;
  module?: string;
  forwardId?: string;
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
  row?(node: Node, index: readonly number[]): RowState | undefined;
  generation?: GenerationState;
}

export interface GenerationState {
  position: number;
  known: number;
  caches: { decl: CacheDecl; kind: CacheKind; nodes: Node[]; values: number; perToken: number }[];
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

function withMarks(info: StepInfo, step: ForwardStep): StepInfo {
  if (step.marks !== undefined) info.marks = step.marks;
  if (step.brace?.formula !== undefined) info.formula = { title: step.brace.title, tex: step.brace.formula };
  return info;
}

function withNotes(info: StepInfo, notes: StepNote[] | undefined): StepInfo {
  if (notes !== undefined && notes.length > 0) info.notes = notes;
  return info;
}

interface Entry {
  info: StepInfo;
  rows?: RowSpec;
  generation?: GenerationState;
  enter?: () => () => void;
  revealed: Set<Node>;
  active: Set<Node>;
}

type Template = (focus: number) => string;

export function backwardGroups(g: Graph, visible: readonly Node[], spec: Exclude<BackwardSpec, 'auto' | 'reverse' | 'none'>): Node[][] {
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
  if (spec === 'none') return [];
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

type Group = { unit?: Unit; module?: StepModule };

interface Unit {
  index: number;
  step: ForwardStep;
  prod: Node[];
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
  private leave: (() => void) | undefined;
  private positions: Positions | undefined;
  private skipped: (node: Node, offset: number, h: number) => boolean = () => false;

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
        this.units.push({ index, step, prod, active: step.active === undefined ? unique([...prod, ...use]) : toList(step.active) });
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
    entries.push(this.entry(withNotes({ dir: 'forward', index: 0, id: 'f:overview', kind: 'overview', title: overview.title, label: overview.label, text: overview.text, nodes: [...this.visible], segment: overview.label }, overview.notes), all, all));
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

  private forwardGroup(group: Group, index: number): { info: StepInfo; prod: Node[]; active: Node[] } {
    const moduleUse = (m: StepModule) => unique([...m.inputs, ...m.params]);
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
      withNotes(withMarks(info, step), step.notes);
    }
    return { info, prod, active };
  }

  private rebuild(): void {
    const spec = this.spec;
    const g = this.graph;
    const explainer = explain(g);
    const loss = g.findLoss();
    const all = new Set(this.visible);
    const overview = this.overviewSpec();
    const entries: Entry[] = [];
    entries.push(this.entry(withNotes({ dir: 'forward', index: 0, id: 'f:overview', kind: 'overview', title: overview.title, label: overview.label, text: overview.text, nodes: [...this.visible], segment: overview.label }, overview.notes), all, all));
    const revealed = new Set<Node>();
    const inputs = spec.forward.find((step) => step.kind === 'inputs');
    if (inputs !== undefined) {
      const show = toList(inputs.show);
      show.forEach((n) => revealed.add(n));
      const label = inputs.label ?? inputs.title;
      entries.push(this.entry(withNotes({ dir: 'forward', index: 1, id: 'f:inputs', kind: 'inputs', title: inputs.title, label, text: inputs.text ?? '', nodes: show, segment: inputs.segment ?? label }, inputs.notes), new Set(revealed), new Set(show)));
    }
    const groups: Group[] = [];
    for (const unit of this.units) {
      const folded = this.outerFolded(unit.step.module);
      if (folded === undefined) groups.push({ unit });
      else if (groups[groups.length - 1]?.module !== folded) groups.push({ module: folded });
    }
    const moduleUse = (m: StepModule) => unique([...m.inputs, ...m.params]);
    if (spec.infer !== undefined) {
      this.buildInfer(entries, groups, revealed, spec.infer);
      this.setEntries(entries);
      return;
    }
    const forwardIds = new Map<(typeof groups)[number], string>();
    for (const group of groups) {
      const { info, prod, active } = this.forwardGroup(group, entries.length);
      prod.forEach((n) => revealed.add(n));
      forwardIds.set(group, info.id);
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
          forwardId: forwardIds.get(group),
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
      const live = unit.prod.filter((n) => n.grad !== null);
      if (live.length === 0) continue;
      const recv = usesOf(live).filter((n) => !unit.prod.includes(n) && hasGrad(n));
      if (recv.length === 0 && position > 0) continue;
      const subject = recv[0] ?? live[0];
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
    if (spec.update !== undefined && loss !== undefined) entries.push(this.updateEntry(spec.update, loss, all, entries.length));
    this.setEntries(entries);
  }

  private buildInfer(entries: Entry[], groups: Group[], revealed: Set<Node>, infer: InferSpec): void {
    const g = this.graph;
    const P = infer.prompt;
    const T = infer.input.shape[0];
    const scored = new Set<Node>();
    for (const n of g.nodes) if (n.inputs.some((i) => i === infer.label || scored.has(i))) scored.add(n);
    const prodOf = (group: Group) => group.module?.outputs ?? group.unit?.prod ?? [];
    const nodesOf = (group: Group) => group.module?.nodes ?? group.unit?.prod ?? [];
    const kept = groups.filter((group) => !prodOf(group).some((n) => scored.has(n)));
    const scope = new Set(kept.flatMap(nodesOf));
    const cache = analyzeCache(g, T, infer.caches, scope, [infer.logits]);
    this.skipped = (node, offset, h) => h >= P && h < T && cache.reads(h).needed.get(node)?.has(offset) !== true && node.kind === 'op';
    this.positions = cache.positions;
    const { horizons, derived } = cache;
    const cacheSet = cache.cached;
    const members = new Set(infer.caches.flatMap((c) => c.members));
    const inCache = (group: Group) => nodesOf(group).some((n) => members.has(n));
    const computes = (n: Node, t: number) => cache.computes(n, t);
    const before = (n: Node, t: number) => horizons(n).some((h) => h !== undefined && h < t);
    const rowUniform = (n: Node) => {
      const h = horizons(n);
      const per = n.shape.length === 0 ? 1 : n.value.size / n.shape[0];
      return h.every((x, o) => x === h[o - (o % per)]);
    };
    const masks = new Set(g.nodes.filter((n) => n.kind === 'buffer' && horizons(n).some((h) => h !== undefined) && !rowUniform(n) && n.value.data.every((v) => v === 0 || v === -Infinity)));
    const positional = g.nodes.filter((n) => n.kind === 'buffer' && horizons(n).some((h) => h !== undefined) && rowUniform(n));
    const ancestors = (node: Node) => {
      const seen = new Set<Node>();
      const walk = (n: Node) => n.inputs.forEach((i) => !seen.has(i) && (seen.add(i), walk(i)));
      walk(node);
      return seen;
    };
    const head = new Set<Node>();
    const back = (n: Node) => {
      if (head.has(n) || n.kind !== 'op') return;
      head.add(n);
      if (n.op !== 'matmul') n.inputs.forEach(back);
    };
    back(infer.logits);
    const scores = new Set<Node>();
    for (const n of g.nodes) {
      if (!members.has(n) || n.kind !== 'op' || derived.has(n)) continue;
      const fromCache = n.inputs.some((i) => derived.has(i)) && n.inputs.some((i) => !derived.has(i) && i.kind === 'op') && !n.inputs.some((i) => scores.has(i)) && (n.op === 'matmul' || n.op === 'select_matmul');
      const along = n.op !== 'matmul' && n.op !== 'select_matmul' && n.inputs.some((i) => scores.has(i) && sameShape(i, n));
      if (fromCache || along) scores.add(n);
    }
    const name = infer.input.name;
    const label = infer.label.name;
    const tex = (nodes: Node[]) => unique(nodes).map((n) => `$${n.tex}$`).join('、');
    const perRow = (n: Node) => (n.shape.length === 0 ? 1 : n.value.size / n.shape[0]);
    const firstIndex = (n: Node, offset: number) => {
      const index: number[] = [];
      let rest = offset;
      for (let k = n.shape.length - 1; k >= 0; k--) {
        index.unshift(rest % n.shape[k]);
        rest = Math.floor(rest / n.shape[k]);
      }
      return index;
    };
    const at = (n: Node, h: number) => {
      const offset = horizons(n).indexOf(h);
      return offset < 0 ? undefined : firstIndex(n, offset);
    };
    const vector = (n: Node, row: number, width: number, digits = ':3') => `\\begin{bmatrix}${Array.from({ length: width }, (_, s) => `{{${n.name}[${row},${s}]${digits}}}`).join('&')}\\end{bmatrix}`;
    const rowValues = (n: Node, row: number) => {
      if (n.shape.length !== 2 || n.shape[0] !== T) return '';
      const width = scores.has(n) ? row + 1 : n.shape[1];
      return width > 8 ? '' : `$${n.tex}$ 的第 ${row + 1} 行：$$${vector(n, row, width)}$$`;
    };
    const subsOf = (parts: { info: StepInfo }[]) => [...new Set(parts.map((part) => part.info.sub ?? part.info.segment).filter((t): t is string => t !== undefined))].join('、');
    const shownBy = (shown: number) => new Set([...revealed].filter((n) => (cache.first(n) ?? 0) < shown));
    const push = (info: Omit<StepInfo, 'index' | 'dir' | 'kind'>, active: Iterable<Node>, rows: RowSpec, notes?: StepNote[]) => {
      const full: StepInfo = { ...info, dir: 'forward', kind: 'op', index: entries.length };
      withNotes(full, notes);
      const entry = this.entry(full, shownBy(rows.shown), new Set(active));
      entry.rows = rows;
      entry.generation = generationAt(current, Math.max(current + 1, rows.labels + 1));
      entries.push(entry);
    };
    let current = P - 1;
    const generationAt = (position: number, known: number): GenerationState => ({
      position,
      known,
      caches: infer.caches.map((decl) => {
        const kind = cache.kind(decl);
        const values = cache.stored(decl, position + 1);
        const nodes = kind === 'state' ? decl.nodes.filter((n) => cache.horizons(n)[0] === position) : decl.nodes;
        return { decl, kind, nodes, values, perToken: kind === 'rows' ? values - cache.stored(decl, position) : 0 };
      }),
    });
    const decodeRows = (t: number, extra: Partial<RowSpec> = {}): RowSpec => ({ shown: t + 1, lit: [t, t + 1], labels: t, ...extra });
    const inputsEntry = entries.findIndex((e) => e.info.kind === 'inputs');
    if (inputsEntry >= 0) {
      entries[inputsEntry].rows = { shown: P, lit: [0, P], labels: 0 };
      entries[inputsEntry].generation = generationAt(P - 1, P);
      entries[inputsEntry].revealed = new Set([...entries[inputsEntry].revealed].filter((n) => (cache.first(n) ?? 0) < P));
    }
    const kinds = new Map(infer.caches.map((c) => [c, cache.kind(c)]));
    const sizeOf = (c: CacheDecl) => (kinds.get(c) === 'state' ? c.nodes[c.nodes.length - 1].value.size : c.nodes.reduce((sum, n) => sum + perRow(n), 0));
    const describe = (c: CacheDecl, t: number, phase: 'prefill' | 'decode'): string => {
      const kind = kinds.get(c);
      const latest = c.nodes.filter((n) => computes(n, t));
      if (kind === 'state') {
        const previous = c.nodes.filter((n) => computes(n, t - 1));
        return phase === 'prefill'
          ? `${tex(latest)} 是处理完 prompt 后的状态，有 ${sizeOf(c)} 个数。之后的位置只需要它，更早的状态都可以丢掉，所以缓存的大小固定，不随序列变长。`
          : `${tex(previous)} 是上一个位置留下的状态，本步只读它一次，算出新的状态 ${tex(latest)} 替换它。状态有 ${sizeOf(c)} 个数，不随序列变长，所以这部分每一步的计算量相同。`;
      }
      const nodes = tex(c.nodes);
      if (kind === 'window') {
        const width = cache.window(c, T - 1).length;
        return phase === 'prefill'
          ? `${nodes} 写入缓存。后面的位置只会读最近 ${width} 行，所以缓存只保留最后 ${width} 行，每行 ${sizeOf(c)} 个数，大小固定。`
          : `本步读缓存中最近的 ${cache.window(c, t).length} 行 ${nodes}，再把第 ${t + 1} 行追加进去，最早的一行不会再被读，可以丢掉。缓存始终最多 ${width} 行，每行 ${sizeOf(c)} 个数。`;
      }
      return phase === 'prefill'
        ? `${nodes} 写入缓存，prompt 的 ${P} 个位置全部写入。之后每个位置都要读前面所有位置的这些值，有了缓存就不必再算。每个 token 在这一层缓存 ${sizeOf(c)} 个数，缓存的大小随序列长度线性增长。`
        : `前 ${t} 行 ${nodes} 已经在缓存里（暗色），本步只把刚算出的第 ${t + 1} 行（亮色）追加进去，缓存变成 ${t + 1} 行，每行 ${sizeOf(c)} 个数。`;
    };
    const roped = (c: CacheDecl) => positional.length > 0 && c.nodes.some((n) => positional.some((b) => ancestors(n).has(b)));
    const why = `能直接复用，是因为第 $s$ 个位置的这些值只取决于前 $s$ 个 token，后面生成的 token 不会改变它们。缓存里的数和把整段序列重新算一遍得到的完全相同，切到训练模式可以对照。缓存省掉的是重复计算，不是近似。`;
    const cacheNotes = (decls: CacheDecl[], t: number, decode: boolean): StepNote[] => {
      const notes: StepNote[] = [];
      for (const c of decls) {
        const kind = kinds.get(c);
        if (kind === 'state') {
          const latest = c.nodes.find((n) => computes(n, t));
          const previous = c.nodes.find((n) => computes(n, t - 1));
          if (decode && previous !== undefined) notes.push({ nodes: [previous], text: '读上一个状态', side: 'left' });
          if (latest !== undefined) notes.push({ nodes: [latest], text: decode ? '新状态替换旧状态' : '保留这个状态', side: 'right' });
          continue;
        }
        const node = c.nodes[0];
        const newest = at(node, t);
        const oldest = at(node, decode ? (cache.window(c, t)[0] ?? 0) : 0);
        if (decode && oldest !== undefined) notes.push({ nodes: [node], index: oldest, text: '从缓存读出', side: 'right' });
        if (newest !== undefined) notes.push({ nodes: [node], index: newest, text: decode ? '本步新算，追加' : '写入缓存', side: 'left' });
      }
      return notes.slice(0, 4);
    };
    const totalWith = T;
    let totalWithout = P;
    for (let r = P; r < T; r++) totalWithout += r + 1;

    const writesAt = (t: number, prefill: boolean) => {
      const byGroup = new Map<number, CacheDecl[]>();
      for (const c of infer.caches) {
        const k = kept.reduce((last, group, j) => (nodesOf(group).some((n) => c.nodes.includes(n) && (prefill ? before(n, P) : computes(n, t))) ? j : last), -1);
        if (k >= 0) byGroup.set(k, [...(byGroup.get(k) ?? []), c]);
      }
      return byGroup;
    };
    const prefillWrites = writesAt(P - 1, true);
    let firstHead = true;
    kept.forEach((group, k) => {
      if (!prodOf(group).some((n) => before(n, P) || cache.first(n) === undefined)) return;
      const { info, prod, active } = this.forwardGroup(group, entries.length);
      prod.forEach((n) => revealed.add(n));
      info.id = `p:${info.id}`;
      if (info.segment !== undefined) info.sub = info.segment;
      info.segment = 'prefill';
      const onlyLast = prod.length > 0 && prod.every((n) => head.has(n));
      if (onlyLast) {
        info.text += firstHead
          ? `\n\n推理时只用最后一行的 $${infer.logits.tex}$ 采样，前面 ${P - 1} 行只在训练算 loss 时需要。HF 的 <code>generate</code> 传入 <code>logits_to_keep=1</code>，prefill 只对最后一个位置做这一步，图里也只亮这一行。`
          : `\n\n推理时这一步同样只算最后一行。`;
        firstHead = false;
      }
      const ranges = new Map<Node, [number, number]>(onlyLast ? prod.map((n) => [n, [P - 1, P]]) : []);
      push(info, active, { shown: P, lit: [0, P], labels: 0, ranges });
      const here = prefillWrites.get(k);
      if (here !== undefined) {
        const nodes = here.flatMap((c) => c.nodes);
        const ranges2 = new Map<Node, [number, number]>();
        for (const c of here) {
          const kind = kinds.get(c);
          const from = kind === 'state' ? P - 1 : kind === 'window' ? P - cache.window(c, T - 1).length : 0;
          c.nodes.forEach((n) => ranges2.set(n, [Math.max(0, from), P]));
        }
        push(
          {
            id: `p:cache:${k}`,
            title: `写入缓存：prompt 的 ${P} 个位置`,
            label: '写入缓存',
            text: `${here.map((c) => describe(c, P - 1, 'prefill') + (c.note ?? '')).join('')}${here.some(roped) ? '缓存存的是加过位置编码之后的值，读出来可以直接用，不需要再按位置旋转。' : ''}`,
            nodes,
            segment: 'prefill',
            sub: '缓存',
          },
          nodes,
          { shown: P, lit: [0, P], labels: 0, ranges: ranges2 },
          cacheNotes(here, P - 1, false),
        );
      }
    });

    const runs: { cache: boolean; groups: Group[] }[] = [];
    for (const group of kept) {
      const isCache = inCache(group);
      const last = runs[runs.length - 1];
      if (last !== undefined && last.cache === isCache) last.groups.push(group);
      else runs.push({ cache: isCache, groups: [group] });
    }

    const sample = (row: number, phase: string) => {
      const V = infer.logits.shape[1];
      const tokens = infer.label.tokens ?? infer.input.tokens;
      const next = row + 1 < T ? `它被写进输入的第 ${row + 2} 个位置，成为下一个 decode 步唯一的新输入。` : `页面只画 ${T} 个位置，生成到这里停止。`;
      const draw = infer.draws.get(row);
      const picked = `{{${label}[${row}]:token}}`;
      let body: string;
      if (infer.sampler === undefined || draw === undefined) {
        const listed = tokens !== undefined && V <= 8 ? `$$${Array.from({ length: V }, (_, v) => `\\log p(\\text{${tokens[v]}})={{${infer.logits.name}[${row},${v}]:3}}`).join(',\\quad ')}$$` : '';
        body = `第 ${row + 1} 行的 $${infer.logits.tex}$ 是词表里每个 token 的对数概率：${listed}greedy 解码取最大的一项，得到 ${picked}。${next}推理没有标签，图里 $${infer.label.tex}$ 这一格显示的就是采样结果。按温度 $\\tau$ 随机采样时，改为按 $\\operatorname{softmax}(\\mathbf{z}/\\tau)$ 的概率抽取，前面的计算不变。`;
      } else {
        const first = row === P - 1;
        body = first
          ? `${samplingSteps(infer.sampler, infer.logits.tex, row, V)}\n\n${samplingTable(draw, tokens, infer.sampler)}这一步抽到 ${picked}。${pickNote(draw, tokens)}${next}推理没有标签，图里 $${infer.label.tex}$ 这一格显示的就是采样结果。随机数来自页面设置的种子 ${infer.sampler.seed}，每次打开页面结果相同；PyTorch 和 vLLM 用各自的随机数生成器，同样的种子抽到的 token 一般不同，但抽取所依据的分布 $q$ 相同。`
          : `第 ${row + 1} 行按同样的采样设置处理：${samplingTable(draw, tokens, infer.sampler)}抽到 ${picked}。${pickNote(draw, tokens)}${next}`;
      }
      push(
        {
          id: `s:${row}`,
          title: `采样第 ${row + 2} 个 token`,
          label: '采样',
          text: body,
          nodes: [infer.logits],
          segment: phase,
          sub: '采样',
        },
        [infer.logits, infer.label],
        { shown: row + 1, lit: [row, row + 1], labels: row + 1 },
      );
    };
    sample(P - 1, 'prefill');

    for (let row = P; row < T; row++) {
      current = row;
      const phase = `decode ${row - P + 1}`;
      const detailed = row === P;
      const pos = row + 1;
      const { reads } = cache.reads(row);
      const writes = writesAt(row, false);
      const groupIndex = new Map(kept.map((group, k) => [group, k]));
      const growing = infer.caches.filter((c) => kinds.get(c) === 'rows');
      const recompute: Node[] = [];
      let seenCacheRun = false;
      let contrasted = false;
      const afterWrite = (...covered: Group[]) => {
        const keys = covered.map((group) => groupIndex.get(group) ?? -1);
        const here = detailed ? keys.flatMap((key) => writes.get(key) ?? []) : [];
        if (here.length === 0) return;
        const nodes = here.flatMap((c) => c.nodes);
        push(
          {
            id: `d:${row}:cache:${keys.join('-')}`,
            title: `${phase}：更新缓存`,
            label: '更新缓存',
            text: `${here.map((c) => describe(c, row, 'decode')).join('')}${contrasted ? '' : why}`,
            nodes: nodes.filter((n) => computes(n, row)),
            segment: phase,
            sub: '缓存',
          },
          nodes,
          decodeRows(row),
          cacheNotes(here, row, true),
        );
        if (contrasted) return;
        contrasted = true;
        const redo = unique([...recompute, ...nodes]);
        push(
          {
            id: `d:${row}:nocache`,
            title: `${phase}：如果不用缓存`,
            label: '不用缓存',
            text: `如果不存 ${tex(nodes)}，第 ${pos} 个位置需要的这些值取决于前面每个位置、这一层和下面每一层的全部运算，只能把 ${pos} 个位置从头重算一遍。图里亮起的就是到这里为止要重算的部分。按这一页的长度，从 ${P} 个 token 的 prompt 生成到 ${T} 个位置，用缓存一共算 ${totalWith} 个位置，不用缓存要算 ${totalWithout} 个；序列长度为 $n$ 时分别约为 $n$ 和 $n^2/2$。两者的结果完全相同，缓存的代价是显存。`,
            nodes: redo,
            segment: phase,
            sub: '对比',
          },
          redo,
          { shown: pos, lit: [0, pos], labels: row },
        );
      };
      runs.forEach((run, k) => {
        const live = run.groups.filter((group) => nodesOf(group).some((n) => computes(n, row) || (reads.has(n) && derived.has(n))));
        if (live.length === 0) return;
        const parts = live.map((group) => ({ group, ...this.forwardGroup(group, entries.length) }));
        const prod = unique(parts.flatMap((part) => part.prod));
        const active = unique(parts.flatMap((part) => part.active));
        const subs = subsOf(parts);
        if (detailed) recompute.push(...prod);
        if (!run.cache) {
          const first = !seenCacheRun && k === 0;
          const usesPosition = positional.filter((b) => active.includes(b));
          const posText = usesPosition.length > 0 ? `位置相关的常量 ${tex(usesPosition)} 只取第 ${pos} 行，也就是位置 ${row}（从 0 数），它等于缓存里已有的行数。` : '';
          const hasLogits = prod.includes(infer.logits);
          const text = first
            ? detailed
              ? `输入只有一个 token：上一步采样得到的 {{${name}[${row}]:token}}。在第一个带缓存的模块之前，这些运算（${subs}）都是逐位置的：第 ${pos} 个位置的输出只取决于这个位置的输入，所以前 ${row} 个位置在 prefill 里算出的结果不会变，这一步不再重算。矩阵乘变成一行向量乘整个权重矩阵，每个权重读进来只做一次乘加，计算量很小，耗时主要花在读权重上。实际的推理服务因此把许多请求各自的这一行拼成一批，一起乘同一份权重。${posText}`
              : `输入是 {{${name}[${row}]:token}}。和 decode 1 一样只算这个位置。${posText}`
            : detailed
              ? `带缓存的模块只输出这一个位置，之后的运算（${subs}）又是逐位置的，同样只算这一行。${hasLogits ? `最后的 $${infer.logits.tex}$ 也只有这一行，正好是采样需要的。` : ''}`
              : `${subs} 只算第 ${pos} 行。`;
          const notes: StepNote[] = [];
          if (detailed && first) {
            notes.push({ nodes: [infer.input], index: [row], text: '上一步采样的 token', side: 'left' });
            const cos = usesPosition[0];
            const index = cos === undefined ? undefined : at(cos, row);
            if (cos !== undefined && index !== undefined) notes.push({ nodes: [cos], index, text: `位置 ${row}`, side: 'left' });
          }
          push(
            {
              id: `d:${row}:${k}`,
              title: first ? `${phase}：第 ${pos} 个 token 进入模型` : `${phase}：逐位置运算`,
              label: first ? '新 token' : '逐位置',
              text,
              nodes: prod,
              segment: phase,
              sub: subs,
            },
            active,
            decodeRows(row),
            notes,
          );
          afterWrite(...live);
          return;
        }
        seenCacheRun = true;
        const runCaches = infer.caches.filter((c) => c.members.some((n) => nodesOf(run.groups[0]).includes(n) || run.groups.some((group) => nodesOf(group).includes(n))));
        const runGrowing = runCaches.some((c) => growing.includes(c));
        const readNodes = [...reads.keys()];
        const show = { reads };
        if (!detailed) {
          const score = [...scores].find((n) => prod.includes(n) && n.shape.length === 2);
          const writes = runCaches.flatMap((c) => c.nodes).filter((n) => computes(n, row));
          const text = runGrowing
            ? `query 仍然只有第 ${pos} 个位置，但缓存现在有 ${pos} 行：分数行的长度变成 ${pos}，要从缓存读的也多了一行。逐位置的部分每一步计算量不变，这部分随位置线性增长，这就是上下文越长、decode 越慢的原因。${score === undefined ? '' : rowValues(score, row)}`
            : `读上一个位置留下的缓存，算出这个位置的输出${writes.length > 0 ? `，并把 ${tex(writes)} 写回缓存` : ''}。这部分缓存的大小固定，每一步的计算量和位置无关。`;
          push(
            { id: `d:${row}:${k}`, title: runGrowing ? `${phase}：读取 ${pos} 行缓存` : `${phase}：读取并更新缓存`, label: '读缓存', text, nodes: prod, segment: phase, sub: subs },
            unique([...active, ...readNodes.filter((n) => members.has(n) || cacheSet.has(n))]),
            decodeRows(row, show),
          );
          afterWrite(...live);
          return;
        }
        const readParts = parts.filter((part) => part.prod.every((n) => derived.has(n)) && part.prod.some((n) => reads.has(n)));
        const pushReads = () => {
          const weights = readParts.some((part) => part.prod.some((n) => n.op === 'matmul' && n.inputs.some((i) => horizons(i).every((h) => h === undefined))));
          const sources = readParts.flatMap((part) => part.active).filter((n) => cacheSet.has(n));
          push(
            {
              id: `d:${row}:${k}:read`,
              title: `${phase}：从缓存恢复`,
              label: '读缓存',
              text: `这几步（${readParts.map((part) => part.info.title).join('、')}）只用到缓存中的 ${tex(sources)}，和本步的 query 无关，所以缓存的 ${pos} 行全部参与，每个 decode 步都要把整个缓存读一遍。${weights ? `其中要乘权重矩阵，相当于每一步都对缓存的全部 ${pos} 行重做一次投影。注意力对 key、value 是线性的，推理框架通常把这个权重合并进 query 和输出投影（权重吸收），直接在缓存上计算，省掉这次重算。` : ''}`,
              nodes: unique(readParts.flatMap((part) => part.prod)),
              segment: phase,
              sub: subs,
            },
            unique([...readParts.flatMap((part) => part.active), ...sources]),
            decodeRows(row, show),
          );
        };
        let pending: typeof parts = [];
        const flush = () => {
          if (pending.length === 0) return;
          const group = pending;
          pending = [];
          push(
            {
              id: `d:${row}:${k}:${parts.indexOf(group[0])}`,
              title: group.length === 1 ? `${phase}：${group[0].info.title}` : `${phase}：${group[0].info.title} 等 ${group.length} 步`,
              label: group[0].info.label,
              text: `${group.length === 1 ? '这一步' : `这 ${group.length} 步（${group.map((part) => part.info.title).join('、')}）`}和 prefill 时的算法相同，都只在本位置内部计算，只算第 ${pos} 个位置。`,
              nodes: unique(group.flatMap((part) => part.prod)),
              segment: phase,
              sub: subsOf(group),
            },
            unique(group.flatMap((part) => part.active)),
            decodeRows(row, show),
          );
          afterWrite(...group.map((part) => part.group));
        };
        parts.forEach((part, j) => {
          if (readParts.includes(part)) {
            flush();
            if (part === readParts[0]) pushReads();
            afterWrite(part.group);
            return;
          }
          if (!part.prod.some((n) => computes(n, row))) return;
          const first = part.prod[0];
          const mask = part.active.find((n) => masks.has(n));
          const direct = unique(part.prod.flatMap((n) => n.inputs.filter((i) => cacheSet.has(i) && reads.has(i))));
          const scoring = part.prod.some((n) => scores.has(n));
          if (part.info.kind !== 'module' && !scoring && mask === undefined && direct.length === 0 && !part.active.some((n) => derived.has(n))) {
            pending.push(part);
            if (writes.has(groupIndex.get(part.group) ?? -1)) flush();
            return;
          }
          flush();
          let text: string;
          if (part.info.kind === 'module') {
            const inner = infer.caches.filter((c) => c.members.some((n) => nodesOf(part.group).includes(n)));
            text = `这个模块折叠着，内部和展开的同类模块一样：只算第 ${pos} 个位置，从缓存读前 ${row} 个位置的 ${tex(inner.flatMap((c) => c.nodes))}，再把本位置的值写回缓存。${rowValues(first, row)}`;
          } else if (scoring && part.active.some((n) => derived.has(n))) {
            text = `query 只有第 ${pos} 个位置。它和缓存里每一行的 key 做点积，分数只有一行，长度是 ${pos}，不再是 prefill 时 $${P}\\times ${P}$ 的方阵。${rowValues(first, row)}`;
          } else if (mask !== undefined) {
            text = `第 ${pos} 行的 mask 是 $${vector(mask, row, pos, ':0')}$，对已有的 ${pos} 个位置全是 0。缓存里只有不晚于当前位置的 key，没有需要挡住的未来位置，所以 decode 实际上不需要 mask，加上的只是 0。${rowValues(first, row)}`;
          } else if (direct.length > 0) {
            text = `这一步直接读缓存中的 ${tex(direct)}，算法和 prefill 的同一步相同，只算第 ${pos} 个位置。${rowValues(first, row)}`;
          } else if (part.active.some((n) => derived.has(n))) {
            text = `第 ${pos} 个位置按权重对缓存里全部 ${pos} 行的 value 加权求和，得到这个位置唯一的一行输出。${rowValues(first, row)}`;
          } else {
            text = `和 prefill 的同一步相同，只算第 ${pos} 个位置，这一行的长度是 ${pos}。${rowValues(first, row)}`;
          }
          push(
            { id: `d:${row}:${k}:${j}`, title: `${phase}：${part.info.title}`, label: part.info.label, text, nodes: part.prod, segment: phase, sub: part.info.sub ?? part.info.segment },
            unique([...part.active, ...direct]),
            decodeRows(row, show),
          );
          afterWrite(part.group);
        });
        flush();
      });
      sample(row, phase);
    }
  }

  private updateEntry(update: UpdateSpec, loss: Node, all: Set<Node>, index: number): Entry {
    const g = this.graph;
    const params = this.visible.filter((n) => n.kind === 'param' && n.grad !== null);
    const before = loss.value.data[0];
    const info: StepInfo = {
      dir: 'forward',
      index,
      id: 'u:adamw',
      kind: 'op',
      title: update.title,
      label: update.label,
      text: `AdamW 第一步：$m_1$、$v_1$ 从 0 开始，偏差修正后 $\\hat m_1=g$、$\\hat v_1=g^2$，每个参数按 $W\\leftarrow W-\\eta\\big(\\tfrac{g}{|g|+\\epsilon}+\\lambda W\\big)$ 更新，$\\eta=${update.lr}$、$\\lambda=${update.weightDecay}$。每个参数移动的幅度都约等于 $\\eta$，方向与梯度相反。更新前 $\\ell=${before.toFixed(4)}$，用新参数再做一次前向，$\\ell={{${loss.name}:4}}$。`,
      nodes: [...params, loss],
      segment: update.label,
    };
    const entry = this.entry(info, all, new Set([...params, loss]));
    entry.enter = () => {
      const saved = params.map((p) => Array.from(p.value.data));
      const steps = params.map((p) => Array.from(p.value.data, (w, i) => {
        const grad = p.grad!.data[i];
        return w - update.lr * (grad / (Math.abs(grad) + 1e-8) + update.weightDecay * w);
      }));
      g.batch(() => params.forEach((p, k) => p.set(reshape(steps[k], p.shape))));
      return () => g.batch(() => params.forEach((p, k) => p.set(reshape(saved[k], p.shape))));
    };
    return entry;
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
    this.leave?.();
    this.leave = undefined;
    this.current = next;
    this.leave = this.entries[next].enter?.();
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
      row: (node, index) => rowState(entry, this.spec.infer, this.positions, this.skipped, node, index),
      ...(entry.generation === undefined ? {} : { generation: entry.generation }),
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
    this.leave?.();
    this.leave = undefined;
    this.subscribers.length = 0;
  }

  private notify(): void {
    [...this.subscribers].forEach((fn) => fn());
  }
}

const PROB_DIGITS = 3;

function samplingSteps(sampler: SamplerSpec, logpTex: string, row: number, vocab: number): string {
  const parts = [`第 ${row + 1} 行的 $${logpTex}$ 是词表里每个 token 的对数概率。页面设置温度 $\\tau=${sampler.temperature}$${sampler.topK === undefined ? '' : `、top-k $k=${sampler.topK}$`}${sampler.topP === undefined ? '' : `、top-p $p=${sampler.topP}$`}。${sampler.note ?? ''}这几步按 Hugging Face 的顺序依次处理，vLLM 与 SGLang 的顺序相同。`];
  parts.push(`温度：$p_\\tau=\\operatorname{softmax}(\\log p/\\tau)$。$\\log p$ 与 logits $\\mathbf{z}$ 的每一项只差同一个常数，除以 $\\tau$ 以后仍然只差常数，所以结果等于 $\\operatorname{softmax}(\\mathbf{z}/\\tau)$。$\\tau<1$ 让分布更集中，$\\tau>1$ 让分布更平。`);
  if (sampler.topK !== undefined) parts.push(`top-k：只保留 $p_\\tau$ 最大的 ${sampler.topK} 个 token，其余的概率置为 0。${sampler.topK >= vocab ? `这里的词表只有 ${vocab} 个 token，所以 top-k 不去掉任何 token；真实词表有十几万个 token，它才起作用。` : ''}`);
  if (sampler.topP !== undefined) parts.push(`top-p：把${sampler.topK === undefined ? '' : '剩下 token 的'}概率重新归一化，从大到小累加；一个 token 之前的累加和已经达到 $p$ 时，它和比它小的 token 都去掉。保留下来的是总概率达到 $p$ 所需的最少几个 token。`);
  parts.push('把保留下来的概率重新归一化得到 $q$，取一个 $[0,1)$ 里的随机数 $u$，从概率最大的 token 开始累加 $q$，累加和第一次超过 $u$ 的 token 就是采样结果。');
  return parts.join('\n\n');
}

function pickNote(draw: SampleDraw, tokens: readonly string[] | undefined): string {
  const top = draw.order[0];
  if (draw.token === top) return '';
  const name = (v: number) => tokens?.[v] ?? String(v);
  const before = draw.order.slice(0, draw.order.indexOf(draw.token)).reduce((sum, v) => sum + draw.final[v], 0);
  return `$u$ 超过了排在前面的 token 的累加和 ${before.toFixed(PROB_DIGITS)}，所以这次抽到的不是概率最大的 ${name(top)}，greedy 解码会选 ${name(top)}。`;
}

function samplingTable(draw: SampleDraw, tokens: readonly string[] | undefined, sampler: SamplerSpec): string {
  const fmt = (x: number) => x.toFixed(PROB_DIGITS);
  let cumulative = 0;
  const rows = draw.order.map((v) => {
    const name = tokens?.[v] ?? String(v);
    const keep = draw.kept[v];
    if (keep) cumulative += draw.final[v];
    const cells = [`\\text{${name}}`, fmt(draw.logp[v]), fmt(draw.scaled[v]), keep ? fmt(draw.final[v]) : '0', keep ? fmt(cumulative) : '\\cdot'];
    return v === draw.token ? cells.map((cell) => `\\mathbf{${cell}}`).join(' & ') : cells.join(' & ');
  });
  const tau = `p_{\\tau=${sampler.temperature}}`;
  return `$$\\begin{array}{lrrrr}\\text{token} & \\log p & ${tau} & q & \\textstyle\\sum q \\\\ \\hline ${rows.join(' \\\\ ')}\\end{array}\\qquad u=${fmt(draw.u ?? 0)}$$`;
}

function rowState(entry: Entry, infer: InferSpec | undefined, positions: Positions | undefined, skipped: (node: Node, offset: number, h: number) => boolean, node: Node, index: readonly number[]): RowState | undefined {
  const rows = entry.rows;
  if (rows === undefined || infer === undefined || positions === undefined) return undefined;
  if (node === infer.label) {
    const position = index[0];
    if (position >= rows.labels || position < infer.prompt - 1) return 'hidden';
    return entry.active.has(node) && position !== rows.labels - 1 ? 'inactive' : undefined;
  }
  const offset = offsetOf(collapse(node.shape, index.length), index);
  const h = positions.horizon(node, offset);
  if (h === undefined) return undefined;
  if (h >= rows.shown || skipped(node, offset, h)) return 'hidden';
  if (!entry.active.has(node)) return undefined;
  if (rows.reads?.get(node)?.has(offset) === true) return undefined;
  const [from, to] = rows.ranges?.get(node) ?? rows.lit;
  return h >= from && h < to ? undefined : 'inactive';
}

function collapse(shape: readonly number[], rank: number): number[] {
  if (rank >= shape.length || rank === 0) return [...shape];
  return [...shape.slice(0, rank - 1), shape.slice(rank - 1).reduce((a, b) => a * b, 1)];
}

function sameShape(a: Node, b: Node): boolean {
  return a.shape.length === b.shape.length && a.shape.every((n, i) => n === b.shape[i]);
}

function reshape(flat: number[], shape: readonly number[]): Nested {
  if (shape.length === 0) return flat[0];
  const stride = shape.slice(1).reduce((a, b) => a * b, 1);
  return Array.from({ length: shape[0] }, (_, i) => reshape(flat.slice(i * stride, (i + 1) * stride), shape.slice(1)));
}

export function createSteps(g: Graph, spec: StepsSpec): Steps {
  return new StepsImpl(g, spec);
}

export function allVisible(_g: Graph): StepState {
  return { dir: 'forward', index: 0, kind: 'overview', focus: 0, revealed: () => true, active: () => true };
}
