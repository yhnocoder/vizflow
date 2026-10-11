import type { Graph, Node } from '../autograd';
import type { Nested } from '../tensor';
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

export interface InferSpec {
  prompt: number;
  input: Node;
  label: Node;
  logits: Node;
  caches: { nodes: Node[]; members: Node[] }[];
}

export interface UpdateSpec {
  lr: number;
  weightDecay: number;
  title: string;
  label: string;
}

export interface RowSpec {
  shown: number;
  active: [number, number];
  labels: number;
  ranges?: ReadonlyMap<Node, [number, number]>;
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
  row?(node: Node, position: number, key?: number): RowState | undefined;
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
  enter?: () => () => void;
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

type Group = { unit?: Unit; module?: StepModule };

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
  private leave: (() => void) | undefined;

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
      const recv = unit.use.filter(hasGrad);
      if (recv.length === 0 && position > 0) continue;
      const subject = recv[0] ?? unit.prod[0];
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
    const kept = groups.filter((group) => !(group.unit?.prod.some((n) => scored.has(n)) ?? false));
    const prodOf = (group: Group) => group.module?.outputs ?? group.unit?.prod ?? [];
    const nodesOf = (group: Group) => group.module?.nodes ?? group.unit?.prod ?? [];
    const members = new Set(infer.caches.flatMap((c) => c.members));
    const cacheNodes = unique(infer.caches.flatMap((c) => c.nodes));
    const cacheSet = new Set(cacheNodes);
    const inCache = (group: Group) => nodesOf(group).some((n) => members.has(n));
    const derived = new Set<Node>();
    for (const n of g.nodes) if (members.has(n) && n.kind === 'op' && n.inputs.every((i) => cacheSet.has(i) || derived.has(i))) derived.add(n);
    const masks = new Set<Node>();
    const isMask = (n: Node) => n.kind === 'buffer' && n.shape.length === 2 && n.shape[0] === T && n.shape[1] === T && n.value.data.every((v) => v === 0 || v === -Infinity);
    for (const n of members) for (const i of n.inputs) if (isMask(i)) masks.add(i);
    const positional = g.nodes.filter((n) => n.kind === 'buffer' && n.shape[0] === T && !masks.has(n));
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
      const fromCache = n.inputs.some((i) => derived.has(i)) && n.inputs.some((i) => !derived.has(i) && i.kind === 'op');
      const along = n.op !== 'matmul' && n.op !== 'select_matmul' && n.inputs.some((i) => scores.has(i) && sameShape(i, n));
      if ((fromCache && n.inputs.every((i) => !scores.has(i))) || along) scores.add(n);
    }
    const name = infer.input.name;
    const label = infer.label.name;
    const tex = (nodes: Node[]) => nodes.map((n) => `$${n.tex}$`).join('、');
    const perToken = cacheNodes.reduce((sum, n) => sum + n.shape.slice(1).reduce((a, b) => a * b, 1), 0);
    const zeros = (n: Node, row: number) => n.shape.map((_, k) => (k === 0 ? row : 0));
    const vector = (n: Node, row: number, width: number, digits = ':3') => `\\begin{bmatrix}${Array.from({ length: width }, (_, s) => `{{${n.name}[${row},${s}]${digits}}}`).join('&')}\\end{bmatrix}`;
    const rowValues = (n: Node, row: number) => {
      if (n.shape.length !== 2) return '';
      const width = scores.has(n) ? row + 1 : n.shape[1];
      return width > 8 ? '' : `$$${n.tex}_{${row + 1}}=${vector(n, row, width)}$$`;
    };
    const subsOf = (parts: { info: StepInfo }[]) => [...new Set(parts.map((part) => part.info.sub ?? part.info.segment).filter((t): t is string => t !== undefined))].join('、');
    const push = (info: Omit<StepInfo, 'index' | 'dir' | 'kind'>, active: Iterable<Node>, rows: RowSpec, notes?: StepNote[]) => {
      const full: StepInfo = { ...info, dir: 'forward', kind: 'op', index: entries.length };
      withNotes(full, notes);
      const entry = this.entry(full, new Set(revealed), new Set(active));
      entry.rows = rows;
      entries.push(entry);
    };
    const rowsAt = (row: number, ranges?: [Iterable<Node>, [number, number]][]): RowSpec => {
      const map = new Map<Node, [number, number]>();
      for (const [nodes, range] of ranges ?? []) for (const n of nodes) map.set(n, range);
      return { shown: row + 1, active: [row, row + 1], labels: row, ranges: map };
    };
    const inputsEntry = entries.findIndex((e) => e.info.kind === 'inputs');
    if (inputsEntry >= 0) entries[inputsEntry].rows = { shown: P, active: [0, P], labels: 0 };

    const lastCacheWrite = kept.reduce((last, group, k) => (prodOf(group).some((n) => cacheSet.has(n)) ? k : last), -1);
    const roped = positional.length > 0 && cacheNodes.some((n) => positional.some((b) => ancestors(n).has(b)));
    const cacheWhy = `能直接复用，是因为因果 mask 让第 $s$ 行的 ${tex(cacheNodes)} 只取决于前 $s$ 个 token，后面生成的 token 不会改变它们。缓存里的数和把整段序列重新算一遍得到的完全相同，切到训练模式可以对照同一行。缓存省掉的是重复计算，不是近似。`;
    kept.forEach((group, k) => {
      const { info, prod, active } = this.forwardGroup(group, entries.length);
      prod.forEach((n) => revealed.add(n));
      info.id = `p:${info.id}`;
      if (info.segment !== undefined) info.sub = info.segment;
      info.segment = 'prefill';
      const onlyLast = prod.length > 0 && prod.every((n) => head.has(n));
      const firstHead = onlyLast && !entries.some((e) => e.info.id.startsWith('p:') && e.rows?.ranges?.size !== undefined && e.rows.ranges.size > 0);
      if (onlyLast && !firstHead) info.text += `\n\n推理时这一步同样只算最后一行。`;
      if (firstHead) info.text += `\n\n推理时只用最后一行的 $${infer.logits.tex}$ 采样，前面 ${P - 1} 行只在训练算 loss 时需要。HF 的 <code>generate</code> 传入 <code>logits_to_keep=1</code>，prefill 只对最后一个位置做这一步，图里也只亮这一行。`;
      const entry = this.entry(info, new Set(revealed), new Set(active));
      entry.rows = { shown: P, active: [0, P], labels: 0, ranges: new Map(onlyLast ? prod.map((n) => [n, [P - 1, P] as [number, number]]) : []) };
      entries.push(entry);
      if (k === lastCacheWrite) {
        push(
          {
            id: 'p:cache',
            title: `写入缓存：prompt 的 ${P} 行`,
            label: '写入缓存',
            text: `${tex(cacheNodes)} 算完以后存进缓存，prompt 的 ${P} 个位置全部写入。之后每个 decode 步的注意力都要用到前面所有位置的这些值，有了缓存就不必再算。${roped ? '存的是加过位置编码之后的值，读出来可以直接用，不需要再按位置旋转。' : ''}每个 token 在这一层缓存 ${perToken} 个数，缓存的大小随序列长度线性增长。`,
            nodes: cacheNodes,
            segment: 'prefill',
            sub: '缓存',
          },
          cacheNodes,
          { shown: P, active: [0, P], labels: 0 },
          [{ nodes: [cacheNodes[0]], index: zeros(cacheNodes[0], P - 1), text: `存入缓存的 ${P} 行`, side: 'right' }],
        );
      }
    });

    const runs: { cache: boolean; groups: Group[] }[] = [];
    for (const group of kept) {
      const cache = inCache(group);
      const last = runs[runs.length - 1];
      if (last !== undefined && last.cache === cache) last.groups.push(group);
      else runs.push({ cache, groups: [group] });
    }
    const totalWith = T;
    let totalWithout = P;
    for (let r = P; r < T; r++) totalWithout += r + 1;

    const sample = (row: number, phase: string) => {
      const V = infer.logits.shape[1];
      const tokens = infer.label.tokens ?? infer.input.tokens;
      const listed = tokens !== undefined && V <= 8 ? `$$${Array.from({ length: V }, (_, v) => `\\log p(\\text{${tokens[v]}})={{${infer.logits.name}[${row},${v}]:3}}`).join(',\\quad ')}$$` : '';
      const next = row + 1 < T ? `它被写进输入的第 ${row + 2} 个位置，成为下一个 decode 步唯一的新输入。` : `页面只画 ${T} 个位置，生成到这里停止。`;
      push(
        {
          id: `s:${row}`,
          title: `采样第 ${row + 2} 个 token`,
          label: '采样',
          text: `第 ${row + 1} 行的 $${infer.logits.tex}$ 是词表里每个 token 的对数概率：${listed}greedy 解码取最大的一项，得到 {{${label}[${row}]:token}}。${next}推理没有标签，图里 $${infer.label.tex}$ 这一格显示的就是采样结果。按温度 $\\tau$ 随机采样时，改为按 $\\operatorname{softmax}(\\mathbf{z}/\\tau)$ 的概率抽取，前面的计算不变。`,
          nodes: [infer.logits],
          segment: phase,
          sub: '采样',
        },
        [infer.logits, infer.label],
        { shown: row + 1, active: [row, row + 1], labels: row + 1 },
      );
    };
    sample(P - 1, 'prefill');

    for (let row = P; row < T; row++) {
      const phase = `decode ${row - P + 1}`;
      const detailed = row === P;
      const pos = row + 1;
      const cacheRange: [number, number] = [0, pos];
      runs.forEach((run, k) => {
        const parts = run.groups.map((group) => this.forwardGroup(group, entries.length));
        const prod = unique(parts.flatMap((part) => part.prod));
        const active = unique(parts.flatMap((part) => part.active));
        const subs = subsOf(parts);
        if (!run.cache) {
          const first = k === 0;
          const usesPosition = positional.filter((b) => active.includes(b));
          const writes = prod.filter((n) => cacheSet.has(n));
          const posText = usesPosition.length > 0 ? `位置相关的常量 ${tex(usesPosition)} 只取第 ${pos} 行，也就是位置 ${row}（从 0 数），它等于缓存里已有的行数。` : '';
          const text = first
            ? detailed
              ? `输入只有一个 token：上一步采样得到的 {{${name}[${row}]:token}}。注意力之前的运算（${subs}）都是逐行的，第 ${pos} 行的输出只取决于第 ${pos} 行的输入，所以前 ${row} 行在 prefill 里算出的结果不会变，这一步不再重算。矩阵乘变成一行向量乘整个权重矩阵，每个权重读进来只做一次乘加，计算量很小，耗时主要花在读权重上。实际的推理服务因此把许多请求各自的这一行拼成一批，一起乘同一份权重。${posText}`
              : `输入是 {{${name}[${row}]:token}}。和 decode 1 一样只算这一行，新算出的 ${tex(writes.length > 0 ? writes : cacheNodes)} 追加到缓存，缓存现在有 ${pos} 行。${posText}`
            : detailed
              ? `注意力只输出第 ${pos} 行，之后的运算（${subs}）又都是逐行的，同样只算这一行。最后的 $${infer.logits.tex}$ 也只有这一行，正好是采样需要的。`
              : `注意力之后的 ${subs} 只算第 ${pos} 行。`;
          const ranges: [Iterable<Node>, [number, number]][] = !detailed && first ? [[cacheNodes, [row, pos]]] : [];
          const notes: StepNote[] = [];
          if (detailed && first) {
            notes.push({ nodes: [infer.input], index: [row], text: '上一步采样的 token', side: 'left' });
            if (usesPosition.length > 0) notes.push({ nodes: [usesPosition[0]], index: zeros(usesPosition[0], row), text: `位置 ${row}`, side: 'left' });
          }
          push(
            {
              id: `d:${row}:${k}`,
              title: first ? `${phase}：第 ${pos} 个 token 进入模型` : `${phase}：注意力之后`,
              label: first ? '新 token' : '逐行',
              text,
              nodes: !detailed && first ? unique([...prod, ...cacheNodes]) : prod,
              segment: phase,
              sub: subs,
            },
            !detailed && first ? unique([...active, ...cacheNodes]) : active,
            rowsAt(row, ranges),
            notes,
          );
          if (detailed && first && writes.length > 0) {
            push(
              {
                id: `d:${row}:cache`,
                title: `${phase}：追加到缓存`,
                label: '追加缓存',
                text: `注意力在第 ${pos} 行需要前面所有位置的 ${tex(cacheNodes)}。前 ${row} 行在 prefill 里已经写进缓存（暗色），这一步只把刚算出的第 ${pos} 行（亮色）追加进去，缓存变成 ${pos} 行。${cacheWhy}`,
                nodes: cacheNodes,
                segment: phase,
                sub: '缓存',
              },
              cacheNodes,
              rowsAt(row),
              [
                { nodes: [cacheNodes[0]], index: zeros(cacheNodes[0], 0), text: `前 ${row} 行：从缓存读出`, side: 'right' },
                { nodes: [cacheNodes[0]], index: zeros(cacheNodes[0], row), text: '本步新算，追加', side: 'left' },
              ],
            );
            const recompute = unique([...prod, ...cacheNodes]);
            push(
              {
                id: `d:${row}:nocache`,
                title: `${phase}：如果不用缓存`,
                label: '不用缓存',
                text: `如果不存 ${tex(cacheNodes)}，第 ${pos} 行的注意力仍然需要前 ${row} 行的这些值，而它们取决于这一层和下面每一层的全部运算，只能把 ${pos} 行从头重算一遍。图里亮起的就是这一层要重算的部分，每一层都一样。按这一页的长度，从 ${P} 个 token 的 prompt 生成到 ${T} 个位置，用缓存一共算 ${totalWith} 行，不用缓存要算 ${totalWithout} 行；序列长度为 $n$ 时分别约为 $n$ 和 $n^2/2$。结果两者完全相同，缓存的代价是显存。`,
                nodes: recompute,
                segment: phase,
                sub: '对比',
              },
              recompute,
              { shown: pos, active: [0, pos], labels: row },
            );
          }
          return;
        }
        const cacheText = `缓存现在有 ${pos} 行`;
        if (!detailed) {
          const score = [...scores].find((n) => prod.includes(n) && n.shape.length === 2);
          push(
            {
              id: `d:${row}:${k}`,
              title: `${phase}：注意力读取 ${pos} 行缓存`,
              label: '注意力',
              text: `query 仍然只有第 ${pos} 行，但${cacheText}：分数行的长度变成 ${pos}，要从缓存读的也多了一行。注意力之外的部分每一步的计算量不变，注意力这部分随位置线性增长，这就是上下文越长、decode 越慢的原因。${score === undefined ? '' : rowValues(score, row)}`,
              nodes: prod,
              segment: phase,
              sub: subs,
            },
            unique([...active, ...cacheNodes]),
            rowsAt(row, [[[...cacheNodes, ...derived], cacheRange]]),
          );
          return;
        }
        const reads = parts.filter((part) => part.prod.every((n) => derived.has(n)));
        if (reads.length > 0) {
          const prodR = unique(reads.flatMap((part) => part.prod));
          push(
            {
              id: `d:${row}:${k}:read`,
              title: `${phase}：读出整个缓存`,
              label: '读缓存',
              text: `这几步（${reads.map((part) => part.info.title).join('、')}）只作用在缓存上，和 query 无关，所以缓存的 ${pos} 行全部参与。每个 decode 步都要把整个缓存读一遍，这部分的开销随位置增长。`,
              nodes: prodR,
              segment: phase,
              sub: subs,
            },
            unique([...reads.flatMap((part) => part.active), ...cacheNodes]),
            rowsAt(row, [[[...cacheNodes, ...derived], cacheRange]]),
          );
        }
        parts.forEach((part, j) => {
          if (reads.includes(part)) return;
          const first = part.prod[0];
          const mask = part.active.find((n) => masks.has(n));
          const isScore = part.prod.some((n) => scores.has(n)) && part.active.some((n) => derived.has(n));
          let text: string;
          if (isScore) {
            text = `query 只有第 ${pos} 行。它和缓存里每一行的 key 做点积，分数只有一行，长度是 ${pos}，不再是 prefill 时 $${P}\\times ${P}$ 的方阵。${rowValues(first, row)}`;
          } else if (mask !== undefined) {
            text = `第 ${pos} 行的 mask 是 $${vector(mask, row, pos, ':0')}$，对已有的 ${pos} 个位置全是 0。缓存里只有不晚于当前位置的 key，没有需要挡住的未来位置，所以 decode 实际上不需要 mask，加上的只是 0。${rowValues(first, row)}`;
          } else {
            text = `和 prefill 的同一步相同，只是只算第 ${pos} 行${scores.has(first) ? `，每行长度是 ${pos}` : ''}。${rowValues(first, row)}`;
          }
          push(
            {
              id: `d:${row}:${k}:${j}`,
              title: `${phase}：${part.info.title}`,
              label: part.info.label,
              text,
              nodes: part.prod,
              segment: phase,
              sub: part.info.sub ?? part.info.segment,
            },
            unique([...part.active, ...cacheNodes]),
            rowsAt(row, [[[...cacheNodes, ...derived], cacheRange]]),
          );
        });
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
      row: (node, position, key) => rowState(entry, this.spec.infer, node, position, key),
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

function rowState(entry: Entry, infer: InferSpec | undefined, node: Node, position: number, key: number | undefined): RowState | undefined {
  const rows = entry.rows;
  if (rows === undefined || infer === undefined) return undefined;
  if (node === infer.label) {
    if (position >= rows.labels || position < infer.prompt - 1) return 'hidden';
    return entry.active.has(node) && position !== rows.labels - 1 ? 'inactive' : undefined;
  }
  if (position >= rows.shown || (key !== undefined && key >= rows.shown)) return 'hidden';
  if (!entry.active.has(node)) return undefined;
  const [from, to] = rows.ranges?.get(node) ?? rows.active;
  return position >= from && position < to ? undefined : 'inactive';
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
