import type { Graph, Node } from '../autograd';
import { explain } from '../explain';
import { compileText } from './template';

export interface ForwardStep {
  show: Node | Node[];
  active?: Node | Node[];
  title: string;
  text: string;
  label?: string;
}

export interface BackwardOverride {
  title?: string;
  text?: string;
  label?: string;
}

export type BackwardSpec = 'auto' | { auto: true; merge?: Node[][]; override?: Record<number, BackwardOverride> };

export interface StepsSpec {
  forward: ForwardStep[];
  overview?: { title: string; text: string; label?: string };
  backward?: BackwardSpec;
}

export interface StepInfo {
  dir: 'forward' | 'backward';
  index: number;
  label: string;
  title: string;
  text: string;
  nodes: Node[];
}

export interface StepState {
  dir: 'forward' | 'backward';
  index: number;
  focus: number;
  revealed(node: Node): boolean;
  active(node: Node): boolean;
}

export interface Steps {
  readonly graph: Graph;
  readonly list: readonly StepInfo[];
  readonly visible: readonly Node[];
  readonly current: number;
  readonly focus: number;
  go(i: number): void;
  next(): void;
  prev(): void;
  setFocus(b: number): void;
  state(): StepState;
  subscribe(fn: () => void): () => void;
  text(i?: number): { title: string; body: string };
  dispose(): void;
}

const DEFAULT_OVERVIEW_TITLE = '总览';

const toList = (nodes: Node | Node[] | undefined): Node[] => (nodes === undefined ? [] : Array.isArray(nodes) ? nodes : [nodes]);

function byId(nodes: Iterable<Node>): Node[] {
  return [...new Set(nodes)].sort((a, b) => a.id - b.id);
}

interface Entry {
  info: StepInfo;
  revealed: Set<Node>;
  active: Set<Node>;
  title: (focus: number) => string;
  body: (focus: number) => string;
}

function backwardGroups(g: Graph, visible: readonly Node[], spec: Exclude<BackwardSpec, 'auto'>): Node[][] {
  const loss = g.loss();
  const ordered = visible.filter((n) => n.kind !== 'label' && n.id <= loss.id).sort((a, b) => b.id - a.id);
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

function backwardEntries(g: Graph, visible: readonly Node[], spec: BackwardSpec): StepInfo[] {
  const resolved = spec === 'auto' ? { auto: true as const } : spec;
  const loss = g.loss();
  const explainer = explain(g);
  const infos = backwardGroups(g, visible, resolved).map((nodes, i): StepInfo => {
    const subject = nodes.find((n) => n !== loss) ?? loss;
    const card = explainer.nodeCard(subject, 'backward');
    const override = resolved.override?.[i + 1] ?? {};
    return {
      dir: 'backward',
      index: i + 1,
      title: override.title ?? `∂ℓ/∂${subject.name}`,
      label: override.label ?? `∂${subject.name}`,
      text: override.text ?? card.tex.map((tex) => `$$${tex}$$`).join(''),
      nodes,
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

class StepsImpl implements Steps {
  readonly list: readonly StepInfo[];
  readonly visible: readonly Node[];
  current = 0;
  focus = 0;
  private readonly entries: Entry[];
  private readonly subscribers: (() => void)[] = [];

  constructor(
    readonly graph: Graph,
    spec: StepsSpec,
  ) {
    const g = graph;
    this.visible = Object.freeze(byId(spec.forward.flatMap((step) => toList(step.show))));
    const entries: Entry[] = [];
    const overview = spec.overview ?? { title: DEFAULT_OVERVIEW_TITLE, text: '' };
    const all = new Set(this.visible);
    entries.push(this.entry(g, { dir: 'forward', index: 0, title: overview.title, label: overview.label ?? overview.title, text: overview.text, nodes: [...this.visible] }, all, all));
    const shown = new Set<Node>();
    spec.forward.forEach((step, k) => {
      const show = toList(step.show);
      show.forEach((n) => shown.add(n));
      const active = new Set(step.active === undefined ? show : toList(step.active));
      const info: StepInfo = { dir: 'forward', index: k + 1, title: step.title, label: step.label ?? step.title, text: step.text, nodes: show };
      entries.push(this.entry(g, info, new Set(shown), active));
    });
    const revealed = new Set<Node>();
    for (const info of backwardEntries(g, this.visible, spec.backward ?? 'auto')) {
      info.nodes.forEach((n) => revealed.add(n));
      entries.push(this.entry(g, info, new Set(revealed), new Set(info.nodes)));
    }
    this.entries = entries;
    this.list = Object.freeze(entries.map((e) => e.info));
  }

  private entry(g: Graph, info: StepInfo, revealed: Set<Node>, active: Set<Node>): Entry {
    return { info, revealed, active, title: compileText(g, info.title), body: compileText(g, info.text) };
  }

  go(i: number): void {
    const next = Math.max(0, Math.min(this.list.length - 1, i));
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
    if (b === this.focus) return;
    this.focus = b;
    this.notify();
  }

  state(): StepState {
    const entry = this.entries[this.current];
    return {
      dir: entry.info.dir,
      index: entry.info.index,
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
    const entry = this.entries[i];
    if (entry === undefined) throw new Error(`step ${i} does not exist (there are ${this.entries.length} steps)`);
    return { title: entry.title(this.focus), body: entry.body(this.focus) };
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
  return { dir: 'forward', index: 0, focus: 0, revealed: () => true, active: () => true };
}
