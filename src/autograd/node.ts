import type { Nested, Tensor } from '../tensor';
import type { Graph } from './graph';

export type NodeKind = 'input' | 'param' | 'label' | 'buffer' | 'op';

export interface CardText {
  title: string;
  tex: string[];
  note: string;
}

export interface NodeOptions {
  tex?: string;
  elem?: string;
  card?: { forward?: Partial<CardText>; backward?: Partial<CardText> };
}

export interface NodeInternals {
  value(): Tensor;
  grad(): Tensor | null;
  set(values: Nested): void;
  transpose(): Node;
}

export interface NodeInit {
  id: number;
  graph: Graph;
  kind: NodeKind;
  name: string;
  op: string | undefined;
  inputs: readonly Node[];
  attrs: Readonly<Record<string, unknown>>;
  shape: readonly number[];
  realShape: readonly number[];
  tex: string;
  elem: string;
  tokens: readonly string[] | undefined;
  card: NodeOptions['card'];
  internals: NodeInternals;
}

export class Node {
  readonly id: number;
  readonly graph: Graph;
  readonly kind: NodeKind;
  readonly name: string;
  readonly op: string | undefined;
  readonly inputs: readonly Node[];
  readonly attrs: Readonly<Record<string, unknown>>;
  readonly shape: readonly number[];
  readonly realShape: readonly number[];
  readonly tex: string;
  readonly elem: string;
  readonly tokens: readonly string[] | undefined;
  readonly card: NodeOptions['card'];
  private readonly internals: NodeInternals;

  constructor(init: NodeInit) {
    this.id = init.id;
    this.graph = init.graph;
    this.kind = init.kind;
    this.name = init.name;
    this.op = init.op;
    this.inputs = init.inputs;
    this.attrs = init.attrs;
    this.shape = init.shape;
    this.realShape = init.realShape;
    this.tex = init.tex;
    this.elem = init.elem;
    this.tokens = init.tokens;
    this.card = init.card;
    this.internals = init.internals;
  }

  get value(): Tensor {
    return this.internals.value();
  }

  get grad(): Tensor | null {
    return this.internals.grad();
  }

  get T(): Node {
    return this.internals.transpose();
  }

  set(values: Nested): void {
    if (this.kind === 'op') {
      throw new Error(`node '${this.name}' is computed by op '${this.op}' and cannot be set`);
    }
    this.internals.set(values);
  }

  index(...idx: number[]): number {
    return Math.round(this.value.get(idx));
  }
}

export function defaultTex(name: string, shape: readonly number[]): { tex: string; elem: string } {
  if (/^[A-Za-z]$/.test(name)) {
    return { tex: shape.length === 0 ? name : `\\mathbf{${name}}`, elem: name };
  }
  return { tex: `\\mathrm{${name}}`, elem: `\\mathrm{${name}}` };
}

export function normalizeName(name: string): string {
  return name.replace(/\s+/g, '');
}

export function transposedTex(tex: string): string {
  return /[\^'_]/.test(tex) ? `{${tex}}^{\\top}` : `${tex}^{\\top}`;
}
