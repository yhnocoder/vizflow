import type { Node } from '../../autograd';
import type { BackwardOverride, ForwardStep } from '../../steps';
import type { Nested } from '../../tensor';
import type { ModuleDef } from '../load';
import type { Where } from '../source';

export interface Num {
  toy: number;
  real: number;
}

export interface OpValue {
  op: string;
}

export interface BufferValue {
  buffer: true;
  shape: Num[];
  values: Nested;
}

export type ParamType = 'int' | 'float' | 'dim' | 'op' | 'tex' | 'text';
export type Value = Node | Num | Node[] | string | boolean | OpValue | BufferValue;
export type Resolved = Value | Scope;

export interface Make {
  path: string;
  tex: string;
  anon: Node[];
  counter: number;
}

export interface Env {
  scope: Scope;
  loops: Map<string, number>;
  where: Where;
  buffers?: boolean;
}

export interface ModuleRec {
  id: string;
  parent: string | null;
  title: string;
  subtitle: string;
  fixed: boolean;
  folded: boolean;
  split: boolean;
  nodes: Node[];
  fold: { forward: BackwardOverride; backward: BackwardOverride };
  decl: Record<string, unknown>;
  env: Env;
  defaults?: { decl: Record<string, unknown>; env: Env };
}

export interface StepRec {
  refs: { owner: string; id: string }[];
  step: ForwardStep;
}

export type Slot =
  | { kind: 'node'; name: string; node: Node; created: Node[] }
  | { kind: 'child'; def: ModuleDef; steps: StepRec[]; created: Node[]; merge?: Slot }
  | { kind: 'loop'; iterations: Slot[][] }
  | { kind: 'group'; decl: Record<string, unknown>; where: Where; loops: Map<string, number>; slots: Slot[] };

export class NeedDim {
  constructor(
    readonly name: string,
    readonly value: Num,
  ) {}
}

export class Scope {
  readonly entries = new Map<string, Node>();
  readonly children = new Map<string, Scope>();
  readonly params = new Map<string, { type: ParamType; value: Num | string | OpValue }>();
  readonly dims = new Map<string, Num>();
  readonly inputTex = new Map<string, string>();
  readonly anon = new Map<Node, Node[]>();
  readonly outputTex = new Map<string, string>();
  readonly texOverride = new Map<string, string>();
  readonly groups = new Set<string>();

  constructor(
    readonly def: ModuleDef,
    readonly path: string,
    readonly marks: string[],
  ) {}

  local(key: string): Node | Scope | undefined {
    return this.entries.get(key) ?? this.children.get(key);
  }
}
