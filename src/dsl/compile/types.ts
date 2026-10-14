import type { Dtype, Node, ProgramBuilder, RealSize } from '../../autograd';
import type { Tensor } from '../../tensor';
import type { ModuleDef } from '../load';
import type { Where } from '../source';

export interface Num {
  toy: number;
  real: number;
  dim?: string;
  derive?: RealSize;
}

export interface DtypeValue {
  dtype: Dtype;
}

export interface ModuleRef {
  module: ModuleDef;
}

export type ViewIndex = number | { start: number; stop: number };

export interface TensorView {
  node: Node;
  grad: boolean;
  index: ViewIndex[];
}

export type RealSizeValues = Readonly<Record<string, number | string>>;

export interface RealSizes {
  realSizes: Readonly<Record<string, RealSizeValues>>;
}

export interface RealSizeEntry {
  realSize: string;
  values: RealSizeValues;
}

export class Eager {
  constructor(
    readonly value: Tensor,
    readonly dtype: Dtype,
    readonly ref: number | undefined,
    readonly tokens?: readonly string[],
  ) {}
}

export class PageFunction {
  constructor(
    readonly name: string,
    readonly run: (args: Value[], kwargs: ReadonlyMap<string, Value>, fail: (message: string) => never) => Value,
  ) {}
}

export type Value = Num | boolean | string | null | DtypeValue | ModuleRef | Node | Eager | PageFunction | Value[] | Instance | InstanceList | Result | TensorView | RealSizes | RealSizeEntry;

export type Member = Instance | InstanceList | Node | null;

export type Mode = { k: 'graph'; make: Make } | { k: 'ref' } | { k: 'eager'; program: ProgramBuilder; reads: boolean };

export class Instance {
  readonly members = new Map<string, Member>();
  readonly args = new Map<string, Value>();
  readonly calls: Call[] = [];

  constructor(
    readonly def: ModuleDef,
    readonly path: string,
    readonly name: string,
    readonly parent: Instance | undefined,
    readonly marks: readonly string[],
  ) {}
}

export class InstanceList {
  readonly items: Instance[] = [];
  readonly calls: ListCall[] = [];

  constructor(
    readonly path: string,
    readonly name: string,
    readonly count: Num,
  ) {}
}

export interface Alias {
  path: string;
  tex: string;
}

export interface Iterations {
  variables: string[];
  values: Map<string, Value>;
  before: Value | undefined;
}

export class Call {
  readonly values = new Map<string, Value>();
  readonly resultAliases = new Map<string, Map<string, Alias>>();
  readonly bound = new Map<string, Num | DtypeValue>();
  readonly inputs = new Map<string, Node>();
  readonly anon = new Map<Node, Node[]>();
  readonly children: Call[] = [];
  readonly outputs = new Map<string, Node>();
  readonly results: string[] = [];
  readonly resultValues: { name: string; value: Value; where: Where }[] = [];
  finished = false;
  readonly iterations = new Map<string, Iterations>();
  readonly lines: { nodes: Node[]; calls: Call[] }[] = [];
  created: Node[] = [];

  constructor(
    readonly instance: Instance,
    readonly prefix: string,
    readonly parent: Call | undefined,
    readonly aliases: ReadonlyMap<string, Alias>,
  ) {}
}

export class ListCall {
  readonly calls: Call[] = [];

  constructor(readonly list: InstanceList) {}
}

export class Result {
  constructor(
    readonly call: Call,
    readonly outputs: ReadonlyMap<string, Node>,
    readonly none: ReadonlySet<string> = new Set(),
  ) {}
}

export interface Env {
  instance: Instance;
  call: Call | undefined;
  locals: ReadonlyMap<string, Value>;
  where: Where;
}

export interface Make {
  name: string;
  path: string;
  suffix?: string;
  tex: string;
  anon: Node[];
}
