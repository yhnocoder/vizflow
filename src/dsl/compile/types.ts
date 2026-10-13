import type { Dtype, Node } from '../../autograd';
import type { ModuleDef } from '../load';
import type { Where } from '../source';

export type RealSize = (sizes: Readonly<Record<string, number>>, shapes: readonly number[][]) => number;

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

export type Value = Num | boolean | string | DtypeValue | ModuleRef | Node | Value[] | Instance | InstanceList | Result | TensorView | RealSizes | RealSizeEntry;

export type Member = Instance | InstanceList | Node;

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

export class Call {
  readonly values = new Map<string, Value>();
  readonly resultAliases = new Map<string, Map<string, Alias>>();
  readonly bound = new Map<string, Num | DtypeValue>();
  readonly inputs = new Map<string, Node>();
  readonly anon = new Map<Node, Node[]>();
  readonly children: Call[] = [];
  readonly outputs = new Map<string, Node>();
  readonly results: string[] = [];
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
  tex: string;
  anon: Node[];
  counter: number;
}
