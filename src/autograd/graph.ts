import { type Nested, Tensor, formatShape, inferShape, normDim } from '../tensor';
import { checkOpNamed } from './check';
import { checkByDefault } from './check-default';
import { DEFAULT_DTYPE, type Dtype, LABEL_DTYPE, checkDtype, promoteDtypes } from './dtype';
import { Node, type NodeInit, type NodeKind, type NodeOptions, defaultTex, normalizeName, transposedTex } from './node';
import type { AnyOpDef } from './op';
import type { BuiltinOps } from './ops';
import { opDef } from './registry';

export interface GraphOptions {
  dims?: Record<string, number>;
  real?: { name?: string } & Record<string, number | string>;
  check?: boolean;
  dtype?: string;
}

export interface OpOptions extends NodeOptions {
  name?: string;
}

export interface LabelOptions extends NodeOptions {
  tokens?: string[];
  classes?: number;
  shape?: (number | string)[];
}

export interface SplitOptions {
  names?: string[];
  tex?: string[];
  elem?: string[];
}

export interface Graph extends BuiltinOps {
  readonly dims: Readonly<Record<string, number>>;
  readonly real: Readonly<Record<string, number | string>> | undefined;
  readonly dtype: Dtype;
  readonly nodes: readonly Node[];
  readonly version: number;
  input(name: string, shape: (number | string)[], values: Nested, options?: NodeOptions): Node;
  param(name: string, shape: (number | string)[], values: Nested, options?: NodeOptions): Node;
  label(name: string, values: Nested, options?: LabelOptions): Node;
  buffer(name: string, shape: (number | string)[], values: Nested, options?: NodeOptions): Node;
  split(tensor: Node, split_size: number | string, dim?: number | SplitOptions, options?: SplitOptions): Node[];
  loss(node?: Node): Node;
  findLoss(): Node | undefined;
  addDim(name: string, value: number, real?: number): void;
  node(name: string): Node | undefined;
  subscribe(fn: () => void): () => void;
  batch(fn: () => void): void;
  reset(): void;
  jacobian(out: Node, input: Node, options?: { row?: number }): Tensor;
}

interface NodeState {
  value: Tensor;
  grad: Tensor;
  initial: Tensor | undefined;
  transposed: Node | undefined;
}

const NODE_OPTION_KEYS = new Set(['name', 'tex', 'elem', 'card', 'dtype']);
const SPLIT_OPTION_KEYS = ['names', 'tex', 'elem'] as const;
const CONSTANT_KINDS: ReadonlySet<NodeKind> = new Set(['label', 'buffer']);

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value) && !(value instanceof Node);
}

function resolveShape(spec: readonly (number | string)[], table: Readonly<Record<string, number>>, name: string): number[] {
  return spec.map((d) => {
    const size = typeof d === 'string' ? table[d] : d;
    if (typeof d === 'string' && size === undefined) {
      throw new Error(`node '${name}': dimension '${d}' is not defined`);
    }
    if (!Number.isInteger(size) || size < 0) {
      throw new Error(`node '${name}': dimension ${JSON.stringify(d)} is not a non-negative integer`);
    }
    return size;
  });
}

function resolveSizes(def: AnyOpDef, attrs: Record<string, unknown>, table: Readonly<Record<string, number>>): Record<string, unknown> {
  const resolved = { ...attrs };
  for (const key of def.sizes ?? []) {
    const lookup = (value: unknown) => {
      if (typeof value !== 'string') return value;
      if (table[value] === undefined) throw new Error(`${def.name}: dimension '${value}' is not defined`);
      return table[value];
    };
    const value = attrs[key];
    resolved[key] = Array.isArray(value) ? value.map(lookup) : lookup(value);
  }
  return resolved;
}

class GraphImpl {
  dims: Readonly<Record<string, number>>;
  real: Readonly<Record<string, number | string>> | undefined;
  readonly dtype: Dtype;
  readonly nodes: Node[] = [];
  version = 0;
  self!: Graph;
  private readonly states: NodeState[] = [];
  private realDims: Readonly<Record<string, number>> | undefined;
  private readonly check: boolean;
  private readonly subscribers: (() => void)[] = [];
  private readonly opCounters = new Map<string, number>();
  private lossNode: Node | undefined;
  private backwardValid = false;
  private batchDepth = 0;
  private pending = false;
  private readonly touched = new Map<number, Tensor>();
  private detachedKey = '';
  private detached = new Set<number>();

  constructor(options: GraphOptions) {
    this.dims = Object.freeze({ ...(options.dims ?? {}) });
    this.real = options.real === undefined ? undefined : Object.freeze({ ...options.real });
    this.check = options.check ?? checkByDefault();
    this.dtype = options.dtype === undefined ? DEFAULT_DTYPE : checkDtype(options.dtype, 'graph');
    if (this.real !== undefined) {
      const numeric: Record<string, number> = { ...this.dims };
      for (const [key, value] of Object.entries(this.real)) {
        if (typeof value === 'number') numeric[key] = value;
      }
      this.realDims = numeric;
    }
  }

  addDim(name: string, value: number, real = value): void {
    const known = this.dims[name];
    if (known !== undefined) {
      if (known !== value || (this.realDims?.[name] ?? known) !== real) throw new Error(`addDim: dimension '${name}' is already defined`);
      return;
    }
    this.dims = Object.freeze({ ...this.dims, [name]: value });
    if (this.real !== undefined && this.realDims !== undefined) {
      this.real = Object.freeze({ ...this.real, [name]: real });
      this.realDims = Object.freeze({ ...this.realDims, [name]: real });
    }
  }

  input(name: string, shape: (number | string)[], values: Nested, options: NodeOptions = {}): Node {
    return this.leaf('input', name, shape, values, options, undefined);
  }

  param(name: string, shape: (number | string)[], values: Nested, options: NodeOptions = {}): Node {
    return this.leaf('param', name, shape, values, options, undefined);
  }

  label(name: string, values: Nested, options: LabelOptions = {}): Node {
    const shape = options.shape ?? inferShape(values);
    const classes = options.classes ?? options.tokens?.length;
    const attrs = classes === undefined ? {} : { classes };
    return this.leaf('label', name, shape, values, options, options.tokens, attrs);
  }

  buffer(name: string, shape: (number | string)[], values: Nested, options: NodeOptions = {}): Node {
    return this.leaf('buffer', name, shape, values, options, undefined);
  }

  split(tensor: Node, splitSize: number | string, dim: number | SplitOptions = 0, options: SplitOptions = {}): Node[] {
    if (isPlainObject(dim)) return this.split(tensor, splitSize, 0, dim as SplitOptions);
    this.ownNode(tensor, 'split');
    for (const key of Object.keys(options)) {
      if (!(SPLIT_OPTION_KEYS as readonly string[]).includes(key)) throw new Error(`split: unknown option '${key}'`);
    }
    const size = typeof splitSize === 'string' ? this.dims[splitSize] : splitSize;
    if (size === undefined) throw new Error(`split: dimension '${splitSize}' is not defined`);
    if (!Number.isInteger(size) || size <= 0) throw new Error(`split: split_size must be a positive integer, got ${size}`);
    const total = tensor.shape[normDim(dim as number, tensor.shape.length)];
    const count = Math.ceil(total / size);
    for (const key of SPLIT_OPTION_KEYS) {
      const given = options[key];
      if (given !== undefined && given.length !== count) {
        throw new Error(`split: ${key} has ${given.length} entries for ${count} chunks`);
      }
    }
    const names = Array.from({ length: count }, (_, i) => options.names?.[i] ?? `${tensor.name}_${i + 1}`);
    names.forEach((name, i) => {
      if (this.node(name) !== undefined || names.slice(0, i).some((n) => normalizeName(n) === normalizeName(name))) {
        throw new Error(`node '${name}' already exists`);
      }
    });
    return names.map((name, i) => {
      const start = i * size;
      const length = start + size <= total ? splitSize : total - start;
      const nodeOptions: OpOptions = { name };
      if (options.tex !== undefined) nodeOptions.tex = options.tex[i];
      if (options.elem !== undefined) nodeOptions.elem = options.elem[i];
      return this.self.narrow(tensor, dim as number, start, length, nodeOptions);
    });
  }

  loss(node?: Node): Node {
    if (node !== undefined) {
      this.ownNode(node, 'loss');
      if (this.states[node.id].value.size !== 1) {
        throw new Error(`loss node '${node.name}' must have one element, got shape ${formatShape(node.shape)}`);
      }
      this.lossNode = node;
      this.backwardValid = false;
      return node;
    }
    const current = this.findLoss();
    if (current === undefined) throw new Error('graph has no loss node');
    return current;
  }

  findLoss(): Node | undefined {
    if (this.lossNode !== undefined) return this.lossNode;
    const { nodes, states } = this;
    for (let i = nodes.length - 1; i >= 0; i--) {
      const node = nodes[i];
      if (node.kind === 'op' && states[i].value.size === 1) return node;
    }
    return undefined;
  }

  node(name: string): Node | undefined {
    const wanted = normalizeName(name);
    return this.nodes.find((n) => normalizeName(n.name) === wanted);
  }

  subscribe(fn: () => void): () => void {
    this.subscribers.push(fn);
    return () => {
      const i = this.subscribers.indexOf(fn);
      if (i >= 0) this.subscribers.splice(i, 1);
    };
  }

  batch(fn: () => void): void {
    this.batchDepth++;
    try {
      fn();
    } catch (error) {
      if (this.batchDepth === 1) this.rollbackLeaves();
      throw error;
    } finally {
      this.batchDepth--;
    }
    if (this.batchDepth === 0 && this.pending) this.recompute();
  }

  reset(): void {
    this.batch(() => {
      this.states.forEach((state, id) => {
        if (state.initial === undefined) return;
        this.replaceLeaf(id, state.initial.clone());
      });
    });
  }

  apply(def: AnyOpDef, args: unknown[]): Node {
    const positional = [...args];
    const last = positional[positional.length - 1];
    const options: Record<string, unknown> = isPlainObject(last) ? (positional.pop() as Record<string, unknown>) : {};
    const restAt = def.rest === undefined ? -1 : def.args.indexOf(def.rest);
    if (restAt >= 0 && positional.length > restAt && !(positional.length === restAt + 1 && Array.isArray(positional[restAt]))) {
      positional.splice(restAt, positional.length - restAt, positional.slice(restAt));
    }
    if (positional.length > def.args.length) {
      throw new Error(`${def.name}: takes at most ${def.args.length} positional arguments, got ${positional.length}`);
    }
    const attrs: Record<string, unknown> = { ...(def.defaults ?? {}) };
    const inputs: Node[] = [];
    def.args.forEach((argName, i) => {
      const given = positional[i];
      if (argName === def.list) {
        if (!Array.isArray(given) || given.length === 0 || !given.every((n) => n instanceof Node)) {
          throw new Error(`${def.name}: argument '${argName}' must be a non-empty array of nodes`);
        }
        for (const n of given as Node[]) this.ownNode(n, def.name);
        inputs.push(...(given as Node[]));
      } else if (def.tensors.includes(argName)) {
        const scalar = def.scalars?.includes(argName) ?? false;
        if (scalar && typeof given === 'number') {
          attrs[argName] = given;
          return;
        }
        if (!(given instanceof Node)) throw new Error(`${def.name}: argument '${argName}' must be a node${scalar ? ' or a number' : ''}`);
        this.ownNode(given, def.name);
        inputs.push(given);
      } else if (given !== undefined) {
        attrs[argName] = given;
      } else if (!(argName in attrs)) {
        throw new Error(`${def.name}: missing argument '${argName}'`);
      }
    });
    const nodeOptions: OpOptions = {};
    for (const [key, value] of Object.entries(options)) {
      if (NODE_OPTION_KEYS.has(key)) (nodeOptions as Record<string, unknown>)[key] = value;
      else if (def.defaults !== undefined && key in def.defaults) attrs[key] = value;
      else throw new Error(`${def.name}: unknown option '${key}'`);
    }
    return this.createOp(def, inputs, attrs, nodeOptions);
  }

  private leaf(
    kind: NodeKind,
    name: string,
    shapeSpec: readonly (number | string)[],
    values: Nested,
    options: NodeOptions,
    tokens: string[] | undefined,
    attrs: Record<string, unknown> = {},
  ): Node {
    const shape = resolveShape(shapeSpec, this.dims, name);
    const realShape = this.realDims === undefined ? shape : resolveShape(shapeSpec, this.realDims, name);
    const value = this.tensorFor(name, shape, values);
    const state: NodeState = { value, grad: Tensor.zeros(shape), initial: value.clone(), transposed: undefined };
    const dtype = kind === 'label' ? LABEL_DTYPE : this.dtype;
    return this.addNode(kind, name, undefined, [], attrs, shape, realShape, options, dtype, tokens, state);
  }

  private createOp(def: AnyOpDef, inputs: Node[], given: Record<string, unknown>, options: OpOptions): Node {
    const attrs = resolveSizes(def, given, this.dims);
    const shape = def.shape(inputs.map((n) => [...n.shape]), attrs);
    const realShape =
      this.realDims === undefined ? shape : def.shape(inputs.map((n) => [...n.realShape]), resolveSizes(def, given, this.realDims));
    const name = options.name ?? this.autoName(def.name);
    if (this.check) checkOpNamed(def, inputs.map((n) => n.value.clone()), attrs, name);
    const value = def.forward(inputs.map((n) => n.value), attrs);
    const state: NodeState = { value, grad: Tensor.zeros(shape), initial: undefined, transposed: undefined };
    const dtype = options.dtype === undefined ? promoteDtypes(inputs.map((n) => n.dtype), this.dtype, `${def.name} '${name}'`) : checkDtype(options.dtype, `node '${name}'`);
    return this.addNode('op', name, def.name, inputs, attrs, shape, realShape, options, dtype, undefined, state);
  }

  private addNode(
    kind: NodeKind,
    name: string,
    op: string | undefined,
    inputs: Node[],
    attrs: Record<string, unknown>,
    shape: number[],
    realShape: number[],
    options: NodeOptions,
    dtype: Dtype,
    tokens: string[] | undefined,
    state: NodeState,
  ): Node {
    if (this.node(name) !== undefined) throw new Error(`node '${name}' already exists`);
    const id = this.nodes.length;
    const defaults = defaultTex(name, shape);
    const init: NodeInit = {
      id,
      graph: this.self,
      kind,
      name,
      op,
      inputs: Object.freeze([...inputs]),
      attrs: Object.freeze({ ...attrs }),
      shape: Object.freeze([...shape]),
      realShape: Object.freeze([...realShape]),
      dtype: options.dtype === undefined ? dtype : checkDtype(options.dtype, `node '${name}'`),
      tex: options.tex ?? defaults.tex,
      elem: options.elem ?? defaults.elem,
      tokens: tokens === undefined ? undefined : Object.freeze([...tokens]),
      card: options.card,
      internals: {
        value: () => this.states[id].value,
        grad: () => this.gradOf(id),
        set: (values) => this.setLeaf(id, values),
        transpose: () => this.transposeOf(id),
      },
    };
    const node = new Node(init);
    this.nodes.push(node);
    this.states.push(state);
    this.backwardValid = false;
    return node;
  }

  private autoName(op: string): string {
    let count = this.opCounters.get(op) ?? 0;
    let name: string;
    do {
      count++;
      name = `${op}${count}`;
    } while (this.node(name) !== undefined);
    this.opCounters.set(op, count);
    return name;
  }

  private ownNode(node: Node, op: string): void {
    if (node.graph !== this.self) throw new Error(`${op}: node '${node.name}' belongs to another graph`);
  }

  private tensorFor(name: string, shape: number[], values: Nested): Tensor {
    try {
      return Tensor.from(values, shape);
    } catch (error) {
      throw new Error(`node '${name}': ${(error as Error).message}`);
    }
  }

  private gradOf(id: number): Tensor | null {
    if (CONSTANT_KINDS.has(this.nodes[id].kind) || this.isDetached(id)) return null;
    this.ensureBackward();
    return this.states[id].grad;
  }

  private setLeaf(id: number, values: Nested): void {
    const node = this.nodes[id];
    const value = this.tensorFor(node.name, [...node.shape], values);
    this.replaceLeaf(id, value);
    if (this.batchDepth === 0) this.recompute();
  }

  private replaceLeaf(id: number, value: Tensor): void {
    if (!this.touched.has(id)) this.touched.set(id, this.states[id].value);
    this.states[id].value = value;
    this.pending = true;
  }

  private rollbackLeaves(): void {
    this.touched.forEach((value, id) => {
      this.states[id].value = value;
    });
    this.touched.clear();
    this.pending = false;
  }

  private transposeOf(id: number): Node {
    const state = this.states[id];
    if (state.transposed !== undefined) return state.transposed;
    const node = this.nodes[id];
    if (node.shape.length !== 2) {
      throw new Error(`transpose: node '${node.name}' with shape ${formatShape(node.shape)} is not two-dimensional`);
    }
    state.transposed = this.self.transpose(node, 0, 1, {
      name: `${node.name}.T`,
      tex: transposedTex(node.tex),
      elem: transposedTex(node.elem),
    });
    return state.transposed;
  }

  private recompute(): void {
    const saved = this.states.map((state) => ({ value: state.value, grad: state.grad }));
    const savedBackwardValid = this.backwardValid;
    try {
      this.nodes.forEach((node, i) => {
        if (node.kind !== 'op') return;
        const def = opDef(node.op as string) as AnyOpDef;
        this.states[i].value = def.forward(node.inputs.map((n) => n.value), node.attrs);
      });
      this.computeBackward();
      if (this.check) {
        this.nodes.forEach((node) => {
          if (node.kind !== 'op') return;
          const def = opDef(node.op as string) as AnyOpDef;
          checkOpNamed(def, node.inputs.map((n) => n.value.clone()), node.attrs, node.name);
        });
      }
    } catch (error) {
      saved.forEach((snapshot, i) => {
        this.states[i].value = snapshot.value;
        this.states[i].grad = snapshot.grad;
      });
      this.backwardValid = savedBackwardValid;
      this.rollbackLeaves();
      throw error;
    }
    this.touched.clear();
    this.pending = false;
    this.version++;
    [...this.subscribers].forEach((fn) => fn());
  }

  private isDetached(id: number): boolean {
    const loss = this.findLoss();
    const key = `${this.nodes.length}:${loss?.id ?? -1}`;
    if (key !== this.detachedKey) {
      this.detached = this.detachedOps(loss);
      this.detachedKey = key;
    }
    return this.detached.has(id);
  }

  private detachedOps(loss: Node | undefined): Set<number> {
    const differentiableInputs = (node: Node): Node[] => {
      const flags = node.kind === 'op' ? opDef(node.op as string)?.differentiable : undefined;
      return node.inputs.filter((_, k) => flags === undefined || flags[k]);
    };
    const tracked = this.nodes.map((node) => node.kind === 'param' || node.kind === 'input');
    for (const node of this.nodes) {
      if (node.kind === 'op') tracked[node.id] = differentiableInputs(node).some((input) => tracked[input.id]);
    }
    const reached = this.nodes.map((node) => loss === undefined || node === loss);
    if (loss !== undefined) {
      for (let i = loss.id; i >= 0; i--) {
        if (reached[i]) for (const input of differentiableInputs(this.nodes[i])) reached[input.id] = true;
      }
    }
    return new Set(this.nodes.filter((node) => node.kind === 'op' && !(tracked[node.id] && reached[node.id])).map((node) => node.id));
  }

  private ensureBackward(): void {
    if (!this.backwardValid) this.computeBackward();
  }

  jacobian(out: Node, input: Node, options: { row?: number } = {}): Tensor {
    this.ownNode(out, 'jacobian');
    this.ownNode(input, 'jacobian');
    const { row } = options;
    if (row === undefined) return this.jacobianBlock(out, input, 0, this.states[out.id].value.size, 0, this.states[input.id].value.size, [...out.shape, ...input.shape]);
    const rows = out.shape[0];
    if (out.shape.length === 0 || input.shape.length === 0 || input.shape[0] !== rows) {
      throw new Error(`jacobian: row needs matching first dimensions, got ${formatShape(out.shape)} and ${formatShape(input.shape)}`);
    }
    if (!Number.isInteger(row) || row < 0 || row >= rows) throw new Error(`jacobian: row ${row} is out of range for ${rows} rows`);
    const outRow = this.states[out.id].value.size / rows;
    const inRow = this.states[input.id].value.size / rows;
    return this.jacobianBlock(out, input, row * outRow, outRow, row * inRow, inRow, [...out.shape.slice(1), ...input.shape.slice(1)]);
  }

  private jacobianBlock(out: Node, input: Node, outStart: number, outCount: number, inStart: number, inCount: number, shape: number[]): Tensor {
    const result = Tensor.zeros(shape);
    for (let i = 0; i < outCount; i++) {
      const seed = Tensor.zeros(out.shape);
      seed.data[outStart + i] = 1;
      const grad = this.propagate(out, seed)[input.id];
      result.data.set(grad.data.subarray(inStart, inStart + inCount), i * inCount);
    }
    return result;
  }

  private computeBackward(): void {
    this.backwardValid = true;
    const loss = this.findLoss();
    if (loss === undefined) {
      this.states.forEach((state, i) => {
        state.grad = Tensor.zeros(this.nodes[i].shape);
      });
      return;
    }
    const seed = Tensor.zeros(loss.shape);
    seed.data.fill(1);
    this.propagate(loss, seed).forEach((grad, i) => {
      this.states[i].grad = grad;
    });
  }

  private propagate(root: Node, seed: Tensor): Tensor[] {
    const grads = this.nodes.map((node) => Tensor.zeros(node.shape));
    grads[root.id] = seed;
    const upstream = new Set<Node>([root]);
    for (let i = root.id; i >= 0; i--) {
      const node = this.nodes[i];
      if (!upstream.has(node)) continue;
      for (const input of node.inputs) upstream.add(input);
      if (node.kind !== 'op') continue;
      const def = opDef(node.op as string) as AnyOpDef;
      const inputGrads = def.backward(grads[i], node.inputs.map((n) => n.value), this.states[i].value, node.attrs);
      node.inputs.forEach((input, k) => {
        if (CONSTANT_KINDS.has(input.kind)) return;
        if (def.differentiable !== undefined && !def.differentiable[k]) return;
        const grad = inputGrads[k];
        if (grad === null || grad === undefined) return;
        if (formatShape(grad.shape) !== formatShape(input.shape)) {
          throw new Error(
            `${def.name}: backward of node '${node.name}' returned ${formatShape(grad.shape)} for input '${input.name}' with shape ${formatShape(input.shape)}`,
          );
        }
        const target = grads[input.id].data;
        for (let j = 0; j < target.length; j++) target[j] += grad.data[j];
      });
    }
    return grads;
  }
}

export function graph(options: GraphOptions = {}): Graph {
  const impl = new GraphImpl(options);
  const proxy = new Proxy(impl, {
    get(target, prop, receiver) {
      if (typeof prop === 'string' && !(prop in target)) {
        const def = opDef(prop);
        if (def !== undefined) return (...args: unknown[]) => target.apply(def, args);
      }
      return Reflect.get(target, prop, receiver);
    },
    has(target, prop) {
      return prop in target || (typeof prop === 'string' && opDef(prop) !== undefined);
    },
  }) as unknown as Graph;
  impl.self = proxy;
  return proxy;
}
