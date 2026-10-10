import { type Nested, Tensor, formatShape, inferShape } from '../tensor';
import { checkOpNamed } from './check';
import { Node, type NodeInit, type NodeKind, type NodeOptions, defaultTex, normalizeName } from './node';
import type { AnyOpDef } from './op';
import type { BuiltinOps } from './ops';
import { opDef } from './registry';

export interface GraphOptions {
  dims?: Record<string, number>;
  real?: { name?: string } & Record<string, number | string>;
  check?: boolean;
}

export interface OpOptions extends NodeOptions {
  name?: string;
}

export interface LabelOptions extends NodeOptions {
  tokens?: string[];
  classes?: number;
}

export interface Graph extends BuiltinOps {
  readonly dims: Readonly<Record<string, number>>;
  readonly real: Readonly<Record<string, number | string>> | undefined;
  readonly nodes: readonly Node[];
  readonly version: number;
  input(name: string, shape: (number | string)[], values: Nested, options?: NodeOptions): Node;
  param(name: string, shape: (number | string)[], values: Nested, options?: NodeOptions): Node;
  label(name: string, values: Nested, options?: LabelOptions): Node;
  loss(node?: Node): Node;
  node(name: string): Node | undefined;
  subscribe(fn: () => void): () => void;
  batch(fn: () => void): void;
  reset(): void;
}

interface NodeState {
  value: Tensor;
  grad: Tensor;
  initial: Tensor | undefined;
  transposed: Node | undefined;
}

const NODE_OPTION_KEYS = new Set(['name', 'tex', 'elem', 'card']);

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

class GraphImpl {
  readonly dims: Readonly<Record<string, number>>;
  readonly real: Readonly<Record<string, number | string>> | undefined;
  readonly nodes: Node[] = [];
  version = 0;
  self!: Graph;
  private readonly states: NodeState[] = [];
  private readonly realDims: Readonly<Record<string, number>> | undefined;
  private readonly check: boolean;
  private readonly subscribers: (() => void)[] = [];
  private readonly opCounters = new Map<string, number>();
  private lossNode: Node | undefined;
  private backwardValid = false;
  private batchDepth = 0;
  private pending = false;

  constructor(options: GraphOptions) {
    this.dims = Object.freeze({ ...(options.dims ?? {}) });
    this.real = options.real === undefined ? undefined : Object.freeze({ ...options.real });
    this.check = options.check ?? import.meta.env.DEV;
    if (this.real !== undefined) {
      const numeric: Record<string, number> = { ...this.dims };
      for (const [key, value] of Object.entries(this.real)) {
        if (typeof value === 'number') numeric[key] = value;
      }
      this.realDims = numeric;
    }
  }

  input(name: string, shape: (number | string)[], values: Nested, options: NodeOptions = {}): Node {
    return this.leaf('input', name, shape, values, options, undefined);
  }

  param(name: string, shape: (number | string)[], values: Nested, options: NodeOptions = {}): Node {
    return this.leaf('param', name, shape, values, options, undefined);
  }

  label(name: string, values: Nested, options: LabelOptions = {}): Node {
    const shape = inferShape(values);
    const classes = options.classes ?? options.tokens?.length;
    const attrs = classes === undefined ? {} : { classes };
    return this.leaf('label', name, shape, values, options, options.tokens, attrs);
  }

  loss(node?: Node): Node {
    if (node !== undefined) {
      this.ownNode(node, 'loss');
      this.lossNode = node;
      this.backwardValid = false;
      return node;
    }
    const current = this.resolveLoss();
    if (current === undefined) throw new Error('graph has no loss node');
    return current;
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
    } finally {
      this.batchDepth--;
    }
    if (this.batchDepth === 0 && this.pending) this.recompute();
  }

  reset(): void {
    this.batch(() => {
      this.states.forEach((state) => {
        if (state.initial === undefined) return;
        state.value = state.initial.clone();
        this.pending = true;
      });
    });
  }

  apply(def: AnyOpDef, args: unknown[]): Node {
    const positional = [...args];
    const last = positional[positional.length - 1];
    const options: Record<string, unknown> = isPlainObject(last) ? (positional.pop() as Record<string, unknown>) : {};
    if (positional.length > def.args.length) {
      throw new Error(`${def.name}: takes at most ${def.args.length} positional arguments, got ${positional.length}`);
    }
    const attrs: Record<string, unknown> = { ...(def.defaults ?? {}) };
    const inputs: Node[] = [];
    def.args.forEach((argName, i) => {
      const given = positional[i];
      if (def.tensors.includes(argName)) {
        if (!(given instanceof Node)) throw new Error(`${def.name}: argument '${argName}' must be a node`);
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
    return this.addNode(kind, name, undefined, [], attrs, shape, realShape, options, tokens, state);
  }

  private createOp(def: AnyOpDef, inputs: Node[], attrs: Record<string, unknown>, options: OpOptions): Node {
    const shape = def.shape(inputs.map((n) => [...n.shape]), attrs);
    const realShape = this.realDims === undefined ? shape : def.shape(inputs.map((n) => [...n.realShape]), attrs);
    const name = options.name ?? this.autoName(def.name);
    if (this.check) checkOpNamed(def, inputs.map((n) => n.value.clone()), attrs, name);
    const value = def.forward(inputs.map((n) => n.value), attrs);
    const state: NodeState = { value, grad: Tensor.zeros(shape), initial: undefined, transposed: undefined };
    return this.addNode('op', name, def.name, inputs, attrs, shape, realShape, options, undefined, state);
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
    if (this.nodes[id].kind === 'label') return null;
    this.ensureBackward();
    return this.states[id].grad;
  }

  private setLeaf(id: number, values: Nested): void {
    const node = this.nodes[id];
    this.states[id].value = this.tensorFor(node.name, [...node.shape], values);
    this.pending = true;
    if (this.batchDepth === 0) this.recompute();
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
      tex: `${node.tex}^{\\top}`,
      elem: `${node.elem}^{\\top}`,
    });
    return state.transposed;
  }

  private resolveLoss(): Node | undefined {
    if (this.lossNode !== undefined) return this.lossNode;
    for (let i = this.nodes.length - 1; i >= 0; i--) {
      const node = this.nodes[i];
      if (node.kind === 'op' && this.states[i].value.size === 1) return node;
    }
    return undefined;
  }

  private recompute(): void {
    this.pending = false;
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
    this.version++;
    [...this.subscribers].forEach((fn) => fn());
  }

  private ensureBackward(): void {
    if (!this.backwardValid) this.computeBackward();
  }

  private computeBackward(): void {
    this.backwardValid = true;
    this.states.forEach((state, i) => {
      state.grad = Tensor.zeros(this.nodes[i].shape);
    });
    const loss = this.resolveLoss();
    if (loss === undefined) return;
    if (loss.shape.length > 0 && this.states[loss.id].value.size !== 1) {
      throw new Error(`loss node '${loss.name}' must have one element, got shape ${formatShape(loss.shape)}`);
    }
    this.states[loss.id].grad.data.fill(1);
    const upstream = new Set<Node>([loss]);
    for (let i = loss.id; i >= 0; i--) {
      const node = this.nodes[i];
      if (!upstream.has(node)) continue;
      for (const input of node.inputs) upstream.add(input);
      if (node.kind !== 'op') continue;
      const def = opDef(node.op as string) as AnyOpDef;
      const grads = def.backward(this.states[i].grad, node.inputs.map((n) => n.value), this.states[i].value, node.attrs);
      node.inputs.forEach((input, k) => {
        if (input.kind === 'label') return;
        if (def.differentiable !== undefined && !def.differentiable[k]) return;
        const grad = grads[k];
        if (grad === null || grad === undefined) return;
        if (formatShape(grad.shape) !== formatShape(input.shape)) {
          throw new Error(
            `${def.name}: backward of node '${node.name}' returned ${formatShape(grad.shape)} for input '${input.name}' with shape ${formatShape(input.shape)}`,
          );
        }
        const target = this.states[input.id].grad.data;
        for (let j = 0; j < target.length; j++) target[j] += grad.data[j];
      });
    }
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
