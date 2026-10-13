import { Tensor, broadcastIndex, broadcastShapes, forEachIndex, formatShape } from '../tensor';
import { opDef } from './registry';

export type ValueOp = '+' | '-' | '*' | '/' | '//' | '**' | '@';

export type ValueIndex = { k: 'at'; i: number } | { k: 'slice'; start?: number; stop?: number } | { k: 'new' } | { k: 'ellipsis' };

export type ValueExpr =
  | { k: 'num'; v: number }
  | { k: 'bool'; v: boolean }
  | { k: 'node'; name: string; grad?: boolean }
  | { k: 'list'; items: ValueExpr[] }
  | { k: 'neg'; arg: ValueExpr }
  | { k: 'bin'; op: ValueOp; a: ValueExpr; b: ValueExpr }
  | { k: 'call'; fn: string; args: ValueExpr[]; kwargs: [string, ValueExpr][] }
  | { k: 'index'; obj: ValueExpr; items: ValueIndex[] };

export type ReadValue = (name: string, grad: boolean) => Tensor;

type Evaluated = Tensor | boolean;

const ELEMENTWISE: Record<Exclude<ValueOp, '@'>, (x: number, y: number) => number> = {
  '+': (x, y) => x + y,
  '-': (x, y) => x - y,
  '*': (x, y) => x * y,
  '/': (x, y) => x / y,
  '//': (x, y) => Math.floor(x / y),
  '**': (x, y) => x ** y,
};

const scalar = (v: number) => new Tensor([], [v]);

function tensorOf(value: Evaluated, what: string): Tensor {
  if (typeof value === 'boolean') throw new Error(`${what} must be a number or a tensor, got ${value}`);
  return value;
}

function numberOf(value: Evaluated, what: string): number {
  if (typeof value === 'boolean' || value.shape.length !== 0) throw new Error(`${what} must be a number`);
  return value.data[0];
}

function zip(a: Tensor, b: Tensor, fn: (x: number, y: number) => number): Tensor {
  const shape = broadcastShapes(a.shape, b.shape);
  const out = new Tensor(shape);
  let k = 0;
  forEachIndex(shape, (index) => {
    out.data[k++] = fn(a.get(broadcastIndex(index, a.shape)), b.get(broadcastIndex(index, b.shape)));
  });
  return out;
}

function stack(items: Evaluated[]): Tensor {
  const tensors = items.map((item) => tensorOf(item, 'a list item'));
  if (tensors.length === 0) throw new Error('a list needs at least one item');
  const inner = tensors[0].shape;
  if (tensors.some((t) => formatShape(t.shape) !== formatShape(inner))) throw new Error('list items must have the same shape');
  const data = new Float64Array(tensors.length * tensors[0].size);
  tensors.forEach((t, i) => data.set(t.data, i * t.size));
  return new Tensor([tensors.length, ...inner], data);
}

function arange(args: number[]): Tensor {
  if (args.length < 1 || args.length > 3) throw new Error('arange takes one to three numbers');
  const [start, stop, step] = args.length === 1 ? [0, args[0], 1] : [args[0], args[1], args[2] ?? 1];
  const length = Math.max(0, Math.ceil((stop - start) / step));
  return new Tensor([length], Array.from({ length }, (_, i) => start + i * step));
}

function index(base: Tensor, items: ValueIndex[]): Tensor {
  const consumed = items.filter((item) => item.k === 'at' || item.k === 'slice').length;
  if (consumed > base.shape.length) throw new Error(`${consumed} indices for a tensor with shape ${formatShape(base.shape)}`);
  if (items.filter((item) => item.k === 'ellipsis').length > 1) throw new Error("an index can use '...' once");
  const rest = base.shape.length - consumed;
  const ellipsis = items.findIndex((item) => item.k === 'ellipsis');
  const full: ValueIndex[] = ellipsis < 0 ? [...items, ...Array.from({ length: rest }, (): ValueIndex => ({ k: 'slice' }))] : [...items.slice(0, ellipsis), ...Array.from({ length: rest }, (): ValueIndex => ({ k: 'slice' })), ...items.slice(ellipsis + 1)];
  const shape: number[] = [];
  const plan: ({ dim: number; at: number } | { dim: number; start: number; out: number } | { out: number })[] = [];
  let dim = 0;
  for (const item of full) {
    if (item.k === 'new') {
      plan.push({ out: shape.length });
      shape.push(1);
      continue;
    }
    if (item.k === 'ellipsis') continue;
    const size = base.shape[dim];
    if (item.k === 'at') {
      const at = item.i < 0 ? size + item.i : item.i;
      if (at < 0 || at >= size) throw new Error(`index ${item.i} is out of range for dimension ${dim} of size ${size}`);
      plan.push({ dim, at });
    } else {
      const bound = (v: number | undefined, fallback: number) => (v === undefined ? fallback : Math.min(size, Math.max(0, v < 0 ? size + v : v)));
      const start = bound(item.start, 0);
      plan.push({ dim, start, out: shape.length });
      shape.push(Math.max(0, bound(item.stop, size) - start));
    }
    dim++;
  }
  const out = new Tensor(shape);
  let k = 0;
  forEachIndex(shape, (outIndex) => {
    const inIndex = new Array<number>(base.shape.length);
    for (const step of plan) {
      if ('at' in step) inIndex[step.dim] = step.at;
      else if ('dim' in step) inIndex[step.dim] = step.start + outIndex[step.out];
    }
    out.data[k++] = base.get(inIndex);
  });
  return out;
}

function callOp(fn: string, args: Evaluated[], kwargs: [string, Evaluated][]): Tensor {
  if (fn === 'arange') {
    if (kwargs.length > 0) throw new Error('arange takes no keyword arguments');
    return arange(args.map((arg) => numberOf(arg, 'an arange argument')));
  }
  const def = opDef(fn);
  if (def === undefined) throw new Error(`unknown function '${fn}'`);
  if (args.length > def.args.length) throw new Error(`${fn} takes at most ${def.args.length} positional arguments, got ${args.length}`);
  const attrs: Record<string, unknown> = { ...(def.defaults ?? {}) };
  const inputs: Tensor[] = [];
  const plain = (value: Evaluated) => (typeof value === 'boolean' || value.shape.length !== 0 ? value : value.data[0]);
  for (const [key, value] of kwargs) {
    if (def.tensors.includes(key) || !(def.args.includes(key) || key in attrs)) throw new Error(`${fn} has no keyword argument '${key}'`);
    attrs[key] = plain(value);
  }
  def.args.forEach((name, i) => {
    const given = args[i];
    if (def.tensors.includes(name)) {
      if (given === undefined) throw new Error(`${fn}: missing argument '${name}'`);
      if (def.scalars?.includes(name) && typeof given !== 'boolean' && given.shape.length === 0) attrs[name] = given.data[0];
      else inputs.push(tensorOf(given, `argument '${name}' of ${fn}`));
    } else if (given !== undefined) attrs[name] = plain(given);
    else if (!(name in attrs)) throw new Error(`${fn}: missing argument '${name}'`);
  });
  def.shape(inputs.map((t) => [...t.shape]), attrs);
  return def.forward(inputs, attrs);
}

function run(expr: ValueExpr, read: ReadValue): Evaluated {
  switch (expr.k) {
    case 'num':
      return scalar(expr.v);
    case 'bool':
      return expr.v;
    case 'node':
      return read(expr.name, expr.grad ?? false);
    case 'list':
      return stack(expr.items.map((item) => run(item, read)));
    case 'neg':
      return zip(scalar(0), tensorOf(run(expr.arg, read), 'the operand of -'), (x, y) => x - y);
    case 'bin': {
      const a = tensorOf(run(expr.a, read), `the left side of '${expr.op}'`);
      const b = tensorOf(run(expr.b, read), `the right side of '${expr.op}'`);
      return expr.op === '@' ? callOp('matmul', [a, b], []) : zip(a, b, ELEMENTWISE[expr.op]);
    }
    case 'call':
      return callOp(
        expr.fn,
        expr.args.map((arg) => run(arg, read)),
        expr.kwargs.map(([key, arg]) => [key, run(arg, read)]),
      );
    case 'index':
      return index(tensorOf(run(expr.obj, read), 'an indexed value'), expr.items);
  }
}

export function evaluateValue(expr: ValueExpr, read: ReadValue): Tensor {
  return tensorOf(run(expr, read), 'the expression');
}
