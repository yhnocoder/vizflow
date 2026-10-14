import { type Node, type OpCtx, type OpExplain, inputDifferentiable, opDef } from '../autograd';
import { broadcastIndex, forEachIndex, indexOf, normDim, normDims, reduceMap } from '../tensor';
import { isBatched } from '../autograd/ops/matmul';
import { type CellRef, type ElemRef, matmulOperands } from './elem';

export interface FlowEdge {
  id: string;
  op: Node;
  arg: number;
  from: ElemRef | CellRef;
  to: ElemRef | CellRef;
  constant?: true;
}

export function contextOf(node: Node): OpCtx {
  return { node, inputs: node.inputs.map((n) => n.value), out: node.value, attrs: node.attrs };
}

function flowArgs(node: Node): number[] {
  const def = opDef(node.op as string);
  return node.inputs.map((_, arg) => arg).filter((arg) => def === undefined || inputDifferentiable(def, arg));
}

function edge(op: Node, arg: number, suffix: string, from: ElemRef | CellRef, to: ElemRef | CellRef): FlowEdge {
  const made: FlowEdge = { id: `${op.name}:${arg}:${suffix}`, op, arg, from, to };
  const input = op.inputs[arg];
  if (input !== undefined && (input.kind === 'buffer' || input.grad === null)) made.constant = true;
  return made;
}

function oneToOne(op: Node, args: number[], inputIndex: (arg: number, out: number[]) => number[]): FlowEdge[] {
  const edges: FlowEdge[] = [];
  forEachIndex(op.shape, (out) => {
    for (const arg of args) {
      edges.push(edge(op, arg, out.join(','), { node: op.inputs[arg], index: inputIndex(arg, out) }, { node: op, index: out }));
    }
  });
  return edges;
}

function manyToOne(op: Node, args: number[], groupOf: (shape: readonly number[]) => Int32Array, outIndexOf: (group: number) => number[]): FlowEdge[] {
  const edges: FlowEdge[] = [];
  for (const arg of args) {
    const input = op.inputs[arg];
    const map = groupOf(input.shape);
    for (let offset = 0; offset < map.length; offset++) {
      const inIndex = indexOf(input.shape, offset);
      const out = outIndexOf(map[offset]);
      edges.push(edge(op, arg, `${inIndex.join(',')}->${out.join(',')}`, { node: input, index: inIndex }, { node: op, index: out }));
    }
  }
  return edges;
}

function reduceEdges(op: Node, args: number[]): FlowEdge[] {
  const attrs = op.attrs as { dim: number | number[] | null; keepdim: boolean };
  return manyToOne(
    op,
    args,
    (shape) => reduceMap(shape, normDims(attrs.dim, shape.length), attrs.keepdim).map,
    (group) => indexOf(op.shape, group),
  );
}

function denseEdges(op: Node, args: number[]): FlowEdge[] {
  const dim = normDim((op.attrs as { dim: number }).dim, op.shape.length);
  const edges: FlowEdge[] = [];
  for (const arg of args) {
    const input = op.inputs[arg];
    const { map } = reduceMap(input.shape, [dim], false);
    forEachIndex(op.shape, (out, outOffset) => {
      const group = map[outOffset];
      for (let offset = 0; offset < map.length; offset++) {
        if (map[offset] !== group) continue;
        const inIndex = indexOf(input.shape, offset);
        edges.push(edge(op, arg, `${inIndex.join(',')}->${out.join(',')}`, { node: input, index: inIndex }, { node: op, index: out }));
      }
    });
  }
  return edges;
}

function batchedContractEdges(op: Node, args: number[]): FlowEdge[] {
  const [A, B] = op.inputs;
  const inner = B.shape[B.shape.length - 2];
  const edges: FlowEdge[] = [];
  forEachIndex(op.shape, (out) => {
    const batch = out.slice(0, -2);
    const row = out[out.length - 2];
    const col = out[out.length - 1];
    for (let k = 0; k < inner; k++) {
      const suffix = `${out.join(',')}:${k}`;
      if (args.includes(0)) edges.push(edge(op, 0, suffix, { node: A, index: [...broadcastIndex(batch, A.shape.slice(0, -2)), row, k] }, { node: op, index: out }));
      if (args.includes(1)) edges.push(edge(op, 1, suffix, { node: B, index: [...broadcastIndex(batch, B.shape.slice(0, -2)), k, col] }, { node: op, index: out }));
    }
  });
  return edges;
}

export function batchedCells(op: Node): boolean {
  const [A, B] = op.inputs;
  return op.op === 'matmul' && A.shape.length === 3 && B.shape.length === 3 && op.shape.length === 3;
}

function contractEdges(op: Node, args: number[]): FlowEdge[] {
  const [A, B] = op.inputs;
  const cells = batchedCells(op);
  if (op.op === 'matmul' && isBatched(A.shape, B.shape) && !cells) return batchedContractEdges(op, args);
  const batch = cells ? op.shape.slice(0, -2) : [];
  const rows = A.shape.length === 1 ? 1 : cells ? A.shape[1] : A.shape[0];
  const inner = B.shape.length === 1 ? B.shape[0] : B.shape[B.shape.length - 2];
  const cols = B.shape.length === 1 ? 1 : B.shape[B.shape.length - 1];
  const edges: FlowEdge[] = [];
  forEachIndex([...batch, rows, inner, cols], (index) => {
    const cell: CellRef = { cell: { op, index } };
    const { a, b, out } = matmulOperands(op, index);
    const suffix = index.join(',');
    edges.push(edge(op, 0, suffix, { node: A, index: a }, cell));
    edges.push(edge(op, 1, suffix, { node: B, index: b }, cell));
    edges.push(edge(op, -1, suffix, cell, { node: op, index: out }));
  });
  return edges;
}

function selectEdges(op: Node, args: number[], def: OpExplain, ctx: OpCtx): FlowEdge[] {
  if (def.selected === undefined) throw new Error(`explain: op '${op.op}' uses the select pattern but has no 'selected'`);
  const edges: FlowEdge[] = [];
  forEachIndex(op.shape, (out) => {
    def.selected!(ctx, out).forEach((pick, ordinal) => {
      if (!args.includes(pick.arg)) return;
      const suffix = `${out.join(',')}#${ordinal}`;
      edges.push(edge(op, pick.arg, suffix, { node: op.inputs[pick.arg], index: pick.index }, { node: op, index: out }));
    });
  });
  return edges;
}

export function edgesOf(op: Node, def: OpExplain): FlowEdge[] {
  const args = flowArgs(op);
  const ctx = contextOf(op);
  switch (def.pattern) {
    case 'elementwise':
      if (def.selected !== undefined) return selectEdges(op, args, def, ctx);
      return oneToOne(op, args, (arg, out) => broadcastIndex(out, op.inputs[arg].shape));
    case 'reduce':
      return reduceEdges(op, args);
    case 'dense':
      return denseEdges(op, args);
    case 'contract':
      return contractEdges(op, args);
    case 'select':
      return selectEdges(op, args, def, ctx);
  }
}

export function cellValue(cell: CellRef['cell']): number {
  const { a, b } = matmulOperands(cell.op, cell.index);
  const [A, B] = cell.op.inputs;
  return A.value.get(a) * B.value.get(b);
}
