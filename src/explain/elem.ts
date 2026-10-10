import type { Node } from '../autograd';

export type ElemRef = { node: Node; index: number[] };
export type CellRef = { cell: { op: Node; index: [number, number, number] } };

export function isCell(ref: ElemRef | CellRef): ref is CellRef {
  return 'cell' in ref;
}

export function subscript(shape: readonly number[], index: readonly number[]): string {
  const shown = index.filter((_, d) => shape[d] !== 1);
  return shown.length === 0 ? '' : `_{${shown.map((i) => i + 1).join(',')}}`;
}

export function elemTexOf(node: Node, index: readonly number[]): string {
  return `${node.elem}${subscript(node.shape, index)}`;
}

export function matmulOperands(op: Node, index: readonly [number, number, number]): { a: number[]; b: number[]; out: number[] } {
  const [r, k, i] = index;
  const [A, B] = op.inputs;
  return {
    a: A.shape.length === 1 ? [k] : [r, k],
    b: B.shape.length === 1 ? [k] : [k, i],
    out: A.shape.length === 1 ? [i] : B.shape.length === 1 ? [r] : [r, i],
  };
}

export function cellTex(cell: CellRef['cell']): string {
  const { a, b } = matmulOperands(cell.op, cell.index);
  const [A, B] = cell.op.inputs;
  return `${elemTexOf(A, a)}${elemTexOf(B, b)}`;
}

export function elemTex(ref: ElemRef | CellRef): string {
  return isCell(ref) ? cellTex(ref.cell) : elemTexOf(ref.node, ref.index);
}

export function refKey(ref: ElemRef | CellRef): string {
  return isCell(ref) ? `cell:${ref.cell.op.id}:${ref.cell.index.join(',')}` : `${ref.node.id}:${ref.index.join(',')}`;
}
