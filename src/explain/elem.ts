import type { Node } from '../autograd';

export type ElemRef = { node: Node; index: number[] };
export type CellRef = { cell: { op: Node; index: [number, number, number] } };

export function isCell(ref: ElemRef | CellRef): ref is CellRef {
  return 'cell' in ref;
}

export function trailingSubscript(elem: string): { base: string; inner: string } | undefined {
  const single = /(?<!\\)_([^{}\\\s])$/.exec(elem);
  if (single !== null) return { base: elem.slice(0, single.index), inner: single[1] };
  if (!elem.endsWith('}')) return undefined;
  let depth = 0;
  for (let i = elem.length - 1; i >= 0; i--) {
    if (elem[i] === '}') depth++;
    else if (elem[i] === '{') depth--;
    if (depth > 0) continue;
    if (i === 0 || elem[i - 1] !== '_' || (i >= 2 && elem[i - 2] === '\\')) return undefined;
    return { base: elem.slice(0, i - 1), inner: elem.slice(i + 1, -1) };
  }
  return undefined;
}

export function elemTexOf(node: Node, index: readonly number[]): string {
  const shown = index.filter((_, d) => node.shape[d] !== 1).map((i) => i + 1);
  if (shown.length === 0) return node.elem;
  const merged = trailingSubscript(node.elem);
  if (merged === undefined) return `${node.elem}_{${shown.join(',')}}`;
  return `${merged.base}_{${merged.inner},${shown.join(',')}}`;
}

export function matmulOperands(op: Node, index: readonly [number, number, number]): { a: number[]; b: number[]; out: number[] } {
  const [r, k, i] = index;
  const [A, B] = op.inputs;
  const head = (op.attrs as { index?: number }).index;
  return {
    a: A.shape.length === 1 ? [k] : head === undefined ? [r, k] : [r, head, k],
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
