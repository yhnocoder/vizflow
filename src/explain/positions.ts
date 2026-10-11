import type { Graph, Node } from '../autograd';
import { indexOf, offsetOf } from '../tensor';
import { matmulOperands } from './elem';
import { edgesOf } from './edges';
import { explainDefOf } from './lookup';

export type Dep = [Node, number];

export interface Positions {
  readonly length: number;
  horizon(node: Node, offset: number): number | undefined;
  deps(node: Node, offset: number): readonly Dep[];
}

const positional = (n: Node, length: number) => (n.kind === 'input' || n.kind === 'label') && n.shape[0] === length;

function dependencies(op: Node, length: number): Dep[][] {
  const out: Dep[][] = Array.from({ length: op.value.size }, () => []);
  const all = op.inputs.map((_, arg) => arg);
  for (const e of edgesOf(op, explainDefOf(op), all)) {
    if ('cell' in e.from || 'cell' in e.to) {
      if (!('cell' in e.to) || e.arg !== 0) continue;
      const { a, b, out: at } = matmulOperands(e.to.cell.op, e.to.cell.index);
      const [A, B] = op.inputs;
      out[offsetOf(op.shape, at)].push([A, offsetOf(A.shape, a)], [B, offsetOf(B.shape, b)]);
      continue;
    }
    out[offsetOf(op.shape, e.to.index)].push([e.from.node, offsetOf(e.from.node.shape, e.from.index)]);
  }
  op.inputs.forEach((input) => {
    if (!positional(input, length) || input.shape.length >= op.shape.length) return;
    if (!input.shape.every((n, k) => n === op.shape[k])) return;
    out.forEach((list, offset) => list.push([input, offsetOf(input.shape, indexOf(op.shape, offset).slice(0, input.shape.length))]));
  });
  return out;
}

export function positionsOf(g: Graph, length: number): Positions {
  const data = new Map<Node, Int32Array>();
  const shown = new Map<Node, Int32Array>();
  const killed = new Map<Node, Uint8Array>();
  const deps = new Map<Node, Dep[][]>();
  const lower = new Map<Node, Int32Array>();
  const isKilled = (node: Node, at: number) => killed.get(node)?.[at] === 1;
  for (const node of g.nodes) {
    if (positional(node, length)) {
      const rows = Int32Array.from({ length: node.value.size }, (_, offset) => indexOf(node.shape, offset)[0]);
      data.set(node, rows);
      shown.set(node, rows);
      continue;
    }
    if (node.kind !== 'op') continue;
    const list = dependencies(node, length);
    deps.set(node, list);
    const size = node.value.size;
    const h = new Int32Array(size).fill(-1);
    const view = new Int32Array(size).fill(-1);
    const dead = new Uint8Array(size);
    const values = node.value.data;
    const pairs = node.op === 'matmul' || node.op === 'select_matmul';
    list.forEach((items, offset) => {
      const value = values[offset];
      dead[offset] = !Number.isFinite(value) || (value === 0 && items.some(([input, at]) => isKilled(input, at))) ? 1 : 0;
      for (let k = 0; k < items.length; k++) {
        const [input, at] = items[k];
        if (pairs) {
          const [other, otherAt] = items[k % 2 === 0 ? k + 1 : k - 1];
          if (isKilled(input, at) || isKilled(other, otherAt)) continue;
        }
        if (!isKilled(input, at)) h[offset] = Math.max(h[offset], data.get(input)?.[at] ?? -1);
        view[offset] = Math.max(view[offset], shown.get(input)?.[at] ?? -1);
      }
      if (dead[offset] === 0) view[offset] = h[offset];
    });
    data.set(node, h);
    shown.set(node, view);
    killed.set(node, dead);
    list.forEach((items, offset) => {
      for (const [input, at] of items) {
        if (input.kind !== 'buffer' || input.shape[0] !== length) continue;
        let floor = lower.get(input);
        if (floor === undefined) lower.set(input, (floor = new Int32Array(input.value.size).fill(length)));
        floor[at] = Math.min(floor[at], view[offset]);
      }
    });
  }
  for (const [buffer, floor] of lower) shown.set(buffer, floor.map((h, offset) => (h === length ? indexOf(buffer.shape, offset)[0] : h)));
  return {
    length,
    horizon: (node, offset) => {
      const h = shown.get(node)?.[offset];
      return h === undefined || h < 0 ? undefined : h;
    },
    deps: (node, offset) => (deps.get(node)?.[offset] ?? []).filter(([input, at]) => !isKilled(input, at)),
  };
}
