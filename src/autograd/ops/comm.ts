import { Tensor, formatShape } from '../../tensor';
import type { OpDef } from '../op';
import { type RoutedPair, routeOrder } from './index_ops';

export interface AllReduceAttrs {
  group: string;
}

export const allReduceOp: OpDef<AllReduceAttrs> = {
  name: 'all_reduce',
  args: ['input', 'group'],
  tensors: ['input'],
  shape: ([a]) => [...a],
  forward: ([a]) => a.clone(),
  backward: (gradOut) => [gradOut.clone()],
};

export interface DispatchAttrs {
  k: number;
  group: string;
  fp8: boolean;
}

export interface CombineAttrs {
  group: string;
}

function dispatchShape(x: readonly number[], sel: readonly number[], k: number): number[] {
  if (x.length !== 2 || sel.length !== 2 || sel[0] !== x[0]) throw new Error(`dispatch: x ${formatShape(x)} must be (T, d) and sel ${formatShape(sel)} must be (T, E)`);
  if (!Number.isInteger(k) || k < 1 || k > sel[1]) throw new Error(`dispatch: k ${k} is out of range for sel ${formatShape(sel)}`);
  return [x[0] * k, x[1]];
}

function combineTopk(y: readonly number[], sel: readonly number[], w: readonly number[]): number {
  const fail = () => new Error(`combine: y ${formatShape(y)} must be (T*k, d), sel ${formatShape(sel)} and w ${formatShape(w)} must both be (T, E)`);
  if (y.length !== 2 || sel.length !== 2 || w.length !== 2 || w[0] !== sel[0] || w[1] !== sel[1]) throw fail();
  const k = y[0] / sel[0];
  if (!Number.isInteger(k) || k < 1 || k > sel[1]) throw fail();
  return k;
}

export function routedPairs(sel: Tensor, rows: number, name: 'dispatch' | 'combine'): RoutedPair[] {
  return routeOrder(sel, name, rows / sel.shape[0]);
}

function gatherRows(x: Tensor, pairs: readonly RoutedPair[]): Tensor {
  const d = x.shape[1];
  const out = new Tensor([pairs.length, d]);
  pairs.forEach((pair, i) => out.data.set(x.data.subarray(pair.token * d, (pair.token + 1) * d), i * d));
  return out;
}

function scatterRows(y: Tensor, pairs: readonly RoutedPair[], T: number, weight: (pair: RoutedPair) => number): Tensor {
  const d = y.shape[1];
  const out = new Tensor([T, d]);
  pairs.forEach((pair, i) => {
    const scale = weight(pair);
    for (let c = 0; c < d; c++) out.data[pair.token * d + c] += scale * y.data[i * d + c];
  });
  return out;
}

export const dispatchOp: OpDef<DispatchAttrs> = {
  name: 'dispatch',
  args: ['x', 'sel', 'k', 'group', 'fp8'],
  tensors: ['x', 'sel'],
  sizes: ['k'],
  defaults: { fp8: false },
  differentiable: [true, false],
  shape: ([x, sel], attrs) => dispatchShape(x, sel, attrs.k),
  forward: ([x, sel], attrs) => {
    dispatchShape(x.shape, sel.shape, attrs.k);
    return gatherRows(x, routedPairs(sel, x.shape[0] * attrs.k, 'dispatch'));
  },
  backward: (gradOut, [x, sel]) => [scatterRows(gradOut, routedPairs(sel, gradOut.shape[0], 'dispatch'), x.shape[0], () => 1), null],
};

export const combineOp: OpDef<CombineAttrs> = {
  name: 'combine',
  args: ['y', 'sel', 'w', 'group'],
  tensors: ['y', 'sel', 'w'],
  differentiable: [true, false, true],
  shape: ([y, sel, w]) => {
    combineTopk(y, sel, w);
    return [sel[0], y[1]];
  },
  forward: ([y, sel, w]) => {
    combineTopk(y.shape, sel.shape, w.shape);
    const E = sel.shape[1];
    return scatterRows(y, routedPairs(sel, y.shape[0], 'combine'), sel.shape[0], (pair) => w.data[pair.token * E + pair.expert]);
  },
  backward: (gradOut, [y, sel, w]) => {
    const E = sel.shape[1];
    const d = y.shape[1];
    const pairs = routedPairs(sel, y.shape[0], 'combine');
    const gy = gatherRows(gradOut, pairs);
    const gw = new Tensor(w.shape);
    pairs.forEach((pair, i) => {
      const scale = w.data[pair.token * E + pair.expert];
      let dot = 0;
      for (let c = 0; c < d; c++) {
        gy.data[i * d + c] *= scale;
        dot += gradOut.data[pair.token * d + c] * y.data[i * d + c];
      }
      gw.data[pair.token * E + pair.expert] = dot;
    });
    return [gy, null, gw];
  },
};
