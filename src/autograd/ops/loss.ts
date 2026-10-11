import { Tensor, formatShape } from '../../tensor';
import type { OpDef } from '../op';
import { NO_FLOPS, numel } from './cost';

export type Reduction = 'mean' | 'sum' | 'none';

export interface NllLossAttrs {
  reduction: Reduction;
  ignore_index: number;
}

const REDUCTIONS: readonly Reduction[] = ['mean', 'sum', 'none'];

function layout(input: readonly number[], target: readonly number[], attrs: NllLossAttrs): { rows: number; classes: number } {
  if (!REDUCTIONS.includes(attrs.reduction)) {
    throw new Error(`nll_loss: reduction must be one of 'mean', 'sum', 'none', got ${JSON.stringify(attrs.reduction)}`);
  }
  const fail = () => new Error(`nll_loss: input ${formatShape(input)} does not match target ${formatShape(target)}`);
  if (input.length === 1 && target.length === 0) return { rows: 1, classes: input[0] };
  if (input.length === 2 && target.length === 1 && input[0] === target[0]) return { rows: input[0], classes: input[1] };
  throw fail();
}

function picks(input: Tensor, target: Tensor, attrs: NllLossAttrs): Int32Array {
  const { rows, classes } = layout(input.shape, target.shape, attrs);
  const offsets = new Int32Array(rows);
  for (let n = 0; n < rows; n++) {
    const t = Math.round(target.data[n]);
    if (t === attrs.ignore_index) {
      offsets[n] = -1;
      continue;
    }
    if (t < 0 || t >= classes) {
      throw new Error(`nll_loss: target ${t} at row ${n} is out of range for ${classes} classes`);
    }
    offsets[n] = n * classes + t;
  }
  return offsets;
}

function counted(offsets: Int32Array): number {
  let count = 0;
  for (const o of offsets) if (o >= 0) count++;
  return count;
}

export const nllLossOp: OpDef<NllLossAttrs> = {
  name: 'nll_loss',
  args: ['input', 'target'],
  tensors: ['input', 'target'],
  defaults: { reduction: 'mean', ignore_index: -100 },
  differentiable: [true, false],
  shape: ([input, target], attrs) => {
    layout(input, target, attrs);
    return attrs.reduction === 'none' ? [...target] : [];
  },
  forward: ([input, target], attrs) => {
    const offsets = picks(input, target, attrs);
    const perRow = Float64Array.from(offsets, (o) => (o < 0 ? 0 : -input.data[o]));
    if (attrs.reduction === 'none') return new Tensor(target.shape, perRow);
    let total = 0;
    for (const v of perRow) total += v;
    return Tensor.from(attrs.reduction === 'sum' ? total : total / counted(offsets));
  },
  backward: (gradOut, [input, target], _out, attrs) => {
    const offsets = picks(input, target, attrs);
    const grad = new Tensor(input.shape);
    const scale = attrs.reduction === 'mean' ? 1 / counted(offsets) : 1;
    offsets.forEach((o, n) => {
      if (o < 0) return;
      const upstream = attrs.reduction === 'none' ? gradOut.data[n] : gradOut.data[0];
      grad.data[o] = -upstream * scale;
    });
    return [grad, null];
  },
  cost: { saves: [[1], []], flops: ([, target]) => ({ forward: [numel(target)], backward: [[numel(target)], NO_FLOPS] }) },
};
