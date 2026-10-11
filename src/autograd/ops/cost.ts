import type { OpCost, OpFlops, Saved } from '../op';

export const numel = (shape: readonly number[]) => shape.reduce((a, b) => a * b, 1);

export const NO_FLOPS: number[] = [0];

const scaled = (factor: number, n: number) => (factor === 0 ? NO_FLOPS : factor === 1 ? [n] : [factor, n]);

export const VIEW_COST: OpCost = {
  view: true,
  saves: [[]],
  flops: (inputs) => ({ forward: NO_FLOPS, backward: inputs.map(() => NO_FLOPS) }),
};

export function pointwise(forward: number, backward: number, saves: Saved[][]): OpCost {
  return {
    saves,
    flops: (inputs, out) => ({ forward: scaled(forward, numel(out)), backward: inputs.map(() => scaled(backward, numel(out))) }),
  };
}

export function overInput(forward: number, backward: number, saves: Saved[][]): OpCost {
  return {
    saves,
    flops: (inputs) => ({ forward: scaled(forward, numel(inputs[0])), backward: inputs.map((_, k) => (k === 0 ? scaled(backward, numel(inputs[0])) : NO_FLOPS)) }),
  };
}

export function movement(scatter: boolean, saves: Saved[][]): OpCost {
  return {
    saves,
    flops: (inputs, out): OpFlops => ({ forward: NO_FLOPS, backward: inputs.map(() => (scatter ? [numel(out)] : NO_FLOPS)) }),
  };
}
