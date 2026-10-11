import type { OpDef } from '../op';

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
