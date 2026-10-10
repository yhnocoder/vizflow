import type { OpExplain } from '../../autograd';
import { normDim } from '../../tensor';

export const gatherExplain: OpExplain = {
  pattern: 'select',
  local: () => 1,
  tex: {
    forward: (s) => `\\operatorname{gather}(${s.in[0]},${s.in[1]})`,
    local: () => '1',
  },
  selected: (ctx, out) => {
    const dim = normDim((ctx.attrs as { dim: number }).dim, out.length);
    const index = [...out];
    index[dim] = Math.round(ctx.inputs[1].get(out));
    return [{ arg: 0, index }];
  },
};
