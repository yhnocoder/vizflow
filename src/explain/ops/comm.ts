import { type OpExplain, routedPairs } from '../../autograd';

export const allReduceExplain: OpExplain = {
  pattern: 'elementwise',
  local: () => 1,
  tex: {
    forward: (s) => `\\operatorname{all\\_reduce}(${s.in[0]})`,
    local: () => '1',
  },
};

export const dispatchExplain: OpExplain = {
  pattern: 'elementwise',
  local: () => 1,
  tex: {
    forward: (s) => `\\operatorname{dispatch}(${s.in[0]},${s.in[1]})`,
    local: () => '1',
  },
  selected: (ctx, out) => {
    const pair = routedPairs(ctx.inputs[1], ctx.out.shape[0], 'dispatch')[out[0]];
    return [{ arg: 0, index: [pair.token, out[1]] }];
  },
};

export const combineExplain: OpExplain = {
  pattern: 'elementwise',
  local: (ctx, out, arg, input) => {
    const [y, sel, w] = ctx.inputs;
    const E = sel.shape[1];
    const d = y.shape[1];
    if (arg === 0) {
      const pair = routedPairs(sel, y.shape[0], 'combine')[input[0]];
      return w.data[pair.token * E + pair.expert];
    }
    const row = routedPairs(sel, y.shape[0], 'combine').findIndex((pair) => pair.token === input[0] && pair.expert === input[1]);
    return y.data[row * d + out[1]];
  },
  tex: {
    forward: (s) => `\\operatorname{combine}(${s.in[0]},${s.in[2]})`,
    local: () => 'w',
  },
  selected: (ctx, out) => {
    const [y, sel] = ctx.inputs;
    const picks: { arg: number; index: number[] }[] = [];
    routedPairs(sel, y.shape[0], 'combine').forEach((pair, i) => {
      if (pair.token !== out[0]) return;
      picks.push({ arg: 0, index: [i, out[1]] }, { arg: 2, index: [pair.token, pair.expert] });
    });
    return picks;
  },
};
