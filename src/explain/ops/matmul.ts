import type { OpCtx, OpExplain } from '../../autograd';
import { groupOf } from '../../autograd/ops/matmul';

const headOf = (attrs: object) => (attrs as { index: number }).index;

export const matmulExplain: OpExplain = {
  pattern: 'contract',
  local: (ctx, out, arg, input) => {
    const [A, B] = ctx.inputs;
    if (arg === 0) {
      const k = input[input.length - 1];
      return B.ndim === 1 ? B.get([k]) : B.get([k, out[out.length - 1]]);
    }
    const k = input[0];
    return A.ndim === 1 ? A.get([k]) : A.get([out[0], k]);
  },
  tex: {
    forward: (s) => `${s.in[0]}${s.in[1]}`,
    local: (s) => (s.arg === 0 ? s.ins[1] : s.ins[0]),
  },
};

function rowsByGroup(ctx: OpCtx): Map<number, number[]> {
  const [, weight, group] = ctx.inputs;
  const rows = new Map<number, number[]>();
  for (let r = 0; r < group.size; r++) {
    const g = groupOf(weight, group, r);
    rows.set(g, [...(rows.get(g) ?? []), r + 1]);
  }
  return rows;
}

export const groupedLinearExplain: OpExplain = {
  pattern: 'select',
  local: (ctx, out, arg, input) => {
    const [x, weight, group] = ctx.inputs;
    if (arg === 0) return weight.get([groupOf(weight, group, out[0]), out[1], input[1]]);
    return x.get([out[0], input[2]]);
  },
  tex: {
    forward: (s) => `\\operatorname{grouped\\_linear}(${s.in[0]},${s.in[1]},${s.in[2]})`,
    local: (s) => (s.arg === 0 ? s.elem(1, [groupOf(s.ctx.inputs[1], s.ctx.inputs[2], s.outIndex[0]), s.outIndex[1], s.inIndex[1]]) : s.elem(0, [s.outIndex[0], s.inIndex[2]])),
  },
  selected: (ctx, out) => {
    const [x, weight, group] = ctx.inputs;
    const g = groupOf(weight, group, out[0]);
    return Array.from({ length: x.shape[1] }, (_, k) => [
      { arg: 0, index: [out[0], k] },
      { arg: 1, index: [g, out[1], k] },
    ]).flat();
  },
  note: (ctx, tex) => {
    const parts = [...rowsByGroup(ctx)].sort(([a], [b]) => a - b).map(([g, rows]) => `第 ${rows.join('、')} 行用 $${tex.in[1]}[${g}]$`);
    return `每一行乘 $${tex.in[2]}$ 给出的那一块矩阵的转置：${parts.join('，')}。块号和 PyTorch 一样从 0 开始，没有出现的块这一次没有用到。`;
  },
};

export const selectMatmulExplain: OpExplain = {
  pattern: 'contract',
  local: (ctx, out, arg, input) => {
    const [A, B] = ctx.inputs;
    if (arg === 0) return B.get([input[2], out[1]]);
    return A.get([out[0], headOf(ctx.attrs), input[0]]);
  },
  tex: {
    forward: (s) => `\\operatorname{select}(${s.in[0]},1,${headOf(s.attrs)})\\,${s.in[1]}`,
    local: (s) => (s.arg === 0 ? s.ins[1] : s.ins[0]),
  },
};
