import type { OpCtx, OpExplain } from '../../autograd';
import { broadcastIndex, normDim } from '../../tensor';

function operand(ctx: OpCtx, arg: number, out: number[]): number {
  const input = ctx.inputs[arg];
  return input.get(broadcastIndex(out, input.shape));
}

function binary(
  forward: (a: string, b: string) => string,
  local: (a: number, b: number, arg: number) => number,
  localTex: (a: string, b: string, arg: number) => string,
): OpExplain {
  return {
    pattern: 'elementwise',
    local: (ctx, out, arg) => local(operand(ctx, 0, out), operand(ctx, 1, out), arg),
    tex: {
      forward: (s) => forward(s.in[0], s.in[1]),
      local: (s) => localTex(s.ins[0], s.ins[1], s.arg),
    },
  };
}

function unary(
  forward: (a: string) => string,
  local: (x: number, y: number) => number,
  localTex: (x: string, y: string) => string,
): OpExplain {
  return {
    pattern: 'elementwise',
    local: (ctx, out, _arg, input) => local(ctx.inputs[0].get(input), ctx.out.get(out)),
    tex: {
      forward: (s) => forward(s.in[0]),
      local: (s) => localTex(s.in, s.out),
    },
  };
}

export const addExplain = binary((a, b) => `${a}+${b}`, () => 1, () => '1');
export const subExplain = binary(
  (a, b) => `${a}-${b}`,
  (_a, _b, arg) => (arg === 0 ? 1 : -1),
  (_a, _b, arg) => (arg === 0 ? '1' : '-1'),
);
export const mulExplain = binary(
  (a, b) => `${a}\\cdot ${b}`,
  (a, b, arg) => (arg === 0 ? b : a),
  (a, b, arg) => (arg === 0 ? b : a),
);
export const divExplain = binary(
  (a, b) => `${a}/${b}`,
  (a, b, arg) => (arg === 0 ? 1 / b : -a / (b * b)),
  (a, b, arg) => (arg === 0 ? `\\frac{1}{${b}}` : `-\\frac{${a}}{${b}^{2}}`),
);
export const negExplain = unary((a) => `-${a}`, () => -1, () => '-1');
export const expExplain = unary((a) => `\\exp(${a})`, (_x, y) => y, (x) => `e^{${x}}`);
export const logExplain = unary((a) => `\\log ${a}`, (x) => 1 / x, (x) => `\\frac{1}{${x}}`);

export const transposeExplain: OpExplain = {
  pattern: 'elementwise',
  local: () => 1,
  tex: {
    forward: (s) => `${s.in[0]}^{\\top}`,
    local: () => '1',
  },
  selected: (ctx, out) => {
    const attrs = ctx.attrs as { dim0: number; dim1: number };
    const d0 = normDim(attrs.dim0, out.length);
    const d1 = normDim(attrs.dim1, out.length);
    const index = [...out];
    index[d0] = out[d1];
    index[d1] = out[d0];
    return [{ arg: 0, index }];
  },
};
