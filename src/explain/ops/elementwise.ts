import type { OpCtx, OpExplain } from '../../autograd';
import { transposedTex } from '../../autograd/node';
import { normalCdf, normalPdf } from '../../autograd/ops/elementwise';
import { broadcastIndex, formatNumber, normDim } from '../../tensor';

function operand(ctx: OpCtx, arg: number, out: number[]): number {
  const input = ctx.inputs[arg];
  if (input === undefined) return (ctx.attrs as { other: number }).other;
  return input.get(broadcastIndex(out, input.shape));
}

function scalarTex(attrs: object): string {
  const v = (attrs as { other: number }).other;
  return v < 0 ? `\\left(${formatNumber(v, { tex: true })}\\right)` : formatNumber(v, { tex: true });
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
      forward: (s) => forward(s.in[0], s.in[1] ?? scalarTex(s.attrs)),
      local: (s) => localTex(s.ins[0], s.ins[1] ?? scalarTex(s.attrs), s.arg),
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
export const rsqrtExplain = unary(
  (a) => `\\frac{1}{\\sqrt{${a}}}`,
  (_x, y) => -0.5 * y ** 3,
  (_x, y) => `-\\frac{1}{2}{${y}}^{3}`,
);
export const sqrtExplain = unary((a) => `\\sqrt{${a}}`, (_x, y) => 0.5 / y, (_x, y) => `\\frac{1}{2${y}}`);
export const sigmoidExplain = unary((a) => `\\sigma(${a})`, (_x, y) => y * (1 - y), (_x, y) => `${y}(1-${y})`);
export const siluExplain = unary(
  (a) => `\\operatorname{SiLU}(${a})`,
  (x) => {
    const s = 1 / (1 + Math.exp(-x));
    return s * (1 + x * (1 - s));
  },
  (x) => `\\sigma(${x})\\left(1+${x}\\left(1-\\sigma(${x})\\right)\\right)`,
);

export const reluExplain = unary(
  (a) => `\\operatorname{ReLU}(${a})`,
  (x) => (x > 0 ? 1 : 0),
  (x) => `\\mathbf{1}[${x}>0]`,
);
export const geluExplain = unary(
  (a) => `\\operatorname{GELU}(${a})`,
  (x) => normalCdf(x) + x * normalPdf(x),
  (x) => `\\Phi(${x})+${x}\\,\\varphi(${x})`,
);
export const identityExplain = unary((a) => a, () => 1, () => '1');

function rotatePosition(shape: readonly number[], attrs: object, out: readonly number[]): { first: boolean; partner: number } {
  const column = out[out.length - 1];
  const width = (attrs as { size?: number | null }).size ?? shape[shape.length - 1];
  const half = width / 2;
  const first = column % width < half;
  return { first, partner: first ? column + half : column - half };
}

export const rotateHalfExplain: OpExplain = {
  pattern: 'elementwise',
  local: (ctx, out) => (rotatePosition(ctx.out.shape, ctx.attrs, out).first ? -1 : 1),
  tex: {
    forward: (s) => `\\operatorname{rotate\\_half}(${s.in[0]})`,
    local: (s) => (rotatePosition(s.ctx.out.shape, s.attrs, s.outIndex).first ? '-1' : '1'),
  },
  selected: (ctx, out) => {
    const index = [...out];
    index[index.length - 1] = rotatePosition(ctx.out.shape, ctx.attrs, out).partner;
    return [{ arg: 0, index }];
  },
};

const exponentOf =(attrs: object) => (attrs as { exponent: number }).exponent;

export const powExplain: OpExplain = {
  pattern: 'elementwise',
  local: (ctx, _out, _arg, input) => exponentOf(ctx.attrs) * ctx.inputs[0].get(input) ** (exponentOf(ctx.attrs) - 1),
  tex: {
    forward: (s) => `{${s.in[0]}}^{${exponentOf(s.attrs)}}`,
    local: (s) => {
      const p = exponentOf(s.attrs);
      return p - 1 === 1 ? `${p}${s.in}` : `${p}{${s.in}}^{${p - 1}}`;
    },
  },
};

export const transposeExplain: OpExplain = {
  pattern: 'elementwise',
  local: () => 1,
  tex: {
    forward: (s) => transposedTex(s.in[0]),
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
