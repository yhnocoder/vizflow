import type { OpCtx, OpExplain } from '../../autograd';
import { transposedTex } from '../../autograd/node';
import { type ClampAttrs, type PowAttrs, SOFTPLUS_THRESHOLD, type TriuAttrs, clampInside, copysignLocal, normalCdf, normalPdf, powOperands, sigmoid, trilKeeps, triuKeeps } from '../../autograd/ops/elementwise';
import { type Tensor, broadcastIndex, formatNumber, normDim } from '../../tensor';

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
export const rsubExplain = binary(
  (a, b) => `${b}-${a}`,
  (_a, _b, arg) => (arg === 0 ? -1 : 1),
  (_a, _b, arg) => (arg === 0 ? '-1' : '1'),
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
export const cosExplain = unary((a) => `\\cos ${a}`, (x) => -Math.sin(x), (x) => `-\\sin ${x}`);
export const sinExplain = unary((a) => `\\sin ${a}`, (x) => Math.cos(x), (x) => `\\cos ${x}`);
export const logExplain = unary((a) => `\\log ${a}`, (x) => 1 / x, (x) => `\\frac{1}{${x}}`);
const rsqrtBase = unary(
  (a) => `\\frac{1}{\\sqrt{${a}}}`,
  (_x, y) => -0.5 * y ** 3,
  (_x, y) => `-\\frac{1}{2}{${y}}^{3}`,
);
export const rsqrtExplain: OpExplain = {
  ...rsqrtBase,
  tex: {
    ...rsqrtBase.tex,
    forward: (s) => ((s.attrs as { eps?: number }).eps ? `\\frac{1}{\\sqrt{${s.in[0]}+\\epsilon}}` : rsqrtBase.tex.forward(s)),
  },
};
export const sqrtExplain = unary((a) => `\\sqrt{${a}}`, (_x, y) => 0.5 / y, (_x, y) => `\\frac{1}{2${y}}`);
export const reciprocalExplain = unary((a) => `\\frac{1}{${a}}`, (_x, y) => -y * y, (x) => `-\\frac{1}{{${x}}^{2}}`);
export const erfExplain = unary(
  (a) => `\\operatorname{erf}(${a})`,
  (x) => (2 / Math.sqrt(Math.PI)) * Math.exp(-x * x),
  (x) => `\\frac{2}{\\sqrt{\\pi}}e^{-{${x}}^{2}}`,
);
export const sigmoidExplain = unary((a) => `\\sigma(${a})`, (_x, y) => y * (1 - y), (_x, y) => `${y}(1-${y})`);
export const siluExplain = unary(
  (a) => `\\operatorname{SiLU}(${a})`,
  (x) => {
    const s = 1 / (1 + Math.exp(-x));
    return s * (1 + x * (1 - s));
  },
  (x) => `\\sigma(${x})\\left(1+${x}\\left(1-\\sigma(${x})\\right)\\right)`,
);
export const softplusExplain = unary(
  (a) => `\\operatorname{softplus}(${a})`,
  (x) => (x > SOFTPLUS_THRESHOLD ? 1 : sigmoid(x)),
  (x) => `\\sigma(${x})`,
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

const lastHalf = (shape: readonly number[]) => shape[shape.length - 1] / 2;

export const rotateHalfExplain: OpExplain = {
  pattern: 'elementwise',
  local: (ctx, out) => (out[out.length - 1] < lastHalf(ctx.out.shape) ? -1 : 1),
  tex: {
    forward: (s) => `\\operatorname{rotate\\_half}(${s.in[0]})`,
    local: (s) => (s.outIndex[s.outIndex.length - 1] < lastHalf(s.ctx.out.shape) ? '-1' : '1'),
  },
  selected: (ctx, out) => {
    const last = out.length - 1;
    const half = lastHalf(ctx.out.shape);
    const index = [...out];
    index[last] = out[last] < half ? out[last] + half : out[last] - half;
    return [{ arg: 0, index }];
  },
};

function clampTex(input: string, { min, max }: ClampAttrs): string {
  const bound = (v: number) => (v < 0 ? `\\left(${formatNumber(v, { tex: true })}\\right)` : formatNumber(v, { tex: true }));
  if (min === null) return `\\min(${input},${bound(max as number)})`;
  if (max === null) return `\\max(${input},${bound(min)})`;
  return `\\operatorname{clamp}(${input},${bound(min)},${bound(max)})`;
}

export const clampExplain: OpExplain = {
  pattern: 'elementwise',
  local: (ctx, _out, _arg, input) => (clampInside(ctx.inputs[0].get(input), ctx.attrs as ClampAttrs) ? 1 : 0),
  tex: {
    forward: (s) => clampTex(s.in[0], s.attrs as ClampAttrs),
    local: (s) => (clampInside(s.ctx.inputs[0].get(s.inIndex), s.attrs as ClampAttrs) ? '1' : '0'),
  },
};

function powRole(attrs: PowAttrs, arg: number): 'base' | 'exponent' {
  return attrs.input === undefined && arg === 0 ? 'base' : 'exponent';
}

function powAt(ctx: OpCtx, out: number[]): { base: number; exponent: number } {
  const { base, exponent } = powOperands(ctx.inputs, ctx.attrs as PowAttrs);
  const at = (value: Tensor | number) => (typeof value === 'number' ? value : value.get(broadcastIndex(out, value.shape)));
  return { base: at(base), exponent: at(exponent) };
}

function powTex(attrs: PowAttrs, ins: readonly string[]): { base: string; exponent: string } {
  const { base, exponent } = powOperands(ins, attrs);
  return { base: typeof base === 'number' ? (base < 0 ? `\\left(${base}\\right)` : `${base}`) : base, exponent: `${exponent}` };
}

export const powExplain: OpExplain = {
  pattern: 'elementwise',
  local: (ctx, out, arg) => {
    const { base, exponent } = powAt(ctx, out);
    return powRole(ctx.attrs as PowAttrs, arg) === 'base' ? exponent * base ** (exponent - 1) : base ** exponent * Math.log(base);
  },
  tex: {
    forward: (s) => {
      const attrs = s.attrs as PowAttrs;
      const { base, exponent } = powTex(attrs, s.in);
      return attrs.exponent === 2 ? `${base}\\odot ${base}` : `{${base}}^{${exponent}}`;
    },
    local: (s) => {
      const attrs = s.attrs as PowAttrs;
      const { base, exponent } = powTex(attrs, s.ins);
      if (powRole(attrs, s.arg) === 'exponent') return `{${base}}^{${exponent}}\\ln ${base}`;
      const p = attrs.exponent;
      if (p === undefined) return `${exponent}{${base}}^{${exponent}-1}`;
      return p - 1 === 1 ? `${p}${base}` : `${p}{${base}}^{${p - 1}}`;
    },
  },
};

export const transposeExplain: OpExplain = {
  pattern: 'elementwise',
  local: () => 1,
  tex: {
    forward: (s) => {
      const { dim0, dim1 } = s.attrs as { dim0: number; dim1: number };
      const matrix = new Set([dim0, dim1]);
      return (matrix.has(0) && matrix.has(1)) || (matrix.has(-2) && matrix.has(-1)) ? transposedTex(s.in[0]) : `\\operatorname{transpose}(${s.in[0]},${dim0},${dim1})`;
    },
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

function whereLocal(ctx: OpCtx, out: number[], arg: number): number {
  const keep = operand(ctx, 0, out) !== 0;
  return (arg === 1 && keep) || (arg === 2 && !keep) ? 1 : 0;
}

export const whereExplain: OpExplain = {
  pattern: 'elementwise',
  local: (ctx, out, arg) => whereLocal(ctx, out, arg),
  tex: {
    forward: (s) => `\\operatorname{where}(${s.in[0]},${s.in[1]},${s.in[2] ?? scalarTex(s.attrs)})`,
    local: (s) => String(whereLocal(s.ctx, s.outIndex, s.arg)),
  },
};

export const triuExplain: OpExplain = {
  pattern: 'elementwise',
  local: (ctx, out) => (triuKeeps(out, ctx.attrs as TriuAttrs) ? 1 : 0),
  tex: {
    forward: (s) => `\\operatorname{triu}(${s.in[0]},${(s.attrs as TriuAttrs).diagonal})`,
    local: (s) => (triuKeeps(s.outIndex, s.attrs as TriuAttrs) ? '1' : '0'),
  },
};

export const trilExplain: OpExplain = {
  pattern: 'elementwise',
  local: (ctx, out) => (trilKeeps(out, ctx.attrs as TriuAttrs) ? 1 : 0),
  tex: {
    forward: (s) => `\\operatorname{tril}(${s.in[0]},${(s.attrs as TriuAttrs).diagonal})`,
    local: (s) => (trilKeeps(s.outIndex, s.attrs as TriuAttrs) ? '1' : '0'),
  },
};

const valueTex = (attrs: object) => {
  const v = (attrs as { value: number }).value;
  return v < 0 ? `\\left(${formatNumber(v, { tex: true })}\\right)` : formatNumber(v, { tex: true });
};

export const maskedFillExplain: OpExplain = {
  pattern: 'elementwise',
  local: (ctx, out, arg) => (arg === 0 && operand(ctx, 1, out) === 0 ? 1 : 0),
  tex: {
    forward: (s) => `\\operatorname{masked\\_fill}(${s.in[0]},${s.in[1]},${valueTex(s.attrs)})`,
    local: (s) => (s.arg === 0 && operand(s.ctx, 1, s.outIndex) === 0 ? '1' : '0'),
  },
};

export const remainderExplain = binary(
  (a, b) => `${a}\\bmod ${b}`,
  (a, b, arg) => (arg === 0 ? 1 : -Math.floor(a / b)),
  (a, b, arg) => (arg === 0 ? '1' : `-\\left\\lfloor ${a}/${b}\\right\\rfloor`),
);
export const floorDivideExplain = binary((a, b) => `\\left\\lfloor ${a}/${b}\\right\\rfloor`, () => 0, () => '0');
export const bitwiseXorExplain = binary((a, b) => `${a}\\oplus ${b}`, () => 0, () => '0');
export const bitwiseAndExplain = binary((a, b) => `${a}\\wedge ${b}`, () => 0, () => '0');
export const logicalNotExplain = unary((a) => `\\lnot ${a}`, () => 0, () => '0');
export const eqExplain = binary((a, b) => `\\mathbf{1}[${a}=${b}]`, () => 0, () => '0');
export const geExplain = binary((a, b) => `\\mathbf{1}[${a}\\ge ${b}]`, () => 0, () => '0');
export const greaterExplain = binary((a, b) => `\\mathbf{1}[${a}>${b}]`, () => 0, () => '0');
export const ltExplain = binary((a, b) => `\\mathbf{1}[${a}<${b}]`, () => 0, () => '0');
export const bitwiseOrExplain = binary((a, b) => `${a}\\vee ${b}`, () => 0, () => '0');
export const copysignExplain = binary(
  (a, b) => `\\operatorname{copysign}(${a},${b})`,
  (a, b, arg) => (arg === 0 ? copysignLocal(a, b) : 0),
  (a, b, arg) => (arg === 0 ? `\\operatorname{sign}(${a})\\operatorname{sign}(${b})` : '0'),
);
export const absExplain = unary((a) => `\\lvert ${a}\\rvert`, (x) => Math.sign(x), (x) => `\\operatorname{sign}(${x})`);
export const signExplain = unary((a) => `\\operatorname{sign}(${a})`, () => 0, () => '0');
const minOf = (attrs: object) => (attrs as { min: number }).min;
export const clampMinExplain: OpExplain = {
  pattern: 'elementwise',
  local: (ctx, _out, _arg, input) => (ctx.inputs[0].get(input) > minOf(ctx.attrs) ? 1 : 0),
  tex: {
    forward: (s) => `\\max(${s.in[0]},${formatNumber(minOf(s.attrs), { tex: true })})`,
    local: (s) => `\\mathbf{1}[${s.in}>${formatNumber(minOf(s.attrs), { tex: true })}]`,
  },
};
