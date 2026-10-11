import { Tensor, formatShape, indexOf } from '../tensor';
import type { AnyOpDef, OpDef } from './op';

export const CHECK_EPSILON = 1e-6;
export const CHECK_ABS_TOLERANCE = 1e-6;
export const CHECK_REL_TOLERANCE = 1e-4;
export const CHECK_KINK_TOLERANCE = 1e-3;
const CHECK_SEED = 0x9e3779b9;

function seededRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function dot(a: Tensor, b: Tensor): number {
  let s = 0;
  for (let i = 0; i < a.data.length; i++) s += a.data[i] * b.data[i];
  return s;
}

export function checkOpNamed<A extends object>(
  def: OpDef<A>,
  inputs: Tensor[],
  attrs: A,
  nodeName: string | undefined,
): void {
  const out = def.forward(inputs, attrs);
  const random = seededRandom(CHECK_SEED);
  const upstream = new Tensor(out.shape, Float64Array.from({ length: out.size }, () => random() * 2 - 1));
  const grads = def.backward(upstream, inputs, out, attrs);
  const base = dot(upstream, out);
  const where = nodeName === undefined ? '' : ` (node '${nodeName}')`;
  inputs.forEach((input, arg) => {
    if (def.differentiable && !def.differentiable[arg]) return;
    const what = `op '${def.name}'${where}, input ${arg} '${def.list === undefined ? def.tensors[arg] : `${def.list}[${arg}]`}'`;
    const grad = grads[arg];
    if (grad === null || grad === undefined) {
      throw new Error(`gradient check failed: ${what} has no gradient`);
    }
    if (formatShape(grad.shape) !== formatShape(input.shape)) {
      throw new Error(
        `gradient check failed: ${what}: backward returned ${formatShape(grad.shape)} for shape ${formatShape(input.shape)}`,
      );
    }
    for (let j = 0; j < input.size; j++) {
      const original = input.data[j];
      input.data[j] = original + CHECK_EPSILON;
      const plus = dot(upstream, def.forward(inputs, attrs));
      input.data[j] = original - CHECK_EPSILON;
      const minus = dot(upstream, def.forward(inputs, attrs));
      input.data[j] = original;
      const numeric = (plus - minus) / (2 * CHECK_EPSILON);
      const tolerance = CHECK_ABS_TOLERANCE + CHECK_REL_TOLERANCE * Math.abs(numeric);
      const rightSided = (plus - base) / CHECK_EPSILON;
      const leftSided = (base - minus) / CHECK_EPSILON;
      if (!(Math.abs(rightSided - leftSided) <= CHECK_KINK_TOLERANCE * (1 + Math.abs(numeric)))) continue;
      const analytic = grad.data[j];
      if (!(Math.abs(analytic - numeric) <= tolerance)) {
        throw new Error(
          `gradient check failed: ${what} at [${indexOf(input.shape, j).join(', ')}]: backward ${analytic.toFixed(6)} vs numeric ${numeric.toFixed(6)}`,
        );
      }
    }
  });
}

export function checkOp(def: AnyOpDef, inputs: Tensor[], attrs: object): void {
  checkOpNamed(def, inputs.map((t) => t.clone()), attrs, undefined);
}
