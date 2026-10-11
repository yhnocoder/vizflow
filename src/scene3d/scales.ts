export const SCALE_FILL = 0.8;
export const BAR_CAP = 1.2;
export const TILE_CAP = 0.85;
export const VECTOR_CAP = 1.0;
export const HEIGHT_CAP = BAR_CAP;
export const MIN_HEIGHT = 0.025;
export const BREAK_THICKNESS = 0.07;
export const BREAK_POSITION = 0.8;
export const ZERO_THRESHOLD = 1e-9;
export const TICK_COUNT = 3;

const SCALE_FLOOR = 1e-6;

export function capFor(shape: 'bar' | 'tile' | 'disc' | 'box', dims: number): number {
  if (shape === 'tile') return TILE_CAP;
  return dims === 1 ? VECTOR_CAP : BAR_CAP;
}

export function scaleFor(values: ArrayLike<number>, cap = BAR_CAP): number {
  let max = 0;
  for (let i = 0; i < values.length; i++) if (Number.isFinite(values[i])) max = Math.max(max, Math.abs(values[i]));
  return max === 0 ? 1 : (cap * SCALE_FILL) / max;
}

export interface BarHeight {
  height: number;
  clipped: boolean;
}

export function barHeight(value: number, scale: number, revealed: boolean, cap = BAR_CAP): BarHeight {
  if (!revealed) return { height: 0, clipped: false };
  let height = value * scale;
  let clipped = false;
  if (Math.abs(height) > cap) {
    height = Math.sign(height) * cap;
    clipped = true;
  }
  if (Math.abs(height) < MIN_HEIGHT) height = value < 0 ? -MIN_HEIGHT : MIN_HEIGHT;
  return { height, clipped };
}

const NICE_STEPS = [1, 2, 2.5, 5];

function niceStepsBetween(span: number): number[] {
  const steps: number[] = [];
  const top = Math.ceil(Math.log10(Math.max(span, SCALE_FLOOR)));
  for (let power = top; power >= top - 3; power--) {
    for (const base of [...NICE_STEPS].reverse()) steps.push(base * 10 ** power);
  }
  return steps;
}

function ticksWithStep(lo: number, hi: number, step: number): number[] {
  const ticks: number[] = [];
  for (let k = Math.floor(lo / step); k <= Math.ceil(hi / step); k++) ticks.push(Number((k * step).toPrecision(12)));
  return ticks;
}

export function niceTicks(values: ArrayLike<number>, scale: number, count = TICK_COUNT, cap = BAR_CAP): number[] {
  let lo = 0;
  let hi = 0;
  for (let i = 0; i < values.length; i++) {
    if (!Number.isFinite(values[i])) continue;
    lo = Math.min(lo, values[i]);
    hi = Math.max(hi, values[i]);
  }
  const limit = cap / scale;
  const within = (ticks: number[]) => ticks.every((v) => Math.abs(v) <= limit + 1e-9);
  const span = Math.max(hi - lo, SCALE_FLOOR);
  let fallback: number[] = [0];
  for (const step of niceStepsBetween(span)) {
    const ticks = ticksWithStep(lo, hi, step);
    if (!within(ticks)) continue;
    if (ticks.length === count) return ticks;
    if (ticks.length > count) break;
    fallback = ticks;
  }
  return fallback;
}
