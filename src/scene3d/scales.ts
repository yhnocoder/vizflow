export const UNIT_HEIGHT = 1.6;
export const HEIGHT_CAP = 2.0;
export const MIN_HEIGHT = 0.025;
export const BREAK_THICKNESS = 0.07;
export const BREAK_POSITION = 0.8;
export const ZERO_THRESHOLD = 1e-9;
export const TICK_COUNT = 3;

const SCALE_FLOOR = 1e-6;

export function scaleFor(values: ArrayLike<number>): number {
  let max = 0;
  for (let i = 0; i < values.length; i++) max = Math.max(max, Math.abs(values[i]));
  return UNIT_HEIGHT / Math.max(max, SCALE_FLOOR);
}

export interface BarHeight {
  height: number;
  clipped: boolean;
}

export function barHeight(value: number, scale: number, revealed: boolean): BarHeight {
  if (!revealed) return { height: 0, clipped: false };
  let height = value * scale;
  let clipped = false;
  if (Math.abs(height) > HEIGHT_CAP) {
    height = Math.sign(height) * HEIGHT_CAP;
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

export function niceTicks(values: ArrayLike<number>, scale: number, count = TICK_COUNT): number[] {
  let lo = 0;
  let hi = 0;
  for (let i = 0; i < values.length; i++) {
    lo = Math.min(lo, values[i]);
    hi = Math.max(hi, values[i]);
  }
  const limit = HEIGHT_CAP / scale;
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
