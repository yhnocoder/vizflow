export const TRANSITION_MS = 200;

export function easeOut(t: number): number {
  const clamped = Math.max(0, Math.min(1, t));
  return 1 - (1 - clamped) ** 3;
}

export class Transition {
  private from: Float64Array;
  private to: Float64Array;
  readonly current: Float64Array;
  private startedAt = 0;
  private running = false;

  constructor(
    size: number,
    private readonly immediate: () => boolean,
  ) {
    this.from = new Float64Array(size);
    this.to = new Float64Array(size);
    this.current = new Float64Array(size);
  }

  retarget(values: ArrayLike<number>, now: number): void {
    let changed = false;
    for (let i = 0; i < this.to.length; i++) {
      if (this.to[i] !== values[i]) changed = true;
    }
    if (!changed) return;
    this.from.set(this.current);
    this.to.set(values);
    if (this.immediate()) {
      this.current.set(values);
      this.running = false;
      return;
    }
    this.startedAt = now;
    this.running = true;
  }

  step(now: number): boolean {
    if (!this.running) return false;
    const t = (now - this.startedAt) / TRANSITION_MS;
    const k = easeOut(t);
    for (let i = 0; i < this.current.length; i++) this.current[i] = this.from[i] + (this.to[i] - this.from[i]) * k;
    if (t >= 1) {
      this.current.set(this.to);
      this.running = false;
    }
    return true;
  }

  get active(): boolean {
    return this.running;
  }
}
