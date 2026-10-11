export const WAVE_INVERSE: Record<string, (y: number) => number> = {
  cos: Math.acos,
  sin: Math.asin,
};

const CHECKS: Record<string, (shape: readonly number[], args: string[]) => string | undefined> = {
  rope_wave(shape, args) {
    if (!(args[0] in WAVE_INVERSE)) return `rope_wave needs cos or sin as its first argument, got '${args[0] ?? ''}'`;
    if (args[1] !== undefined && !(Number.isInteger(Number(args[1])) && Number(args[1]) >= 1)) return `rope_wave positions must be a positive integer, got '${args[1]}'`;
    if (args.length > 2) return 'rope_wave takes at most 2 arguments';
    if (shape.length !== 2 || shape[0] < 2 || shape[1] % 2 !== 0) return 'rope_wave needs a (T, d_h) table with at least 2 rows and an even d_h';
    return undefined;
  },
};

export const figureNames: readonly string[] = Object.keys(CHECKS);

export function checkFigure(name: string, shape: readonly number[], args: string[]): string | undefined {
  const check = CHECKS[name];
  if (check === undefined) return `unknown figure '${name}'`;
  return check(shape, args);
}

export function figureTag(name: string, node: string, args: string[]): string {
  return `<vf-figure name="${name}" node="${node}" args="${args.join(':')}"></vf-figure>`;
}
