import type { Program } from '../autograd';

export type FigureArgs = Record<string, number>;

const KEYS: Record<string, readonly string[]> = { lines: ['shade'] };

const CHECKS: Record<string, (shape: readonly number[], args: FigureArgs) => string | undefined> = {
  lines(shape, args) {
    if (shape.length !== 2 || shape[0] < 2 || shape[1] < 1) return `lines needs a (N, K) tensor with at least 2 rows, got shape (${shape.join(', ')})`;
    const shade = args.shade;
    if (shade !== undefined && !(Number.isInteger(shade) && shade >= 0 && shade <= shape[0])) return `lines shade must be an integer from 0 to ${shape[0]}, got ${shade}`;
    return undefined;
  },
};

export const figureNames: readonly string[] = Object.keys(CHECKS);

export function figureArgs(name: string, args: readonly string[]): FigureArgs | string {
  const keys = KEYS[name];
  if (keys === undefined) return `unknown figure '${name}'`;
  const out: FigureArgs = {};
  for (const arg of args) {
    const eq = arg.indexOf('=');
    const key = eq < 0 ? arg : arg.slice(0, eq).trim();
    if (eq < 0 || !keys.includes(key)) return `${name} takes ${keys.map((k) => `${k}=value`).join(', ')}, got '${arg}'`;
    if (key in out) return `${name} gives '${key}' twice`;
    const value = Number(arg.slice(eq + 1).trim());
    if (!Number.isFinite(value)) return `${name} ${key} must be a number, got '${arg.slice(eq + 1).trim()}'`;
    out[key] = value;
  }
  return out;
}

export function checkFigure(name: string, shape: readonly number[], args: readonly string[]): string | undefined {
  const parsed = figureArgs(name, args);
  if (typeof parsed === 'string') return parsed;
  return CHECKS[name](shape, parsed);
}

const NON_FINITE = 'nonFinite';

export function encodeProgram(program: Program): string {
  const json = JSON.stringify(program, (_key, value: unknown) => (typeof value === 'number' && !Number.isFinite(value) ? { [NON_FINITE]: String(value) } : value));
  const bytes = new TextEncoder().encode(json);
  return btoa(Array.from(bytes, (b) => String.fromCharCode(b)).join(''));
}

export function decodeProgram(text: string): Program {
  const binary = atob(text);
  const json = new TextDecoder().decode(Uint8Array.from(binary, (c) => c.charCodeAt(0)));
  return JSON.parse(json, (_key, value: unknown) =>
    typeof value === 'object' && value !== null && NON_FINITE in value && Object.keys(value).length === 1 ? Number((value as Record<string, string>)[NON_FINITE]) : value,
  ) as Program;
}

export function figureTag(name: string, program: Program, args: readonly string[]): string {
  return `<vf-figure name="${name}" program="${encodeProgram(program)}" args="${args.join(':')}"></vf-figure>`;
}
