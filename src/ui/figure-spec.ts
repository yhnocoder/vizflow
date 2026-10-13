import type { ValueExpr } from '../autograd';

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

export function encodeExpr(expr: ValueExpr): string {
  const bytes = new TextEncoder().encode(JSON.stringify(expr));
  return btoa(Array.from(bytes, (b) => String.fromCharCode(b)).join(''));
}

export function decodeExpr(text: string): ValueExpr {
  const binary = atob(text);
  return JSON.parse(new TextDecoder().decode(Uint8Array.from(binary, (c) => c.charCodeAt(0)))) as ValueExpr;
}

export function figureTag(name: string, expr: ValueExpr, args: readonly string[]): string {
  return `<vf-figure name="${name}" expr="${encodeExpr(expr)}" args="${args.join(':')}"></vf-figure>`;
}
