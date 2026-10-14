import { figureNames } from '../../ui/figure-spec';

export const INTERPOLATION = /\{\{([^{}]*)\}\}/g;
export const VIZ = /\\viz\{([^{}]+)\}/g;

export interface Interpolation {
  expr: string;
  format: string | undefined;
  arg: string | undefined;
}

export type FigureArg = string | [key: string, value: string];

function splitFilter(source: string): { expr: string; filter: string | undefined } {
  let depth = 0;
  let quote = '';
  for (let i = 0; i < source.length; i++) {
    const c = source[i];
    if (quote !== '') {
      if (c === quote) quote = '';
    } else if (c === '"' || c === "'") quote = c;
    else if (c === '[' || c === '(') depth++;
    else if (c === ']' || c === ')') depth--;
    else if (c === '|' && depth === 0) return { expr: source.slice(0, i).trim(), filter: source.slice(i + 1).trim() };
  }
  return { expr: source.trim(), filter: undefined };
}

export function interpolation(source: string): Interpolation {
  const { expr, filter } = splitFilter(source);
  if (filter === undefined) return { expr, format: undefined, arg: undefined };
  const colon = filter.indexOf(':');
  return colon < 0 ? { expr, format: filter.trim(), arg: undefined } : { expr, format: filter.slice(0, colon).trim(), arg: filter.slice(colon + 1).trim() };
}

export function figureArgs(arg: string | undefined): FigureArg[] {
  return (arg === undefined ? [] : arg.split(':')).map((part) => {
    const eq = part.indexOf('=');
    return eq < 0 ? part.trim() : [part.slice(0, eq).trim(), part.slice(eq + 1).trim()];
  });
}

export function vizReferences(text: string): string[] {
  return [...text.matchAll(VIZ)].map((match) => match[1].trim());
}

export function templateExpressions(text: string): string[] {
  const found: string[] = [];
  for (const match of text.matchAll(INTERPOLATION)) {
    const { expr, format, arg } = interpolation(match[1]);
    found.push(expr);
    if (format !== undefined && figureNames.includes(format)) for (const item of figureArgs(arg)) if (typeof item !== 'string') found.push(item[1]);
  }
  return [...found, ...vizReferences(text)];
}
