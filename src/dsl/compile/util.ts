import { type Where, child, fail } from '../source';
import type { Num, RealSize } from './types';

export function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

export function isNum(value: unknown): value is Num {
  return isRecord(value) && typeof value.toy === 'number' && typeof value.real === 'number';
}

export function num(n: number): Num {
  return { toy: n, real: n };
}

export function realOf(n: Num): RealSize {
  return n.derive ?? (() => n.real);
}

export function combine(nums: Num[], fn: (...values: number[]) => number): Num {
  const out: Num = { toy: fn(...nums.map((n) => n.toy)), real: fn(...nums.map((n) => n.real)) };
  if (nums.some((n) => n.derive !== undefined)) {
    const parts = nums.map(realOf);
    out.derive = (sizes, shapes) => fn(...parts.map((part) => part(sizes, shapes)));
  }
  return out;
}

export function join(path: string, name: string): string {
  return path === '' ? name : `${path}.${name}`;
}

export function list<T>(value: T | T[] | undefined): T[] {
  return value === undefined ? [] : Array.isArray(value) ? value : [value];
}

export function dtypeOption(decl: Record<string, unknown>, at: Where): { dtype?: string } {
  if (decl.dtype === undefined) return {};
  if (typeof decl.dtype !== 'string') fail(child(at, decl, 'dtype'), 'dtype must be a string');
  return { dtype: decl.dtype };
}

export function checkKeys(obj: Record<string, unknown>, allowed: ReadonlySet<string>, where: Where): void {
  for (const key of Object.keys(obj)) if (!allowed.has(key)) fail(child(where, obj, key), `unknown key '${key}'`);
}

export function record(value: unknown, where: Where, what: string): Record<string, unknown> {
  if (value === undefined) return {};
  if (!isRecord(value)) fail(where, `${what} must be a mapping`);
  return value;
}
