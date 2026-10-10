import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { expect } from 'vitest';
import { type Nested, Tensor } from '../../src/core';

export const ABS_TOLERANCE = 1e-9;
export const REL_TOLERANCE = 1e-9;

export type EncodedNumber = number | 'nan' | 'inf' | '-inf';

export interface EncodedTensor {
  shape: number[];
  values: EncodedNumber[];
}

export interface FixtureFile<C> {
  torch: string;
  command: string;
  seed: number;
  cases: C[];
}

export function loadFixture<C>(name: string): FixtureFile<C> {
  const path = resolve(import.meta.dirname, '../fixtures/torch', name);
  return JSON.parse(readFileSync(path, 'utf8')) as FixtureFile<C>;
}

export function decode(values: readonly EncodedNumber[]): number[] {
  return values.map((v) => {
    if (v === 'nan') return Number.NaN;
    if (v === 'inf') return Number.POSITIVE_INFINITY;
    if (v === '-inf') return Number.NEGATIVE_INFINITY;
    return v;
  });
}

export function nested(encoded: EncodedTensor): Nested {
  return Tensor.from(decode(encoded.values), encoded.shape).toNested();
}

function matches(actual: number, expected: number): boolean {
  if (Number.isNaN(expected)) return Number.isNaN(actual);
  if (!Number.isFinite(expected)) return actual === expected;
  return Math.abs(actual - expected) <= ABS_TOLERANCE + REL_TOLERANCE * Math.abs(expected);
}

export function expectTensor(actual: Tensor | null, expected: EncodedTensor, what: string): void {
  expect(actual, what).not.toBeNull();
  const tensor = actual as Tensor;
  expect([...tensor.shape], `${what} shape`).toEqual(expected.shape);
  const values = decode(expected.values);
  const mismatches: string[] = [];
  values.forEach((e, i) => {
    const a = tensor.data[i];
    if (!matches(a, e)) mismatches.push(`[${i}] vizflow ${a} vs torch ${e}`);
  });
  expect(mismatches, what).toEqual([]);
}
