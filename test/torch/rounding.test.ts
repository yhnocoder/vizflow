import { describe, expect, it } from 'vitest';
import type { Dtype } from '../../src/autograd';
import { roundToDtype } from '../../src/autograd/rounding';
import { type EncodedNumber, decode, loadFixture } from './fixtures';

interface RoundingCase {
  dtype: Dtype;
  x: EncodedNumber[];
  y: EncodedNumber[];
}

const fixture = loadFixture<RoundingCase>('rounding.json');

describe('rounding matches torch casts from float64', () => {
  for (const c of fixture.cases) {
    it(c.dtype, () => {
      const x = decode(c.x);
      const y = decode(c.y);
      const mismatches = x.flatMap((v, i) => (Object.is(roundToDtype(v, c.dtype), y[i]) || (y[i] === 0 && roundToDtype(v, c.dtype) === 0) ? [] : [`${v}: vizflow ${roundToDtype(v, c.dtype)} vs torch ${y[i]}`]));
      expect(mismatches).toEqual([]);
    });
  }
});
