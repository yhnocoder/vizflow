import { describe, expect, it } from 'vitest';
import { formatNumber, formatShape } from '../../src/tensor';

describe('formatShape', () => {
  it('writes shapes like Python tuples', () => {
    expect(formatShape([])).toBe('()');
    expect(formatShape([5])).toBe('(5,)');
    expect(formatShape([1, 4])).toBe('(1, 4)');
  });
});

describe('formatNumber', () => {
  it('picks digits by magnitude', () => {
    expect(formatNumber(123.456)).toBe('123');
    expect(formatNumber(12.345)).toBe('12.3');
    expect(formatNumber(1.2345)).toBe('1.23');
    expect(formatNumber(0.12345)).toBe('0.123');
  });

  it('uses the given digits', () => {
    expect(formatNumber(1.2345, { digits: 1 })).toBe('1.2');
    expect(formatNumber(1.2345, { digits: 4 })).toBe('1.2345');
  });

  it('drops the sign of negative zero', () => {
    expect(formatNumber(-0.0001)).toBe('0.000');
    expect(formatNumber(-0)).toBe('0.000');
    expect(formatNumber(-0.0001, { digits: 0 })).toBe('0');
  });

  it('writes minus as U+2212 unless tex is requested', () => {
    expect(formatNumber(-1.5)).toBe('−1.50');
    expect(formatNumber(-1.5, { tex: true })).toBe('-1.50');
  });
});
