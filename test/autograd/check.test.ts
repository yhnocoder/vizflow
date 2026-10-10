import { describe, expect, it } from 'vitest';
import { Tensor, checkOp, opDef } from '../../src/core';

function filled(shape: number[], seed: number, offset = 0): Tensor {
  const data = Float64Array.from({ length: shape.reduce((a, b) => a * b, 1) }, (_, i) => {
    const v = Math.sin(seed * 7.3 + i * 1.7) * 1.5;
    return v + offset;
  });
  return new Tensor(shape, data);
}

function positive(shape: number[], seed: number): Tensor {
  const t = filled(shape, seed);
  return new Tensor(shape, t.data.map((v) => Math.abs(v) + 0.5));
}

function check(name: string, inputs: Tensor[], attrs: object = {}): void {
  const def = opDef(name);
  expect(def, name).toBeDefined();
  checkOp(def!, inputs, attrs);
}

const binaryShapes: [number[], number[]][] = [
  [[2, 3], [2, 3]],
  [[2, 3], [3]],
  [[2, 3], [1, 3]],
  [[2, 1], [1, 3]],
  [[], [4]],
  [[1, 5], []],
  [[3, 1, 2], [4, 1]],
];

describe('checkOp on elementwise ops', () => {
  it.each(binaryShapes)('add, sub, mul and div with %j and %j', (a, b) => {
    for (const name of ['add', 'sub', 'mul']) check(name, [filled(a, 1), filled(b, 2)]);
    check('div', [filled(a, 3), positive(b, 4)]);
  });

  it.each([[[]], [[5]], [[1, 4]], [[2, 3, 2]]])('neg, exp and log with %j', (shape) => {
    check('neg', [filled(shape, 5)]);
    check('exp', [filled(shape, 6)]);
    check('log', [positive(shape, 7)]);
  });

  it.each([
    [[2, 3], 0, 1],
    [[2, 3], -1, -2],
    [[2, 3, 4], 0, 2],
    [[2, 3, 4], 1, 1],
  ])('transpose with %j dims %i and %i', (shape, dim0, dim1) => {
    check('transpose', [filled(shape, 8)], { dim0, dim1 });
  });

  it('does not modify the inputs it perturbs', () => {
    const x = filled([3], 9);
    const before = Array.from(x.data);
    check('exp', [x]);
    expect(Array.from(x.data)).toEqual(before);
  });
});
