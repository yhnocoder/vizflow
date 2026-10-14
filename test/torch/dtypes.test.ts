import { describe, expect, it } from 'vitest';
import { type Node, graph } from '../../src/core';
import { loadFixture } from './fixtures';

interface DtypeCase {
  op: string;
  inputs: string[];
  args: number[];
  output: string;
}

const fixture = loadFixture<DtypeCase>('dtypes.json');

describe(`PyTorch ${fixture.torch} result dtype`, () => {
  for (const c of fixture.cases) {
    it(`${c.op}(${c.inputs.join(', ')}) is ${c.output}`, () => {
      const g = graph();
      const inputs = c.inputs.map((dtype, i) => g.input(`x${i}`, [2], [1, 2], { dtype }));
      const call = (g as unknown as Record<string, (...a: unknown[]) => Node>)[c.op];
      expect(call(...inputs, ...c.args, { name: 'out' }).dtype).toBe(c.output);
    });
  }
});
