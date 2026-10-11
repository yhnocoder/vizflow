import { describe, expect, it } from 'vitest';
import { graph } from '../../src/core';

describe('dtype', () => {
  it('uses the graph dtype for leaves, int64 for labels and promotes floating inputs for ops', () => {
    const g = graph({ dims: { T: 2, V: 3 }, dtype: 'bf16' });
    const ids = g.input('ids', ['T'], [0, 2], { dtype: 'int64' });
    const E = g.param('E', ['V', 'V'], [[1, 0, 0], [0, 1, 0], [0, 0, 1]]);
    const x = g.embedding(ids, E, { name: 'x' });
    const z = g.log_softmax(x, -1, { name: 'z', dtype: 'fp32' });
    const y = g.label('y', [1, 0], { classes: 3 });
    const loss = g.nll_loss(z, y, { name: 'loss' });
    const back = g.mul(z, x, { name: 'back' });
    expect([ids, E, x, z, y, loss, back].map((n) => n.dtype)).toEqual(['int64', 'bf16', 'bf16', 'fp32', 'int64', 'fp32', 'fp32']);
    expect(E.T.dtype).toBe('bf16');
  });

  it('defaults to fp32 and promotes bf16 with fp16 to fp32', () => {
    const g = graph({ dims: { V: 2 } });
    const a = g.input('a', ['V'], [1, 2], { dtype: 'bf16' });
    const b = g.input('b', ['V'], [1, 2], { dtype: 'fp16' });
    expect(g.dtype).toBe('fp32');
    expect(g.add(a, b).dtype).toBe('fp32');
    expect(g.add(a, a).dtype).toBe('bf16');
  });

  it('rejects unknown dtypes', () => {
    expect(() => graph({ dtype: 'float' })).toThrow(/unknown dtype 'float'/);
    const g = graph({ dims: { V: 2 } });
    expect(() => g.input('a', ['V'], [1, 2], { dtype: 'f32' })).toThrow(/node 'a': unknown dtype 'f32'/);
  });
});
