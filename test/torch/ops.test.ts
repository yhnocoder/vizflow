import { describe, expect, it } from 'vitest';
import { type Node, builtinOps, graph } from '../../src/core';
import { type EncodedTensor, decode, expectFinite, expectTensor, loadFixture, nested } from './fixtures';

interface OpCase {
  op: string;
  title: string;
  tensors: (EncodedTensor & { kind: 'input' | 'label' })[];
  args: (number | number[] | { tensor: number } | { tensor: number }[])[];
  options: Record<string, unknown>;
  output: EncodedTensor;
  upstream: EncodedTensor;
  grads: (EncodedTensor | null)[];
  finite?: boolean;
}

const fixture = loadFixture<OpCase>('ops.json');

function isTensorRef(arg: unknown): arg is { tensor: number } {
  return typeof arg === 'object' && arg !== null && !Array.isArray(arg);
}

function run(c: OpCase) {
  const g = graph();
  const leaves = c.tensors.map((t, i) =>
    t.kind === 'label' ? g.label(`t${i}`, nested(t)) : g.input(`x${i}`, t.shape, decode(t.values)),
  );
  const args = c.args.map((arg) => {
    if (isTensorRef(arg)) return leaves[arg.tensor];
    if (Array.isArray(arg) && arg.every(isTensorRef)) return arg.map((ref) => leaves[ref.tensor]);
    return arg;
  });
  const call = (g as unknown as Record<string, (...a: unknown[]) => Node>)[c.op];
  const out = call(...args, { ...c.options, name: 'out' });
  const upstream = g.input('upstream', c.upstream.shape, decode(c.upstream.values));
  g.loss(g.sum(g.mul(out, upstream)));
  return { out, leaves };
}

describe(`PyTorch ${fixture.torch} reference for each op`, () => {
  it('covers every builtin op', () => {
    expect(new Set(fixture.cases.map((c) => c.op))).toEqual(new Set(builtinOps.map((def) => def.name)));
  });

  for (const c of fixture.cases) {
    it(`${c.op}: ${c.title}`, () => {
      const { out, leaves } = run(c);
      expectTensor(out.value, c.output, 'forward');
      leaves.forEach((leaf, i) => {
        const expected = c.grads[i];
        if (expected === null) expect(leaf.grad).toBeNull();
        else expectTensor(leaf.grad, expected, `grad of tensor ${i}`);
      });
      if (c.finite) {
        expectFinite(out.value, 'forward');
        leaves.forEach((leaf, i) => expectFinite(leaf.grad, `grad of tensor ${i}`));
      }
    });
  }
});
