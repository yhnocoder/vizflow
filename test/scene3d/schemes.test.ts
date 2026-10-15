import { describe, expect, it } from 'vitest';
import type { Node } from '../../src/autograd';
import { type Place, type SchemeGraph, schemeLevels } from '../../src/scene3d/schemes';

interface Toy {
  name: string;
  inputs?: string[];
  scope?: string;
  pack?: string;
  rides?: boolean;
  iteration?: string;
}

function toyGraph(toys: Toy[], places: { node: string; key: Place['key']; of: string }[] = []): { g: SchemeGraph; node: (name: string) => Node } {
  const nodes = toys.map((t, id) => ({ id, name: t.name }) as Node);
  const byName = new Map(nodes.map((n) => [n.name, n]));
  const toyOf = new Map(nodes.map((n, i) => [n, toys[i]]));
  const node = (name: string) => byName.get(name)!;
  const g: SchemeGraph = {
    nodes,
    preds: (n) => (toyOf.get(n)!.inputs ?? []).map(node),
    constant: () => false,
    scope: (n) => toyOf.get(n)!.scope ?? null,
    parent: () => null,
    pack: (n) => toyOf.get(n)!.pack,
    rides: (n) => toyOf.get(n)!.rides ?? false,
    late: () => false,
    iteration: (n) => toyOf.get(n)!.iteration,
    places: places.map((p) => ({ node: node(p.node), key: p.key, of: node(p.of) })),
  };
  return { g, node };
}

const EXPERTS = 6;

describe('nested levels', () => {
  it('keeps per-expert copies on one level when the width cap is exceeded', () => {
    const toys: Toy[] = [{ name: 'x' }, { name: 'd', inputs: ['x'] }, { name: 's1', inputs: ['d'] }, { name: 's2', inputs: ['s1'] }, { name: 's3', inputs: ['s2'] }, { name: 's4', inputs: ['s3'] }];
    for (let e = 0; e < EXPERTS; e++) toys.push({ name: `xs[${e}]`, inputs: ['d'] }, { name: `h[${e}]`, inputs: [`xs[${e}]`] }, { name: `y[${e}]`, inputs: [`h[${e}]`] });
    toys.push({ name: 'out', inputs: ['s4', ...Array.from({ length: EXPERTS }, (_, e) => `y[${e}]`)] });
    const { g, node } = toyGraph(toys);
    for (const iterations of ['overlap', 'together'] as const) {
      const level = schemeLevels('nested', g, { iterations, width: 4 });
      for (const prefix of ['xs', 'h', 'y']) expect(new Set(Array.from({ length: EXPERTS }, (_, e) => level.get(node(`${prefix}[${e}]`))))).toHaveLength(1);
    }
  });

  it('keeps each head copy of the log-sum-exp next to its probabilities', () => {
    const toys: Toy[] = [{ name: 'q' }];
    for (let h = 0; h < 3; h++) {
      toys.push(
        { name: `S0[${h}]`, inputs: ['q'], scope: 'round0' },
        { name: `P0[${h}]`, inputs: [`S0[${h}]`], scope: 'round0' },
        { name: `lse0[${h}]`, inputs: [`S0[${h}]`], scope: 'round0' },
        { name: `o0[${h}]`, inputs: [`P0[${h}]`], scope: 'round0' },
        { name: `on0[${h}]`, inputs: [`o0[${h}]`], scope: 'round0' },
        { name: `merged[${h}]`, inputs: [`on0[${h}]`, `lse0[${h}]`], scope: 'round0' },
      );
    }
    const { g, node } = toyGraph(toys);
    for (const iterations of ['overlap', 'together'] as const) {
      const level = schemeLevels('nested', g, { iterations, width: 4 });
      for (let h = 0; h < 3; h++) {
        expect(level.get(node(`P0[${h}]`))).toBe(level.get(node('P0[0]')));
        expect(level.get(node(`lse0[${h}]`))).toBe(level.get(node('P0[0]')));
      }
    }
  });

  it('aligns communications of the same group in one scope to the latest of them', () => {
    const { g, node } = toyGraph([
      { name: 'x' },
      { name: 'v', inputs: ['x'] },
      { name: 'k', inputs: ['x'] },
      { name: 'kr', inputs: ['k'] },
      { name: 'k1', inputs: ['kr'], pack: 'send_recv:cp' },
      { name: 'v1', inputs: ['v'], pack: 'send_recv:cp' },
    ]);
    const level = schemeLevels('nested', g);
    expect(level.get(node('v1'))).toBe(level.get(node('k1')));
    expect(level.get(node('k1'))).toBe(level.get(node('kr'))! + 1);
  });

  it('does not hold a ring attention round back for a communication it does not use', () => {
    const toys: Toy[] = [{ name: 'x' }];
    for (let r = 0; r < 2; r++) toys.push({ name: `k[${r}]`, inputs: ['x'] });
    for (let r = 0; r < 2; r++) toys.push({ name: `K[${r}][0]`, inputs: [`k[${r}]`] }, { name: `S[${r}][0]`, inputs: [`K[${r}][0]`] }, { name: `recv[${r}][0]`, inputs: [`k[${1 - r}]`], pack: 'send_recv:cp' });
    for (let r = 0; r < 2; r++) toys.push({ name: `K[${r}][1]`, inputs: [`recv[${r}][0]`] }, { name: `S[${r}][1]`, inputs: [`K[${r}][1]`] });
    const { g, node } = toyGraph(toys);
    for (const iterations of ['overlap', 'together'] as const) {
      const level = schemeLevels('nested', g, { iterations, width: 8 });
      for (let r = 0; r < 2; r++) {
        expect(level.get(node(`K[${r}][0]`))).toBe(level.get(node(`k[${r}]`))! + 1);
        expect(level.get(node(`S[${r}][0]`))).toBe(level.get(node(`K[${r}][0]`))! + 1);
        expect(level.get(node(`S[${r}][0]`))).toBeLessThan(level.get(node(`S[${r}][1]`))!);
        expect(level.get(node(`S[${r}][1]`))).toBe(level.get(node('S[0][1]')));
      }
    }
  });

  it('starts each ring attention round as soon as its own key block is there when rounds are loops over heads', () => {
    const toys: Toy[] = [{ name: 'x' }, { name: 'kr', inputs: ['x'] }, { name: 'q', inputs: ['x'] }, { name: 'qs', inputs: ['q'] }];
    const round = (r: number, keys: string) => {
      for (let h = 0; h < 2; h++) toys.push({ name: `K${r}[${h}]`, inputs: [keys], iteration: `${r + 1}.${h}` }, { name: `S${r}[${h}]`, inputs: ['qs', `K${r}[${h}]`], iteration: `${r + 1}.${h}` });
    };
    round(0, 'kr');
    toys.push({ name: 'k1', inputs: ['kr'], pack: 'send_recv:cp' });
    round(1, 'k1');
    for (let h = 0; h < 2; h++) toys.push({ name: `om[${h}]`, inputs: [`S0[${h}]`, `S1[${h}]`], iteration: `3.${h}` });
    const { g, node } = toyGraph(toys);
    for (const iterations of ['overlap', 'together'] as const) {
      const level = schemeLevels('nested', g, { iterations, width: 8 });
      for (let h = 0; h < 2; h++) {
        expect(level.get(node(`K0[${h}]`))).toBe(level.get(node('k1')));
        expect(level.get(node(`S0[${h}]`))).toBeLessThan(level.get(node(`S1[${h}]`))!);
      }
    }
  });

  it('puts a left matrix shared by matmuls on their level', () => {
    const { g, node } = toyGraph([{ name: 'x' }, { name: 'h', inputs: ['x'], rides: true }, { name: 'q', inputs: ['h'] }, { name: 'k', inputs: ['h'] }]);
    const level = schemeLevels('nested', g);
    expect(level.get(node('h'))).toBe(level.get(node('q')));
    expect(level.get(node('q'))).toBe(level.get(node('x'))! + 1);
  });

  it('rejects a place that puts a node on or below its input', () => {
    const { g } = toyGraph([{ name: 'x' }, { name: 'a', inputs: ['x'] }, { name: 'b', inputs: ['a'] }], [{ node: 'b', key: 'with', of: 'a' }]);
    for (const scheme of ['asap', 'nested'] as const) expect(() => schemeLevels(scheme, g)).toThrow(/place 'with' of 'b' puts 'b' on or below its input 'a'/);
  });
});
