import { describe, expect, it } from 'vitest';
import type { Node } from '../../src/autograd';
import { graph } from '../../src/core';
import '../../src/explain';
import { SCALAR_ROW_GAP } from '../../src/scene3d/layout';
import { DISC_SIDE, type LevelSpec, levelPlan } from '../../src/scene3d/levels';
import type { StepsSpec } from '../../src/steps';

const T = 2;
const D = 4;

function normGraph(together: boolean) {
  const g = graph({ dims: { T, d: D } });
  const x = g.input('x', [T, D], Array.from({ length: T }, (_, t) => Array.from({ length: D }, (_, c) => Math.sin(1 + t + c * 0.6))));
  const sq = g.pow(x, 2, { name: 'sq' });
  const ms = g.mean(sq, -1, { keepdim: true, name: 'ms' });
  const shifted = g.add(ms, 1e-6, { name: 'shifted' });
  const r = g.rsqrt(shifted, { name: 'r' });
  const hat = g.mul(x, r, { name: 'hat' });
  const loss = g.loss(g.sum(hat, null, { name: 'loss' }));
  const show = (nodes: Node[]) => ({ show: nodes, title: nodes.map((n) => n.name).join(', ') });
  const forward = together ? [show([x]), show([sq, ms, shifted, r]), show([hat]), show([loss])] : [show([x]), show([sq]), show([ms, shifted, r]), show([hat]), show([loss])];
  const spec: StepsSpec = { forward, backward: 'reverse' };
  const levels: LevelSpec = { lines: [[x], [sq], [ms], [shifted], [r], [hat], [loss]], sides: [], merges: [], slots: [], grids: [] };
  return { spec, levels, sq, ms, shifted, r };
}

function blocksOf(together: boolean) {
  const { spec, levels, ...nodes } = normGraph(together);
  const { blocks } = levelPlan(spec, levels).resolve({ folded: () => false, aligned: new Map() });
  const at = (n: Node) => blocks.find((b) => b.node === n)!;
  return { at, ...nodes };
}

describe('one-column reductions of a tile in the same step', () => {
  it('lie on the level of the tile, to its right, along the dependency chain', () => {
    const { at, sq, ms, shifted, r } = blocksOf(true);
    const host = at(sq);
    const right = Math.max(...host.elements.map((e) => e.x));
    const chain = [ms, shifted, r].map(at);
    for (const block of chain) expect(block.shape).toBe('disc');
    for (const block of chain) expect(block.elements[0].y).toBeCloseTo(host.elements[0].y);
    chain.forEach((block, k) => expect(block.elements[0].x).toBeCloseTo(right + DISC_SIDE + k * SCALAR_ROW_GAP));
  });

  it('alternate their titles above and below the row', () => {
    const { at, ms, shifted, r } = blocksOf(true);
    expect([ms, shifted, r].map((n) => at(n).title.anchor)).toEqual(['left', 'below', 'left']);
  });

  it('keep their own levels when they are shown in another step than the tile', () => {
    const { at, sq, ms, r } = blocksOf(false);
    expect(at(ms).elements[0].y).toBeGreaterThan(at(sq).elements[0].y);
    expect(at(r).elements[0].y).toBeGreaterThan(at(ms).elements[0].y);
  });
});
