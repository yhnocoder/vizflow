import { expect } from 'vitest';
import type { CompiledPage } from '../../src/dsl';

const PREFIX = 'moe_layer.mlp';
const NO_GRAD = ['scores_for_choice', 'grouped', 'top2', 'group_scores', 'group_mask', 'score_mask', 'masked_scores', 'idx'];
const GROUP_SIZE = 3;

export function expectRouting(page: CompiledPage, hf: { topk_indices: number[][]; experts: Record<string, number[]> }) {
  const node = (name: string) => page.nodes.get(`${PREFIX}.${name}`)!;
  const idx = node('idx').value.toNested() as number[][];
  idx.forEach((row, t) => {
    expect([...row].sort(), `row ${t}`).toEqual([...hf.topk_indices[t]].sort());
    expect(new Set(row.map((e) => Math.floor(e / GROUP_SIZE))).size, `row ${t} uses one group`).toBe(1);
  });
  const expertIdx = Array.from(node('experts.expert_idx').value.data);
  const tokenIdx = Array.from(node('experts.token_idx').value.data);
  const tokensOf: Record<string, number[]> = {};
  expertIdx.forEach((e, r) => (tokensOf[String(e)] ??= []).push(tokenIdx[r]));
  for (const tokens of [...Object.values(tokensOf), ...Object.values(hf.experts)]) tokens.sort();
  expect(tokensOf).toEqual(hf.experts);
  for (const name of NO_GRAD) expect(node(name).grad, name).toBeNull();
}
