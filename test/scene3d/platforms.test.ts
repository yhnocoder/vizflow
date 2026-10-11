// @vitest-environment jsdom
import { beforeAll, describe, expect, it } from 'vitest';
import type { CommTransfer } from '../../src/autograd';
import { type CompiledPage, loadPage } from '../../src/dsl';
import { PLACEHOLDER_WIDTH, type PlatformBox, arrangePlatforms, commCalls, localPiece, platformEdges, platformSlots, slotOf, slotText } from '../../src/scene3d/platforms';
import { examplesUrl, readFileUrl } from '../dsl/load';

const SPAN: PlatformBox = { x0: 0, x1: 6, y0: 0, y1: 3, z0: 0, z1: 2 };
const FRONT: [number, number, number] = [0, 0.3, 1];

function transfer(src: number, dst: number, bytes: number, link: CommTransfer['link'], hop = 0): CommTransfer {
  return { src, dst, bytes, link, hop };
}

describe('platformSlots', () => {
  it('draws every rank when the axis has at most four', () => {
    const slots = platformSlots({ size: 4, nodes: 2 });
    expect(slots.map((s) => (s.kind === 'rank' ? [s.rank, s.device.node, s.device.gpu] : s.kind))).toEqual([
      [0, 0, 0],
      [1, 0, 1],
      [2, 1, 0],
      [3, 1, 1],
    ]);
  });

  it('draws rank 0, rank 1, one placeholder and the last rank above four', () => {
    const slots = platformSlots({ size: 8, nodes: 2 });
    expect(slots.map(slotText)).toEqual(['rank 0', 'rank 1', 'rank 2 … rank 6', 'rank 7']);
    expect([0, 1, 2, 4, 6, 7].map((r) => slotOf(slots, r))).toEqual([0, 1, 2, 2, 2, 3]);
  });
});

describe('arrangePlatforms', () => {
  it('puts ranks of one node in one row and nodes in separate rows', () => {
    const topology = { size: 4, nodes: 2 };
    const { boxes, shifts, rows } = arrangePlatforms(topology, platformSlots(topology), SPAN, FRONT);
    expect(rows.map((r) => r.node)).toEqual([0, 1]);
    expect(shifts[0]![1]).toBe(shifts[1]![1]);
    expect(shifts[2]![1]).toBe(shifts[3]![1]);
    expect(shifts[0]![1]).toBeGreaterThan(shifts[2]![1]);
    expect(boxes[1].x0).toBeGreaterThan(boxes[0].x1);
    expect(boxes[3].x0).toBeGreaterThan(boxes[2].x1);
    expect(boxes[0].x0).toBe(boxes[2].x0);
  });

  it('gives the placeholder a fixed width and no block shift', () => {
    const topology = { size: 8, nodes: 1 };
    const { boxes, shifts } = arrangePlatforms(topology, platformSlots(topology), SPAN, FRONT);
    expect(shifts[2]).toBeUndefined();
    expect(boxes[2].x1 - boxes[2].x0).toBe(PLACEHOLDER_WIDTH);
    expect(boxes[3].x0).toBeGreaterThan(boxes[2].x1);
    expect(new Set(shifts.filter((s) => s !== undefined).map((s) => s[1])).size).toBe(1);
  });
});

describe('platformEdges', () => {
  it('drops local and same-slot transfers and merges the rest by direction and link', () => {
    const slots = platformSlots({ size: 8, nodes: 2 });
    const edges = platformEdges(
      [transfer(0, 0, 10, 'local'), transfer(2, 3, 10, 'nvlink'), transfer(0, 4, 5, 'rdma', 0), transfer(0, 5, 7, 'rdma', 0), transfer(1, 4, 3, 'nvlink', 1), transfer(4, 1, 2, 'nvlink', 1), transfer(0, 1, 4, 'nvlink', 0), transfer(0, 1, 6, 'nvlink', 1)],
      slots,
    );
    const byKey = new Map(edges.map((e) => [`${e.from}>${e.to}>${e.link}`, e]));
    expect([...byKey.keys()].sort()).toEqual(['0>1>nvlink', '0>2>rdma', '1>2>nvlink', '2>1>nvlink']);
    expect(byKey.get('0>2>rdma')).toMatchObject({ bytes: 12, messages: 2, pairs: [[0, 4], [0, 5]] });
    expect(byKey.get('0>1>nvlink')!.hops).toEqual([
      { hop: 0, messages: 1, bytes: 4 },
      { hop: 1, messages: 1, bytes: 6 },
    ]);
  });

  it('keeps the ring phases of an all-reduce', () => {
    const slots = platformSlots({ size: 2, nodes: 1 });
    const edges = platformEdges([{ ...transfer(0, 1, 1, 'nvlink'), phase: 'reduce_scatter' }, { ...transfer(0, 1, 1, 'nvlink'), phase: 'all_gather' }], slots);
    expect(edges).toHaveLength(1);
    expect(edges[0].phases).toEqual(['reduce_scatter', 'all_gather']);
  });
});

describe('platform pieces of the EP page', () => {
  let ep: CompiledPage;
  let tp: CompiledPage;

  beforeAll(async () => {
    ep = await loadPage(new URL('deepseek_v3-ep.yaml', examplesUrl).href, { read: readFileUrl });
    tp = await loadPage(new URL('qwen3-tp-dsl.yaml', examplesUrl).href, { read: readFileUrl });
  });

  it('sends the hidden states and the per-row weights of one selection in one dispatch call', () => {
    const calls = commCalls(ep.graph.nodes);
    const dispatch = calls.filter((c) => c.op === 'dispatch');
    expect(dispatch).toHaveLength(1);
    expect([dispatch[0].node, ...dispatch[0].carried].map((n) => n.name).sort()).toEqual(['mlp.P', 'mlp.R']);
    expect(calls.filter((c) => c.op === 'combine')).toHaveLength(1);
  });

  it('gives each rank exactly the rows it owns of a sharded or routed tensor', () => {
    const p = ep.placements!;
    for (const name of ['x', 'mlp.R', 'mlp.Wg']) {
      const node = ep.nodes.get(name)!;
      const owners = Array.from({ length: node.value.size }, () => 0);
      for (let rank = 0; rank < p.mesh.size; rank++) {
        const piece = localPiece(p, node, rank, 'forward', true);
        expect(piece.partial).toBe(false);
        piece.held!.forEach((h, i) => (owners[i] += h ? 1 : 0));
      }
      expect(owners.every((n) => n === 1), name).toBe(true);
    }
  });

  it('shows the per-rank partial sums of a partial tensor, which add up to the full value', () => {
    const p = tp.placements!;
    const node = tp.graph.nodes.find((n) => p.of(n).kind === 'partial')!;
    expect(node).toBeDefined();
    const pieces = Array.from({ length: p.mesh.size }, (_, rank) => localPiece(p, node, rank, 'forward', true));
    expect(pieces.every((piece) => piece.partial && piece.held === undefined)).toBe(true);
    for (let i = 0; i < node.value.size; i++) expect(pieces.reduce((s, piece) => s + piece.values[i], 0)).toBeCloseTo(node.value.data[i], 6);
  });
});
