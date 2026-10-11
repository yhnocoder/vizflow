import { describe, expect, it } from 'vitest';
import { type OverlapItem, overlapSchedule } from '../../src/sim';

const at = (items: OverlapItem[], phase: string, batch: number, layer = 0) =>
  items.find((item) => item.phase === phase && item.batch === batch && item.layer === layer)!;

describe('two micro-batch overlap', () => {
  it('hides communication behind the other batch when compute is longer', () => {
    const items = overlapSchedule({ attention: 5, dispatch: 4, moe: 6, combine: 3 }, 2);
    expect(items.map((item) => [item.phase, item.batch, item.start, item.end])).toEqual([
      ['attention', 0, 0, 5],
      ['dispatch', 0, 5, 9],
      ['attention', 1, 5, 10],
      ['dispatch', 1, 10, 14],
      ['moe', 0, 10, 16],
      ['combine', 0, 16, 19],
      ['moe', 1, 16, 22],
      ['combine', 1, 22, 25],
      ['attention', 0, 22, 27],
      ['dispatch', 0, 27, 31],
      ['attention', 1, 27, 32],
      ['dispatch', 1, 32, 36],
      ['moe', 0, 32, 38],
      ['combine', 0, 38, 41],
      ['moe', 1, 38, 44],
      ['combine', 1, 44, 47],
    ]);
  });

  it('keeps each stream in order and each batch in dependency order', () => {
    const items = overlapSchedule({ attention: 2, dispatch: 7, moe: 3, combine: 5 }, 3);
    for (const stream of ['compute', 'comm']) {
      const own = items.filter((item) => item.stream === stream);
      own.slice(1).forEach((item, i) => expect(item.start).toBeGreaterThanOrEqual(own[i].end));
    }
    for (const batch of [0, 1]) {
      const own = items.filter((item) => item.batch === batch);
      own.slice(1).forEach((item, i) => expect(item.start).toBeGreaterThanOrEqual(own[i].end));
    }
  });

  it('leaves the compute stream idle when communication is longer', () => {
    const items = overlapSchedule({ attention: 2, dispatch: 7, moe: 3, combine: 5 }, 1);
    expect(at(items, 'moe', 0).start).toBe(at(items, 'dispatch', 0).end);
    expect(at(items, 'moe', 0).start).toBeGreaterThan(at(items, 'attention', 1).end);
  });
});
