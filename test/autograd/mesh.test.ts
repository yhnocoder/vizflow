import { describe, expect, it } from 'vitest';
import { linkKind, rankDevice, ringAllReduce } from '../../src/core';

describe('mesh topology', () => {
  it('numbers ranks node by node and tells NVLink from RDMA', () => {
    const topology = { size: 4, nodes: 2 };
    expect([0, 1, 2, 3].map((r) => rankDevice(topology, r))).toEqual([{ node: 0, gpu: 0 }, { node: 0, gpu: 1 }, { node: 1, gpu: 0 }, { node: 1, gpu: 1 }]);
    expect(linkKind(topology, 0, 1)).toBe('nvlink');
    expect(linkKind(topology, 1, 2)).toBe('rdma');
    expect(linkKind(topology, 3, 3)).toBe('local');
  });

  it('lists a ring all_reduce as reduce_scatter and all_gather, each rank sending 2(n-1)/n of the message', () => {
    const transfers = ringAllReduce({ size: 4, nodes: 2 }, 1024);
    expect(transfers).toHaveLength(8);
    expect(transfers.filter((t) => t.src === 1).reduce((s, t) => s + t.bytes, 0)).toBe(1536);
    expect(transfers.filter((t) => t.phase === 'reduce_scatter').map((t) => [t.src, t.dst, t.link])).toEqual([[0, 1, 'nvlink'], [1, 2, 'rdma'], [2, 3, 'nvlink'], [3, 0, 'rdma']]);
  });
});
