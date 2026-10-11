import type { EpTopology, ExpandedRow, PipelineResult, ScheduleOptions, Tile, Transfer } from '../sim';
import type { GpuBadgeLine, GpuFlow, GpuKind, GpuPacket, GpuTiles, GpuTopology } from './gpu-layout';

export interface GpuState {
  highlight: Record<string, GpuKind>;
  flows: GpuFlow[];
  packets: GpuPacket[];
  badges: Record<string, GpuBadgeLine[]>;
}

const ceilDiv = (a: number, b: number) => Math.floor((a + b - 1) / b);

export function epTopology(topo: EpTopology, extra: Omit<GpuTopology, 'nodes' | 'gpusPerNode'>): GpuTopology {
  return { ...extra, nodes: topo.numScaleoutRanks, gpusPerNode: topo.numScaleupRanks };
}

export function transferState(transfers: readonly Transfer[], colorOf: (t: Transfer) => number = (t) => t.token, spacing = 200): GpuState {
  const flows: GpuFlow[] = [];
  const packets: GpuPacket[] = [];
  const highlight: Record<string, GpuKind> = {};
  const index = new Map<string, number>();
  const counts: number[] = [];
  for (const t of transfers) {
    if (t.via === 'local') continue;
    const kind: GpuKind = t.via === 'rdma' ? 'rdma' : 'nvlink';
    const key = `${t.srcRank}>${t.dstRank}`;
    let flow = index.get(key);
    if (flow === undefined) {
      flow = flows.length;
      index.set(key, flow);
      flows.push({ from: `gpu:${t.srcRank}`, to: `gpu:${t.dstRank}`, kind });
      counts.push(0);
    }
    packets.push({ flow, color: colorOf(t), delay: counts[flow] * spacing });
    counts[flow] += 1;
    highlight[`gpu:${t.srcRank}`] ??= kind;
    highlight[`gpu:${t.dstRank}`] = kind;
  }
  return { highlight, flows, packets, badges: {} };
}

export function expandedBadges(layout: readonly ExpandedRow[][], colorOf: (row: ExpandedRow & { kind: 'token' }) => number = (row) => row.token): Record<string, GpuBadgeLine[]> {
  const badges: Record<string, GpuBadgeLine[]> = {};
  layout.forEach((rows, rank) => {
    const experts = [...new Set(rows.map((row) => row.expert))];
    badges[`gpu:${rank}`] = experts.map((expert) => {
      const own = rows.filter((row) => row.expert === expert);
      const tokens = own.filter((row): row is ExpandedRow & { kind: 'token' } => row.kind === 'token');
      const padding = own.length - tokens.length;
      return { text: `e${expert}${padding > 0 ? ` +${padding} 行补零` : ''}`, chips: tokens.map(colorOf) };
    });
  });
  return badges;
}

export function scheduleTiles(opts: ScheduleOptions, schedule: readonly Tile[][], wave?: number): GpuTiles {
  const owners = new Map<string, Tile>();
  for (const tiles of schedule) for (const tile of tiles) owners.set(`${tile.mBlock},${tile.nBlock}`, tile);
  return {
    rows: ceilDiv(opts.m, opts.blockM),
    cols: ceilDiv(opts.n, opts.blockN),
    owner: (row, col) => {
      const tile = owners.get(`${row},${col}`);
      return tile === undefined ? null : { sm: tile.sm, wave: tile.iter };
    },
    wave,
  };
}

export function pipelineState(result: PipelineResult, numStages: number, time: number): GpuState {
  const highlight: Record<string, GpuKind> = {};
  const flows: GpuFlow[] = [];
  const badges: Record<string, GpuBadgeLine[]> = {};
  const held = new Map<number, { tile: number; kBlock: number }>();
  for (const event of result.events) {
    if (event.row !== 'tma' || event.end > time) continue;
    const release = result.releases.find((r) => r.tile === event.tile && r.kBlock === event.kBlock);
    if (release === undefined || release.time > time) held.set(event.stage, { tile: event.tile, kBlock: event.kBlock });
  }
  for (let stage = 0; stage < numStages; stage += 1) {
    const content = held.get(stage);
    badges[`smem:stage${stage}`] = content === undefined ? ['空'] : [`tile ${content.tile} k-block ${content.kBlock}`];
  }
  for (const event of result.events) {
    if (event.start > time || time >= event.end) continue;
    const stage = `smem:stage${event.stage}`;
    if (event.row === 'tma') {
      highlight.hbm = 'copy';
      highlight.tma = 'copy';
      highlight[stage] = 'copy';
      badges[stage] = [`tile ${event.tile} k-block ${event.kBlock}`, '正在装入'];
      flows.push({ from: 'hbm', to: stage, via: ['tma'], kind: 'copy' });
    } else if (event.row === 'promote') {
      highlight['cuda-core'] = 'reduce';
      flows.push({ from: `wg:${event.wg}`, to: 'cuda-core', kind: 'reduce' });
    } else if (event.row === 'epilogue') {
      highlight.epilogue = 'copy';
      highlight.hbm = 'copy';
      flows.push({ from: 'epilogue', to: 'hbm', kind: 'copy' });
    } else {
      highlight[`wg:${event.wg}`] = 'compute';
      highlight[stage] ??= 'compute';
      flows.push({ from: stage, to: `wg:${event.wg}`, kind: 'compute' });
    }
  }
  return { highlight, flows, packets: [], badges };
}
