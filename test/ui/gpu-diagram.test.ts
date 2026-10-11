// @vitest-environment jsdom
import { afterEach, describe, expect, it } from 'vitest';
import { dispatchPlan, expandedLayout, pipeline, schedule, type EpTopology, type ScheduleOptions } from '../../src/sim';
import { epTopology, expandedBadges, mountGpuDiagram, pipelineState, scheduleTiles, transferState, type GpuDiagram } from '../../src/ui';

const mounted: GpuDiagram[] = [];

afterEach(() => {
  mounted.splice(0).forEach((m) => m.dispose());
  document.body.replaceChildren();
});

const EP: EpTopology = { numScaleoutRanks: 2, numScaleupRanks: 4, numExperts: 16 };
const topkIdx = Array.from({ length: 8 }, (_, rank) => (rank === 1 ? [[2, 6, 10, 14], [3, 8, 9, 15]] : []));
const topology = epTopology(EP, { smsPerGpu: 4, nic: true });

function mount(props: Parameters<typeof mountGpuDiagram>[1]): { el: HTMLElement; diagram: GpuDiagram } {
  const el = document.createElement('div');
  document.body.append(el);
  const diagram = mountGpuDiagram(el, props);
  mounted.push(diagram);
  return { el, diagram };
}

describe('mountGpuDiagram', () => {
  it('colors highlighted units, mutes the rest and draws flows and packets', () => {
    const { el, diagram } = mount({
      topology,
      level: 'node',
      highlight: { 'gpu:1': 'rdma', ib: 'rdma' },
      flows: [{ from: 'gpu:1', to: 'gpu:5', kind: 'rdma', label: '2 个 token' }],
      packets: [{ flow: 0, color: 0 }, { flow: 0, delay: 200 }],
      badges: { 'gpu:1': ['t0', { text: 't1', chips: [1] }] },
    });
    const unit = (id: string) => el.querySelector(`[data-unit="${id}"]`)!;
    expect(unit('gpu:1').classList.contains('vf-gpu-rdma')).toBe(true);
    expect(unit('gpu:0').classList.contains('vf-gpu-muted')).toBe(true);
    expect(unit('node:0').classList.contains('vf-gpu-frame')).toBe(true);
    expect(el.querySelectorAll('.vf-gpu-flow')).toHaveLength(1);
    expect(el.querySelector('.vf-gpu-flowlabel')!.textContent).toBe('2 个 token');
    expect([...el.querySelectorAll('.vf-gpu-packet')].map((p) => p.getAttribute('class'))).toEqual(['vf-gpu-packet vf-gpu-s0', 'vf-gpu-packet vf-gpu-rdma']);
    expect([...unit('gpu:1').querySelectorAll('.vf-gpu-line')].map((t) => t.textContent)).toEqual(['t0', 't1']);
    expect(unit('gpu:1').querySelectorAll('.vf-gpu-chip')).toHaveLength(1);

    diagram.set({ topology, level: 'sm', sm: { numStages: 2 } });
    expect(el.querySelectorAll('[data-unit^="smem:"]')).toHaveLength(2);
    expect(el.querySelector('.vf-gpu-muted')).toBeNull();
  });

  it('rejects highlights and badges that name units of another level', () => {
    expect(() => mount({ topology, level: 'gpu', highlight: { 'gpu:0': 'compute' } })).toThrow("highlight names unknown unit 'gpu:0' at level gpu");
    expect(() => mount({ topology, level: 'node', badges: { tma: ['x'] } })).toThrow("badges names unknown unit 'tma' at level node");
  });

  it('rejects packets that name a missing flow', () => {
    expect(() => mount({ topology, level: 'node', packets: [{ flow: 1 }] })).toThrow('packet 0 names flow 1, but there are 0 flows');
  });
});

describe('diagram state from the simulators', () => {
  it('turns a DeepEP dispatch plan into one flow per GPU pair and one packet per transfer', () => {
    const plan = dispatchPlan(EP, topkIdx, { mode: 'hybrid', hidden: 256, dtype: 'fp8' });
    const state = transferState(plan);
    const moving = plan.filter((t) => t.via !== 'local');
    expect(state.packets).toHaveLength(moving.length);
    expect(state.flows.find((f) => f.kind === 'rdma')).toEqual({ from: 'gpu:1', to: 'gpu:5', kind: 'rdma' });
    expect(new Set(state.flows.map((f) => `${f.from}>${f.to}`)).size).toBe(state.flows.length);
    expect(state.highlight['gpu:5']).toBeDefined();
  });

  it('lists received rows per expert with padding', () => {
    const badges = expandedBadges(expandedLayout(EP, topkIdx, 2));
    expect(badges['gpu:1']).toEqual([{ text: 'e2 +1 行补零', chips: [0] }, { text: 'e3 +1 行补零', chips: [1] }]);
    expect(badges['gpu:4']).toEqual([{ text: 'e8 +1 行补零', chips: [1] }, { text: 'e9 +1 行补零', chips: [1] }]);
  });

  it('places each scheduled tile at its m and n block with its SM and wave', () => {
    const opts: ScheduleOptions = { type: 'normal', m: 256, n: 512, k: 256, blockM: 128, blockN: 128, blockK: 128, numSms: 4, numMulticast: 1, multicastOnA: false };
    const tiles = scheduleTiles(opts, schedule(opts), 0);
    expect([tiles.rows, tiles.cols]).toEqual([2, 4]);
    const owners = Array.from({ length: 2 }, (_, r) => Array.from({ length: 4 }, (_, c) => tiles.owner(r, c)));
    expect(owners.flat().filter((o) => o?.wave === 0).map((o) => o!.sm).sort()).toEqual([0, 1, 2, 3]);
    expect(owners.flat().every((o) => o !== null)).toBe(true);
  });

  it('highlights the SM units busy at a pipeline time', () => {
    const opts: ScheduleOptions = { type: 'normal', m: 128, n: 128, k: 512, blockM: 128, blockN: 128, blockK: 128, numSms: 1, numMulticast: 1, multicastOnA: false };
    const result = pipeline({ schedule: opts, sm: 0, numStages: 2 });
    const tma = result.events.find((e) => e.row === 'tma')!;
    const loading = pipelineState(result, 2, (tma.start + tma.end) / 2);
    expect(loading.highlight).toMatchObject({ tma: 'copy', hbm: 'copy', 'smem:stage0': 'copy' });
    expect(loading.flows).toContainEqual({ from: 'hbm', to: 'smem:stage0', via: ['tma'], kind: 'copy' });
    const wg = result.events.find((e) => e.row === 'wg1')!;
    const computing = pipelineState(result, 2, (wg.start + wg.end) / 2);
    expect(computing.highlight['wg:1']).toBe('compute');
    expect(computing.badges[`smem:stage${wg.stage}`]).toEqual([`tile 0 k-block ${wg.kBlock}`]);
  });
});
