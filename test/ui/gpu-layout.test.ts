import { describe, expect, it } from 'vitest';
import { GPU_NARROW_WIDTH, GPU_WIDE_WIDTH, gpuUnitIds, layoutGpuDiagram, pathLength, pointAt, type GpuDiagramProps, type GpuTopology, type Point } from '../../src/ui/gpu-layout';

const topology: GpuTopology = { nodes: 2, gpusPerNode: 8, smsPerGpu: 8, nic: true };

const inside = (p: Point, b: { x: number; y: number; w: number; h: number }) => p.x >= b.x && p.x <= b.x + b.w && p.y >= b.y && p.y <= b.y + b.h;

function orthogonal(points: Point[]): boolean {
  return points.slice(1).every((p, i) => p.x === points[i].x || p.y === points[i].y);
}

describe('gpuUnitIds', () => {
  it('lists nodes, GPUs, NVSwitches, one NIC per GPU and the IB switch at node level', () => {
    const ids = gpuUnitIds({ topology, level: 'node' });
    expect(ids.filter((id) => id.startsWith('node:'))).toEqual(['node:0', 'node:1']);
    expect(ids.filter((id) => id.startsWith('gpu:'))).toHaveLength(16);
    expect(ids.filter((id) => id.startsWith('nic:'))).toHaveLength(16);
    expect(ids.filter((id) => id.startsWith('nvswitch:'))).toEqual(['nvswitch:0', 'nvswitch:1']);
    expect(ids).toContain('ib');
  });

  it('omits NICs without the nic option and the IB switch with one node', () => {
    const ids = gpuUnitIds({ topology: { ...topology, nodes: 1, nic: false }, level: 'node' });
    expect(ids.some((id) => id.startsWith('nic:') || id === 'ib')).toBe(false);
  });

  it('lists SMs, L2 and HBM at gpu level', () => {
    expect(gpuUnitIds({ topology: { ...topology, smsPerGpu: 3 }, level: 'gpu' })).toEqual(['sm:0', 'sm:1', 'sm:2', 'l2', 'hbm']);
  });

  it('lists SM parts with the configured stages and warpgroups, and tmem only on sm100', () => {
    const sm90 = gpuUnitIds({ topology, level: 'sm', sm: { numStages: 3, mathWarpgroups: 1 } });
    expect(sm90.sort()).toEqual(['cuda-core', 'epilogue', 'hbm', 'regs', 'smem:stage0', 'smem:stage1', 'smem:stage2', 'tma', 'wg:0']);
    const sm100 = gpuUnitIds({ topology, level: 'sm', sm: { arch: 'sm100' } });
    expect(sm100).toContain('tmem');
    expect(sm100.filter((id) => id.startsWith('smem:'))).toHaveLength(4);
    expect(sm100.filter((id) => id.startsWith('wg:'))).toHaveLength(2);
  });
});

describe('flow routing', () => {
  const props: GpuDiagramProps = { topology, level: 'node' };

  it('routes GPUs in one node through their NVSwitch', () => {
    for (const width of [GPU_WIDE_WIDTH, GPU_NARROW_WIDTH]) {
      const layout = layoutGpuDiagram(props, width);
      const flow = { from: 'gpu:2', to: 'gpu:7', kind: 'nvlink' as const };
      expect(layout.units(flow)).toEqual(['gpu:2', 'nvswitch:0', 'gpu:7']);
      const points = layout.route(flow);
      const sw = layout.boxes.find((b) => b.id === 'nvswitch:0')!;
      expect(points.some((p) => inside(p, sw))).toBe(true);
      expect(orthogonal(points)).toBe(true);
    }
  });

  it('routes GPUs in different nodes through both NICs and the IB switch', () => {
    for (const width of [GPU_WIDE_WIDTH, GPU_NARROW_WIDTH]) {
      const layout = layoutGpuDiagram(props, width);
      const flow = { from: 'gpu:2', to: 'gpu:10', kind: 'rdma' as const };
      expect(layout.units(flow)).toEqual(['gpu:2', 'nic:2', 'ib', 'nic:10', 'gpu:10']);
      const points = layout.route(flow);
      for (const id of ['nic:2', 'ib', 'nic:10']) {
        const b = layout.boxes.find((x) => x.id === id)!;
        expect(points.some((p) => inside(p, b)), id).toBe(true);
      }
      expect(orthogonal(points)).toBe(true);
    }
  });

  it('routes across nodes straight to the IB switch without NICs', () => {
    const layout = layoutGpuDiagram({ topology: { ...topology, nic: false }, level: 'node' }, GPU_WIDE_WIDTH);
    expect(layout.units({ from: 'gpu:0', to: 'gpu:9', kind: 'rdma' })).toEqual(['gpu:0', 'ib', 'gpu:9']);
  });

  it('follows via stops', () => {
    const layout = layoutGpuDiagram(props, GPU_WIDE_WIDTH);
    expect(layout.units({ from: 'gpu:2', to: 'gpu:13', via: ['gpu:10'], kind: 'rdma' })).toEqual(['gpu:2', 'nic:2', 'ib', 'nic:10', 'gpu:10', 'nvswitch:1', 'gpu:13']);
  });

  it('routes an SM to HBM through L2 and HBM to a stage through TMA', () => {
    const gpu = layoutGpuDiagram({ topology, level: 'gpu' }, GPU_WIDE_WIDTH);
    expect(gpu.units({ from: 'sm:5', to: 'hbm', kind: 'copy' })).toEqual(['sm:5', 'l2', 'hbm']);
    const sm = layoutGpuDiagram({ topology, level: 'sm' }, GPU_WIDE_WIDTH);
    expect(sm.units({ from: 'hbm', to: 'smem:stage2', kind: 'copy' })).toEqual(['hbm', 'tma', 'smem:stage2']);
    expect(sm.units({ from: 'epilogue', to: 'hbm', kind: 'copy' })).toEqual(['epilogue', 'hbm']);
    expect(orthogonal(sm.route({ from: 'hbm', to: 'wg:1', kind: 'copy' }))).toBe(true);
  });

  it('rejects unknown units', () => {
    const layout = layoutGpuDiagram(props, GPU_WIDE_WIDTH);
    expect(() => layout.route({ from: 'gpu:2', to: 'gpu:16', kind: 'nvlink' })).toThrow("unknown unit 'gpu:16' at level node");
  });
});

describe('layout', () => {
  it('keeps every unit inside the view box at both widths', () => {
    const levels: GpuDiagramProps[] = [
      { topology, level: 'node', badges: { 'gpu:3': ['a', 'b', 'c'] } },
      { topology: { ...topology, smsPerGpu: 132 }, level: 'gpu', tiles: { rows: 4, cols: 8, owner: () => null } },
      { topology, level: 'sm', sm: { arch: 'sm100' } },
    ];
    for (const props of levels) {
      for (const width of [GPU_WIDE_WIDTH, GPU_NARROW_WIDTH]) {
        const layout = layoutGpuDiagram(props, width);
        for (const b of [...layout.frames, ...layout.boxes]) {
          expect(b.x, b.id).toBeGreaterThanOrEqual(0);
          expect(b.x + b.w, b.id).toBeLessThanOrEqual(width);
          expect(b.y + b.h, b.id).toBeLessThanOrEqual(layout.height);
        }
      }
    }
  });

  it('marks tiles of earlier waves done and the current wave current', () => {
    const layout = layoutGpuDiagram(
      { topology, level: 'gpu', tiles: { rows: 1, cols: 3, owner: (_, c) => (c === 2 ? null : { sm: c, wave: c }), wave: 1 } },
      GPU_WIDE_WIDTH,
    );
    expect(layout.tiles!.cells.map((c) => [c.state, c.lines])).toEqual([
      ['done', ['SM 0', '第 1 轮']],
      ['current', ['SM 1', '第 2 轮']],
      ['empty', []],
    ]);
  });
});

describe('pointAt', () => {
  it('walks a polyline by length', () => {
    const points = [{ x: 0, y: 0 }, { x: 10, y: 0 }, { x: 10, y: 30 }];
    expect(pathLength(points)).toBe(40);
    expect(pointAt(points, 0.5)).toEqual({ x: 10, y: 10 });
    expect(pointAt(points, 2)).toEqual({ x: 10, y: 30 });
    expect(pointAt(points, -1)).toEqual({ x: 0, y: 0 });
  });
});
