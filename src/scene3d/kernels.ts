import { BoxGeometry, EdgesGeometry, LineBasicMaterial, LineSegments, Mesh, MeshBasicMaterial, Vector3 } from 'three';
import { realSizeOf } from '../autograd/graph';
import type { ThemeColors } from '../theme';
import type { SceneController } from './controller';
import type { Label } from './labels';
import type { ChipBox, KernelRegion, LevelGrid } from './levels';
import { segmentDistance } from './edgelines';
import { LineBatch, type Polyline } from './lines';
import { copyName } from '../steps';
import { busyCopies, copyIndices, currentGrid, sameCopy } from './slots';

export const STACK_DX = 0.5;
export const STACK_DZ = 0.9;
export const STACK_MAX = 15;
export const SHARED_TEXT = 'shared memory';
export const REGISTER_TEXT = '寄存器';
export const OUTLINE_HOVER_DISTANCE = 6;
export const COPY_LABEL_STEP_PX = 18;
export const COPY_LABEL_TOP_PX = 72;
const TOP_SEARCH_STEPS = 24;

const SHARED_OPACITY = 0.1;
const REGISTER_OPACITY = 0.05;
const CHIP_EDGE_OPACITY = 0.3;
const FRAME_OPACITY = 0.7;
const OUTLINE_OPACITY = 0.45;

export interface Outline {
  grid: LevelGrid;
  index: number[];
  busy: boolean;
  points: number[];
}

export function stackedCopies(grid: LevelGrid, front: readonly number[], expand: readonly number[]): number[][] {
  const ownExpand = (index: readonly number[]) => sameCopy(index.filter((_, d) => grid.arrangement[d] === 'expand'), expand);
  return copyIndices(grid.counts).filter((index) => ownExpand(index) && !sameCopy(index, front));
}

export function frontIn(grid: LevelGrid, front: readonly number[], expand: readonly number[]): number[] {
  let k = 0;
  return grid.arrangement.map((arrangement, d) => (arrangement === 'expand' ? expand[k++] : front[d]));
}

export function copyTotal(grid: LevelGrid, sizes: Readonly<Record<string, number>> | undefined, shapes: readonly (readonly number[])[]): { drawn: number; real: number | undefined } {
  const drawn = grid.counts.reduce((p, n) => p * n, 1);
  if (sizes === undefined) return { drawn, real: undefined };
  const real = grid.real.map((size) => realSizeOf(size, sizes, shapes));
  return { drawn, real: real.some((n) => n === undefined) ? undefined : (real as number[]).reduce((p, n) => p * n, 1) };
}

export function faceRect(box: ChipBox): number[] {
  const [x0, x1] = box.x;
  const [y0, y1] = box.y;
  const z = box.z[1];
  return [x0, y0, z, x1, y0, z, x1, y1, z, x0, y1, z, x0, y0, z];
}

export function stackEdge(box: ChipBox, j: number): number[] {
  const [x0, x1] = box.x;
  const [y0, y1] = box.y;
  const dx = j * STACK_DX;
  const z = box.z[1] - j * STACK_DZ;
  return [x0 + dx, y1, z, x1 + dx, y1, z, x1 + dx, y0, z];
}

interface Rung {
  label: Label;
  frame: ChipBox;
  x: number;
  z: number;
  j: number;
}

interface ChipMesh {
  mesh: Mesh;
  material: MeshBasicMaterial;
  edgeMaterial: LineBasicMaterial;
  label: Label;
}

interface RegionView {
  shared: ChipMesh;
  registers: ChipMesh;
  frameLabel: Label;
  total: Label;
  more: Label;
  copies: Label[];
}

const projected = new Vector3();

export class KernelRegions {
  private readonly geometry = new BoxGeometry(1, 1, 1);
  private readonly edges = new EdgesGeometry(this.geometry);
  private readonly busy: LineBatch;
  private readonly idle: LineBatch;
  private readonly views: RegionView[] = [];
  private totals = new WeakMap<LevelGrid, { drawn: number; real: number | undefined }>();
  private screen: { outline: Outline; points: number[] }[] = [];
  private screenVersion = -1;
  private readonly labelled = new Map<Label, Outline>();
  private ladder: Rung[] = [];
  outlines: Outline[] = [];

  constructor(private readonly c: SceneController) {
    this.busy = new LineBatch(c.theme.pos);
    this.idle = new LineBatch(c.theme.lineStrong);
    c.scene.add(this.busy.object, this.idle.object);
  }

  private chip(text: string, opacity: number): ChipMesh {
    const material = new MeshBasicMaterial({ transparent: true, opacity, depthWrite: false });
    const edgeMaterial = new LineBasicMaterial({ transparent: true, opacity: CHIP_EDGE_OPACITY });
    const mesh = new Mesh(this.geometry, material);
    mesh.add(new LineSegments(this.edges, edgeMaterial));
    this.c.scene.add(mesh);
    const label = this.c.labels.add('tag');
    label.anchor = 'above-right';
    label.setText(text);
    return { mesh, material, edgeMaterial, label };
  }

  private view(i: number): RegionView {
    while (this.views.length <= i) {
      const frameLabel = this.c.labels.add('copy');
      frameLabel.anchor = 'left-middle';
      const total = this.c.labels.add('tag');
      total.anchor = 'above-right';
      const more = this.c.labels.add('tag');
      more.anchor = 'left-middle';
      this.views.push({ shared: this.chip(SHARED_TEXT, SHARED_OPACITY), registers: this.chip(REGISTER_TEXT, REGISTER_OPACITY), frameLabel, total, more, copies: [] });
    }
    return this.views[i];
  }

  private copyLabel(view: RegionView, k: number): Label {
    while (view.copies.length <= k) {
      const label = this.c.labels.add('copy');
      label.anchor = 'left-middle';
      label.el.addEventListener('click', () => {
        const outline = this.labelled.get(label);
        if (outline !== undefined) this.setFront(outline);
      });
      view.copies.push(label);
    }
    return view.copies[k];
  }

  private placeChip(chip: ChipMesh, box: ChipBox | undefined, show: boolean): void {
    chip.mesh.visible = box !== undefined;
    chip.label.visible = box !== undefined && show;
    if (box === undefined) return;
    const theme = this.c.theme;
    const [x0, x1] = box.x;
    const [y0, y1] = box.y;
    const [z0, z1] = box.z;
    chip.mesh.scale.set(x1 - x0, y1 - y0, z1 - z0);
    chip.mesh.position.set((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2);
    chip.material.color.setRGB(...theme.muted);
    chip.edgeMaterial.color.setRGB(...theme.ink);
    chip.label.pos = [x0, y0, z1];
  }

  private total(grid: LevelGrid): { drawn: number; real: number | undefined } {
    const known = this.totals.get(grid);
    if (known !== undefined) return known;
    const g = this.c.g;
    const found = g.real === undefined ? copyTotal(grid, undefined, []) : copyTotal(grid, g.realSizes({}), g.realShapes({}));
    this.totals.set(grid, found);
    return found;
  }

  update(): void {
    const c = this.c;
    const regions: readonly KernelRegion[] = c.levelLayout?.kernels ?? [];
    const named = !c.lod;
    const busyLines: Polyline[] = [];
    const idleLines: Polyline[] = [];
    const outlines: Outline[] = [];
    this.labelled.clear();
    const ladder: Rung[] = [];
    regions.forEach((region, i) => {
      const view = this.view(i);
      this.placeChip(view.shared, region.shared, named);
      this.placeChip(view.registers, region.registers, named);
      const grid = currentGrid(region.grids, c.state);
      const stacked = grid.arrangement.some((a, d) => a === 'stack' && grid.counts[d] > 1);
      const front = frontIn(grid, c.frontOf(grid), region.expand);
      const busy = busyCopies(grid, (n) => c.state.active(n));
      const isBusy = (index: readonly number[]) => busy.some((b) => sameCopy(b, index));
      const frame = region.frame;
      (isBusy(front) ? busyLines : idleLines).push({ points: faceRect(frame), opacity: FRAME_OPACITY });
      const [x0, x1] = frame.x;
      const z1 = frame.z[1];
      view.frameLabel.visible = named && grid.names.length > 0;
      ladder.push({ label: view.frameLabel, frame, x: x1, z: z1, j: 0 });
      view.frameLabel.setText(copyName(grid, front));
      view.frameLabel.setFlag('vf-on', isBusy(front));
      const total = this.total(grid);
      view.total.visible = named && stacked;
      ladder.push({ label: view.total, frame, x: x0, z: z1, j: 0 });
      view.total.setText(`共 ${total.drawn} 个副本同时执行${total.real === undefined ? '' : `（真实尺寸 ${total.real} 个）`}`);
      const others = stacked ? stackedCopies(grid, front, region.expand) : [];
      const drawn = others.slice(0, STACK_MAX);
      drawn.forEach((index, k) => {
        const j = k + 1;
        const points = stackEdge(frame, j);
        const on = isBusy(index);
        (on ? busyLines : idleLines).push({ points, opacity: OUTLINE_OPACITY });
        const label = this.copyLabel(view, k);
        label.visible = named;
        ladder.push({ label, frame, x: x1 + j * STACK_DX, z: z1 - j * STACK_DZ, j });
        label.setText(copyName(grid, index));
        label.setFlag('vf-on', on);
        const outline = { grid, index, busy: on, points };
        outlines.push(outline);
        this.labelled.set(label, outline);
      });
      view.copies.slice(drawn.length).forEach((label) => (label.visible = false));
      const rest = others.length - drawn.length;
      view.more.visible = named && rest > 0;
      if (rest > 0) {
        const j = drawn.length;
        ladder.push({ label: view.more, frame, x: x1 + j * STACK_DX, z: z1 - j * STACK_DZ, j: j + 1 });
        view.more.setText(`还有 ${rest} 个`);
      }
    });
    for (const view of this.views.slice(regions.length)) {
      for (const chip of [view.shared, view.registers]) {
        chip.mesh.visible = false;
        chip.label.visible = false;
      }
      for (const label of [view.frameLabel, view.total, view.more, ...view.copies]) label.visible = false;
    }
    this.outlines = outlines;
    this.ladder = ladder;
    this.place();
    this.screenVersion = -1;
    this.busy.setPolylines(busyLines);
    this.idle.setPolylines(idleLines);
  }

  place(): void {
    const camera = this.c.camera;
    const perUnit = (this.c.height / (camera.top - camera.bottom)) * camera.zoom;
    const step = perUnit > 0 ? COPY_LABEL_STEP_PX / perUnit : 0;
    const tops = new Map<ChipBox, number>();
    for (const { label, frame, x, z, j } of this.ladder) {
      const top = tops.get(frame) ?? tops.set(frame, this.visibleTop(frame, x, z)).get(frame)!;
      label.pos = [x, Math.max(frame.y[0], top - j * step), z];
    }
  }

  private visibleTop(frame: ChipBox, x: number, z: number): number {
    const c = this.c;
    const screenY = (y: number) => {
      projected.set(x, y, z).project(c.camera);
      return ((1 - projected.y) / 2) * c.height;
    };
    let [lo, hi] = frame.y;
    if (screenY(hi) >= COPY_LABEL_TOP_PX) return hi;
    if (screenY(lo) < COPY_LABEL_TOP_PX) return lo;
    for (let k = 0; k < TOP_SEARCH_STEPS; k++) {
      const mid = (lo + hi) / 2;
      if (screenY(mid) >= COPY_LABEL_TOP_PX) lo = mid;
      else hi = mid;
    }
    return lo;
  }

  private setFront(outline: Outline): void {
    this.c.steps?.setFront(outline.grid.module, outline.index);
  }

  outlineAt(x: number, y: number): Outline | undefined {
    const c = this.c;
    if (this.screenVersion !== c.screenVersion) {
      this.screen = this.outlines.map((outline) => {
        const points: number[] = [];
        for (let i = 0; i < outline.points.length; i += 3) {
          projected.set(outline.points[i], outline.points[i + 1], outline.points[i + 2]).project(c.camera);
          points.push(((projected.x + 1) / 2) * c.width, ((1 - projected.y) / 2) * c.height);
        }
        return { outline, points };
      });
      this.screenVersion = c.screenVersion;
    }
    let best: Outline | undefined;
    let bestDistance = OUTLINE_HOVER_DISTANCE;
    for (const { outline, points } of this.screen) {
      for (let i = 0; i + 3 < points.length; i += 2) {
        const distance = segmentDistance(x, y, points[i], points[i + 1], points[i + 2], points[i + 3]);
        if (distance < bestDistance) {
          bestDistance = distance;
          best = outline;
        }
      }
    }
    return best;
  }

  click(x: number, y: number): boolean {
    const outline = this.outlineAt(x, y);
    if (outline === undefined) return false;
    this.setFront(outline);
    return true;
  }

  setTheme(theme: ThemeColors): void {
    this.busy.setColor(theme.pos);
    this.idle.setColor(theme.lineStrong);
  }

  dispose(): void {
    this.busy.dispose();
    this.idle.dispose();
    this.geometry.dispose();
    this.edges.dispose();
    for (const view of this.views) {
      for (const chip of [view.shared, view.registers]) {
        chip.material.dispose();
        chip.edgeMaterial.dispose();
      }
    }
  }
}
