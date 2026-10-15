import { BoxGeometry, EdgesGeometry, LineBasicMaterial, LineSegments, Mesh, MeshBasicMaterial, Vector3 } from 'three';
import { realSizeOf } from '../autograd/graph';
import type { LinearRgb, ThemeColors } from '../theme';
import type { SceneController } from './controller';
import { sampleCurve } from './edgelines';
import type { Label } from './labels';
import { type Block, LINE_HEIGHT, type Vec3 } from './layout';
import type { CardBox, KernelGround, LevelGrid } from './levels';
import { segmentDistance } from './edgelines';
import { LineBatch, type Polyline } from './lines';
import { renderableFor } from './renderable';
import { copyName } from '../steps';
import { busyCopies, copyIndices, currentGrid, sameCopy } from './slots';

export const OUTLINE_STEP: Vec3 = [0.9, -0.25, -1.3];
export const OUTLINE_MAX = 15;
export const CARD_DROP = 0.1;
export const CARD_THICKNESS = 0.04;
export const HBM_TEXT = '显存 HBM';
export const CHIP_TEXT = '片上';
export const SHARED_TEXT = 'shared memory';
export const REGISTER_TEXT = '寄存器';
export const OUTLINE_HOVER_DISTANCE = 6;
export const LINK_OPACITY = 0.35;

const HBM_OPACITY = 0.12;
const CHIP_OPACITY = 0.1;
const PLATE_OPACITY = 0.08;
const CARD_EDGE_OPACITY = 0.6;
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

export function outlineRect(card: CardBox, y: number, j: number): number[] {
  const [dx, dy, dz] = OUTLINE_STEP.map((v) => v * j);
  const [x0, x1] = card.x;
  const [z0, z1] = card.z;
  const h = y + dy;
  return [x0 + dx, h, z0 + dz, x1 + dx, h, z0 + dz, x1 + dx, h, z1 + dz, x0 + dx, h, z1 + dz, x0 + dx, h, z0 + dz];
}

function linkPoints(original: Block, copy: Block): number[] {
  const centre = (block: Block): Vec3 => {
    const p = block.plate;
    if (p !== undefined) return [p.x, p.y + LINE_HEIGHT, p.z];
    const e = block.elements[0];
    return [e.x, e.y + LINE_HEIGHT, e.z];
  };
  return sampleCurve(centre(original), centre(copy), false);
}

interface Card {
  mesh: Mesh;
  material: MeshBasicMaterial;
  edgeMaterial: LineBasicMaterial;
  label: Label;
}

interface ChipView {
  card: Card;
  shared: Card;
  registers: Card;
  total: Label;
  more: Label;
  copies: Label[];
}

interface GroundView {
  hbm: Card;
  chips: ChipView[];
}

const projected = new Vector3();

export class KernelRegions {
  private readonly geometry = new BoxGeometry(1, 1, 1);
  private readonly edges = new EdgesGeometry(this.geometry);
  private readonly busy: LineBatch;
  private readonly idle: LineBatch;
  private readonly links: LineBatch;
  private readonly views: GroundView[] = [];
  private totals = new WeakMap<LevelGrid, { drawn: number; real: number | undefined }>();
  private screen: { outline: Outline; points: number[] }[] = [];
  private screenVersion = -1;
  private readonly labelled = new Map<Label, Outline>();
  outlines: Outline[] = [];

  constructor(private readonly c: SceneController) {
    this.busy = new LineBatch(c.theme.pos);
    this.idle = new LineBatch(c.theme.lineStrong);
    this.links = new LineBatch(c.theme.ink);
    c.scene.add(this.busy.object, this.idle.object, this.links.object);
  }

  private card(text: string, extraClass: string, anchor: Label['anchor']): Card {
    const material = new MeshBasicMaterial({ transparent: true, depthWrite: false });
    const edgeMaterial = new LineBasicMaterial({ transparent: true, opacity: CARD_EDGE_OPACITY });
    const mesh = new Mesh(this.geometry, material);
    mesh.add(new LineSegments(this.edges, edgeMaterial));
    this.c.scene.add(mesh);
    const label = this.c.labels.add('tag', extraClass);
    label.anchor = anchor;
    label.avoidsBlocks = true;
    label.setText(text);
    return { mesh, material, edgeMaterial, label };
  }

  private chipView(view: GroundView, k: number): ChipView {
    while (view.chips.length <= k) {
      const total = this.c.labels.add('tag', 'vf-chip');
      total.anchor = 'above-right';
      total.avoidsBlocks = true;
      const more = this.c.labels.add('copy');
      more.anchor = 'left-middle';
      view.chips.push({
        card: this.card(CHIP_TEXT, 'vf-chip', 'below-right'),
        shared: this.card(SHARED_TEXT, '', 'below-right'),
        registers: this.card(REGISTER_TEXT, '', 'below-right'),
        total,
        more,
        copies: [],
      });
    }
    return view.chips[k];
  }

  private view(i: number): GroundView {
    while (this.views.length <= i) this.views.push({ hbm: this.card(HBM_TEXT, 'vf-hbm', 'below-right'), chips: [] });
    return this.views[i];
  }

  private copyLabel(view: ChipView, k: number): Label {
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

  private placeCard(card: Card, box: CardBox | undefined, y: number, color: LinearRgb, opacity: number, named: boolean, inside: boolean): void {
    card.mesh.visible = box !== undefined;
    card.label.visible = box !== undefined && named;
    if (box === undefined) return;
    const [x0, x1] = box.x;
    const [z0, z1] = box.z;
    card.mesh.scale.set(x1 - x0, CARD_THICKNESS, z1 - z0);
    card.mesh.position.set((x0 + x1) / 2, y - CARD_DROP, (z0 + z1) / 2);
    card.material.color.setRGB(...color);
    card.material.opacity = opacity;
    card.edgeMaterial.color.setRGB(...color);
    card.label.pos = inside ? [x0, y - CARD_DROP, z0] : [x0, y - CARD_DROP, z1];
  }

  private hide(card: Card): void {
    card.mesh.visible = false;
    card.label.visible = false;
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
    const theme = c.theme;
    const grounds: readonly KernelGround[] = c.levelLayout?.kernels ?? [];
    const busyLines: Polyline[] = [];
    const idleLines: Polyline[] = [];
    const outlines: Outline[] = [];
    this.labelled.clear();
    grounds.forEach((ground, i) => {
      const view = this.view(i);
      if (c.focusGround !== undefined && c.focusGround !== ground) {
        this.hide(view.hbm);
        for (const chipView of view.chips) this.hideChip(chipView);
        return;
      }
      const grid = currentGrid(ground.grids, c.state);
      const busy = busyCopies(grid, (n) => c.state.active(n));
      const named = !c.lod && busy.length > 0;
      const y = ground.y;
      this.placeCard(view.hbm, ground.hbm, y, theme.neg, HBM_OPACITY, named, false);
      const isBusy = (index: readonly number[]) => busy.some((b) => sameCopy(b, index));
      const stacked = grid.arrangement.some((a, d) => a === 'stack' && grid.counts[d] > 1);
      ground.chips.forEach((chip, k) => {
        const chipView = this.chipView(view, k);
        const front = frontIn(grid, c.frontOf(grid), chip.expand);
        const chipNamed = named && busy.some((index) => sameCopy(index.filter((_, d) => grid.arrangement[d] === 'expand'), chip.expand));
        this.placeCard(chipView.card, chip.card, y, theme.pos, CHIP_OPACITY, chipNamed, false);
        this.placeCard(chipView.shared, chip.shared, y + CARD_THICKNESS, theme.pos, PLATE_OPACITY, chipNamed, true);
        this.placeCard(chipView.registers, chip.registers, y + CARD_THICKNESS, theme.pos, PLATE_OPACITY, chipNamed, true);
        chipView.card.label.setText(grid.names.length === 0 ? CHIP_TEXT : `${CHIP_TEXT}：${copyName(grid, front)}`);
        chipView.card.label.setFlag('vf-on', isBusy(front));
        (isBusy(front) ? busyLines : idleLines).push({ points: outlineRect(chip.card, y - CARD_DROP, 0), opacity: FRAME_OPACITY });
        const others = stacked ? stackedCopies(grid, front, chip.expand) : [];
        const drawn = others.slice(0, OUTLINE_MAX);
        const [x1, z0] = [chip.card.x[1], chip.card.z[0]];
        const corner = (j: number): Vec3 => [x1 + OUTLINE_STEP[0] * j, y - CARD_DROP + OUTLINE_STEP[1] * j, z0 + OUTLINE_STEP[2] * j];
        drawn.forEach((index, n) => {
          const j = n + 1;
          const points = outlineRect(chip.card, y - CARD_DROP, j);
          const on = isBusy(index);
          (on ? busyLines : idleLines).push({ points, opacity: OUTLINE_OPACITY });
          const label = this.copyLabel(chipView, n);
          label.visible = chipNamed;
          label.pos = corner(j);
          label.setText(copyName(grid, index));
          label.setFlag('vf-on', on);
          const outline = { grid, index, busy: on, points };
          outlines.push(outline);
          this.labelled.set(label, outline);
        });
        chipView.copies.slice(drawn.length).forEach((label) => (label.visible = false));
        const rest = others.length - drawn.length;
        chipView.more.visible = chipNamed && rest > 0;
        if (rest > 0) {
          chipView.more.pos = corner(drawn.length + 1);
          chipView.more.setText(`还有 ${rest} 个`);
        }
        const total = this.total(grid);
        chipView.total.visible = chipNamed && stacked;
        const back = corner(drawn.length + (rest > 0 ? 1 : 0));
        chipView.total.pos = [chip.card.x[0] + OUTLINE_STEP[0] * drawn.length, back[1], back[2]];
        chipView.total.setText(`共 ${total.drawn} 个副本同时执行${total.real === undefined ? '' : `（真实尺寸 ${total.real} 个）`}`);
      });
      for (const chipView of view.chips.slice(ground.chips.length)) this.hideChip(chipView);
    });
    for (const view of this.views.slice(grounds.length)) {
      this.hide(view.hbm);
      for (const chipView of view.chips) this.hideChip(chipView);
    }
    this.outlines = outlines;
    this.screenVersion = -1;
    this.busy.setPolylines(busyLines);
    this.idle.setPolylines(idleLines);
    this.links.setPolylines(this.linkLines());
  }

  private linkLines(): Polyline[] {
    const c = this.c;
    if (c.lod) return [];
    return (c.levelLayout?.links ?? []).flatMap(({ original, copy }) => {
      const from = renderableFor(c, original);
      const to = renderableFor(c, copy);
      if (from === undefined || to === undefined || (from.far && to.far) || from.away || to.away || !c.state.revealed(original.node)) return [];
      return [{ points: linkPoints(original, copy), opacity: LINK_OPACITY, dashed: true }];
    });
  }

  private hideChip(view: ChipView): void {
    for (const card of [view.card, view.shared, view.registers]) this.hide(card);
    for (const label of [view.total, view.more, ...view.copies]) label.visible = false;
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
    this.links.setColor(theme.ink);
  }

  dispose(): void {
    this.busy.dispose();
    this.idle.dispose();
    this.links.dispose();
    this.geometry.dispose();
    this.edges.dispose();
    for (const view of this.views) {
      for (const card of [view.hbm, ...view.chips.flatMap((chip) => [chip.card, chip.shared, chip.registers])]) {
        card.material.dispose();
        card.edgeMaterial.dispose();
      }
    }
  }
}
