import { type Camera, Vector3 } from 'three';
import { type Placement, avoidOverlaps } from './avoid';
import type { Vec3 } from './layout';

export type LabelKind = 'val' | 'tick' | 'ttl' | 'ntl' | 'nv' | 'cls' | 'kk' | 'colh' | 'mod' | 'tok' | 'boxlb';

const ANCHORS: Record<string, [number, number]> = {
  'top-center': [-0.5, -1],
  'bottom-center': [-0.5, 0],
  'right-middle': [-1, -0.5],
  'left-middle': [0, -0.5],
  'above-right': [0, -1],
};

export type AnchorName = keyof typeof ANCHORS;

export const LABEL_SAFE_MARGIN = 12;
export const LABEL_CLIP = 1.2;

export function anchorOffset(anchor: AnchorName): [number, number] {
  return ANCHORS[anchor];
}

const DEFAULT_ANCHOR: Record<LabelKind, AnchorName> = {
  val: 'top-center',
  tick: 'right-middle',
  ttl: 'right-middle',
  ntl: 'above-right',
  nv: 'left-middle',
  cls: 'bottom-center',
  kk: 'right-middle',
  colh: 'top-center',
  mod: 'right-middle',
  tok: 'left-middle',
  boxlb: 'above-right',
};

export const PRIORITY = { title: 5, node: 4, value: 3, category: 2, minor: 1 } as const;

export class Label {
  readonly el: HTMLDivElement;
  pos: Vec3 | undefined = undefined;
  anchor: AnchorName;
  readonly priority: number;
  readonly movable: boolean;
  readonly hideOnOverlap: boolean;
  visible = true;
  size: [number, number] | undefined = undefined;
  under: Label[] = [];
  offset: [number, number] | undefined = undefined;
  base: [number, number] = [0, 0];
  placed: [number, number] = [0, 0];
  private content = '';
  private alternate: { key: string; build: () => Node } | undefined = undefined;
  private alternateSize: [number, number] | undefined = undefined;

  constructor(readonly kind: LabelKind, extraClass: string) {
    this.el = document.createElement('div');
    this.el.className = `vf-lb vf-${kind}${extraClass ? ` ${extraClass}` : ''}`;
    this.anchor = DEFAULT_ANCHOR[kind];
    this.priority =
      kind === 'ttl' || kind === 'mod' || kind === 'boxlb' ? PRIORITY.title : kind === 'ntl' ? PRIORITY.node : kind === 'val' || kind === 'nv' ? PRIORITY.value : kind === 'cls' || kind === 'colh' || kind === 'tok' ? PRIORITY.category : PRIORITY.minor;
    this.movable = kind === 'ttl' || kind === 'ntl' || kind === 'mod';
    this.hideOnOverlap = !this.movable && kind !== 'tok' && kind !== 'boxlb';
  }

  setText(text: string): void {
    if (this.content === text) return;
    this.content = text;
    this.el.textContent = text;
    this.size = undefined;
  }

  setContent(key: string, build: () => Node): void {
    if (this.content === key) return;
    this.content = key;
    this.el.replaceChildren(build());
    this.size = undefined;
  }

  setAlternate(key: string, build: () => Node): void {
    if (this.alternate?.key === key) return;
    this.alternate = { key, build };
    this.alternateSize = undefined;
  }

  measure(): [number, number] {
    if (this.size === undefined) this.size = [this.el.offsetWidth, this.el.offsetHeight];
    return this.size;
  }

  frameSize(): [number, number] {
    const [w, h] = this.measure();
    if (this.alternate === undefined || this.el.parentElement === null) return [w, h];
    if (this.alternateSize === undefined) {
      const probe = this.el.cloneNode(false) as HTMLDivElement;
      probe.classList.add('vf-off');
      probe.style.transform = '';
      probe.append(this.alternate.build());
      this.el.parentElement.append(probe);
      this.alternateSize = [probe.offsetWidth, probe.offsetHeight];
      probe.remove();
    }
    return [Math.max(w, this.alternateSize[0]), Math.max(h, this.alternateSize[1])];
  }

  forgetSizes(): void {
    this.size = undefined;
    this.alternateSize = undefined;
  }

  setFlag(className: string, on: boolean): void {
    if (this.el.classList.contains(className) === on) return;
    this.el.classList.toggle(className, on);
    this.size = undefined;
  }
}

const projected = new Vector3();

export class LabelLayer {
  readonly root: HTMLDivElement;
  readonly labels: Label[] = [];

  constructor(parent: HTMLElement, onRemeasure: () => void) {
    this.root = document.createElement('div');
    this.root.className = 'vf-labels';
    parent.appendChild(this.root);
    document.fonts?.ready.then(() => {
      for (const label of this.labels) label.forgetSizes();
      onRemeasure();
    });
  }

  add(kind: LabelKind, extraClass = ''): Label {
    const label = new Label(kind, extraClass);
    this.root.appendChild(label.el);
    this.labels.push(label);
    return label;
  }

  project(camera: Camera, width: number, height: number, avoid = true): void {
    const shown: Label[] = [];
    for (const label of this.labels) {
      let off = label.pos === undefined || !label.visible;
      if (!off) {
        projected.set(...label.pos!).project(camera);
        off = Math.abs(projected.x) > LABEL_CLIP || Math.abs(projected.y) > LABEL_CLIP;
      }
      label.el.classList.toggle('vf-off', off);
      if (!off) shown.push(label);
    }
    for (const label of shown) label.measure();
    const bounds = { x: LABEL_SAFE_MARGIN, y: LABEL_SAFE_MARGIN, w: width - 2 * LABEL_SAFE_MARGIN, h: height - 2 * LABEL_SAFE_MARGIN };
    const placements: Placement[] = shown.map((label) => {
      projected.set(...label.pos!).project(camera);
      const px = ((projected.x + 1) / 2) * width;
      const py = ((1 - projected.y) / 2) * height;
      const [ax, ay] = ANCHORS[label.anchor];
      const [w, h] = label.size!;
      const box = { x: px + ax * w, y: py + ay * h, w, h };
      label.base = [box.x, box.y];
      if (label.offset !== undefined) {
        box.x += label.offset[0];
        box.y += label.offset[1];
        return { box, movable: false, hideOnOverlap: false, priority: label.priority };
      }
      return {
        box,
        movable: label.movable,
        hideOnOverlap: label.hideOnOverlap,
        priority: label.priority,
      };
    });
    const index = new Map(shown.map((label, i) => [label, i]));
    shown.forEach((label, i) => {
      const { box } = placements[i];
      for (const above of label.under) {
        const j = index.get(above);
        if (j === undefined) continue;
        const other = placements[j].box;
        if (other.x < box.x + box.w && box.x < other.x + other.w) box.y = Math.max(box.y, other.y + other.h);
      }
    });
    const adjustments = avoid ? avoidOverlaps(placements, bounds) : placements.map(() => ({ dx: 0, dy: 0, hidden: false }));
    shown.forEach((label, i) => {
      const { box } = placements[i];
      const { dx, dy, hidden } = adjustments[i];
      label.el.classList.toggle('vf-off', hidden);
      label.placed = [box.x + dx, box.y + dy];
      label.el.style.transform = `translate(${(box.x + dx).toFixed(1)}px, ${(box.y + dy).toFixed(1)}px)`;
    });
  }

  dispose(): void {
    this.root.remove();
    this.labels.length = 0;
  }
}
