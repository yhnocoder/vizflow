import { type Camera, Vector3 } from 'three';
import { type Placement, avoidOverlaps } from './avoid';
import type { Vec3 } from './layout';

export type LabelKind = 'val' | 'tick' | 'ttl' | 'node' | 'cls' | 'kk' | 'colh';

const ANCHORS: Record<string, [number, number]> = {
  'top-center': [-0.5, -1],
  'bottom-center': [-0.5, 0],
  'right-middle': [-1, -0.5],
  'left-middle': [0, -0.5],
};

export type AnchorName = keyof typeof ANCHORS;

const DEFAULT_ANCHOR: Record<LabelKind, AnchorName> = {
  val: 'top-center',
  tick: 'right-middle',
  ttl: 'right-middle',
  node: 'top-center',
  cls: 'bottom-center',
  kk: 'right-middle',
  colh: 'top-center',
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
  private content = '';

  constructor(readonly kind: LabelKind, extraClass: string) {
    this.el = document.createElement('div');
    this.el.className = `vf-lb vf-${kind}${extraClass ? ` ${extraClass}` : ''}`;
    this.anchor = DEFAULT_ANCHOR[kind];
    this.priority =
      kind === 'ttl' ? PRIORITY.title : kind === 'node' ? PRIORITY.node : kind === 'val' ? PRIORITY.value : kind === 'cls' || kind === 'colh' ? PRIORITY.category : PRIORITY.minor;
    this.movable = kind === 'ttl' || kind === 'node';
    this.hideOnOverlap = !this.movable;
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
      for (const label of this.labels) label.size = undefined;
      onRemeasure();
    });
  }

  add(kind: LabelKind, extraClass = ''): Label {
    const label = new Label(kind, extraClass);
    this.root.appendChild(label.el);
    this.labels.push(label);
    return label;
  }

  project(camera: Camera, width: number, height: number): void {
    const shown: Label[] = [];
    for (const label of this.labels) {
      const off = label.pos === undefined || !label.visible;
      label.el.classList.toggle('vf-off', off);
      if (!off) shown.push(label);
    }
    for (const label of shown) {
      if (label.size === undefined) label.size = [label.el.offsetWidth, label.el.offsetHeight];
    }
    const placements: Placement[] = shown.map((label) => {
      projected.set(...label.pos!).project(camera);
      const px = ((projected.x + 1) / 2) * width;
      const py = ((1 - projected.y) / 2) * height;
      const [ax, ay] = ANCHORS[label.anchor];
      const [w, h] = label.size!;
      return {
        box: { x: px + ax * w, y: py + ay * h, w, h },
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
    const adjustments = avoidOverlaps(placements);
    shown.forEach((label, i) => {
      const { box } = placements[i];
      const { dx, dy, hidden } = adjustments[i];
      label.el.classList.toggle('vf-off', hidden);
      label.el.style.transform = `translate(${(box.x + dx).toFixed(1)}px, ${(box.y + dy).toFixed(1)}px)`;
    });
  }

  dispose(): void {
    this.root.remove();
    this.labels.length = 0;
  }
}
