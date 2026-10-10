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

export const PRIORITY = { fixed: 4, title: 3, node: 2, value: 1 } as const;

export interface Label {
  readonly el: HTMLDivElement;
  readonly kind: LabelKind;
  pos: Vec3 | undefined;
  anchor: AnchorName;
  priority: number;
  movable: boolean;
  hideOnOverlap: boolean;
  visible: boolean;
}

const projected = new Vector3();

export class LabelLayer {
  readonly root: HTMLDivElement;
  readonly labels: Label[] = [];

  constructor(parent: HTMLElement) {
    this.root = document.createElement('div');
    this.root.className = 'vf-labels';
    parent.appendChild(this.root);
  }

  add(kind: LabelKind, extraClass = ''): Label {
    const el = document.createElement('div');
    el.className = `vf-lb vf-${kind}${extraClass ? ` ${extraClass}` : ''}`;
    this.root.appendChild(el);
    const label: Label = {
      el,
      kind,
      pos: undefined,
      anchor: DEFAULT_ANCHOR[kind],
      priority: kind === 'ttl' ? PRIORITY.title : kind === 'node' ? PRIORITY.node : kind === 'val' ? PRIORITY.value : PRIORITY.fixed,
      movable: kind === 'ttl' || kind === 'node',
      hideOnOverlap: kind === 'val',
      visible: true,
    };
    this.labels.push(label);
    return label;
  }

  project(camera: Camera, width: number, height: number): void {
    const shown: Label[] = [];
    const sizes: [number, number][] = [];
    const points: [number, number][] = [];
    for (const label of this.labels) {
      const off = label.pos === undefined || !label.visible;
      label.el.classList.toggle('vf-off', off);
      if (off) continue;
      projected.set(...label.pos!).project(camera);
      const px = ((projected.x + 1) / 2) * width;
      const py = ((1 - projected.y) / 2) * height;
      shown.push(label);
      points.push([px, py]);
      sizes.push([label.el.offsetWidth, label.el.offsetHeight]);
    }
    const placements: Placement[] = shown.map((label, i) => {
      const [ax, ay] = ANCHORS[label.anchor];
      const [w, h] = sizes[i];
      return {
        box: { x: points[i][0] + ax * w, y: points[i][1] + ay * h, w, h },
        movable: label.movable,
        hideOnOverlap: label.hideOnOverlap,
        priority: label.priority,
      };
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
