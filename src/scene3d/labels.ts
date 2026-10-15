import { type Camera, Vector3 } from 'three';
import { type Box, type Placement, avoidOverlaps, spreadLane } from './avoid';
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
export const LANE_GAP = 4;

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
  lane: string | undefined = undefined;
  size: [number, number] | undefined = undefined;
  under: Label[] = [];
  offset: [number, number] | undefined = undefined;
  base: [number, number] = [0, 0];
  placed: [number, number] = [0, 0];
  private content = '';
  private off = false;
  private transform = '';
  private readonly data: Partial<Record<'node' | 'index', string>> = {};
  private alternate: { key: string; build: () => Node } | undefined = undefined;
  private alternateSize: [number, number] | undefined = undefined;

  constructor(readonly kind: LabelKind, extraClass: string, private readonly host: HTMLElement) {
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
    if (this.el.parentElement === null) this.host.appendChild(this.el);
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

  setData(name: 'node' | 'index', value: string): void {
    if (this.data[name] === value) return;
    this.data[name] = value;
    this.el.dataset[name] = value;
  }

  setOff(off: boolean): void {
    if (this.off === off) return;
    this.off = off;
    this.el.classList.toggle('vf-off', off);
  }

  moveTo(x: number, y: number): void {
    const transform = `translate(${x.toFixed(1)}px, ${y.toFixed(1)}px)`;
    if (transform === this.transform) return;
    this.transform = transform;
    this.el.style.transform = transform;
  }

  hasFlag(className: string): boolean {
    return this.el.classList.contains(className);
  }

  setFlag(className: string, on: boolean): void {
    if (this.el.classList.contains(className) === on) return;
    this.el.classList.toggle(className, on);
    this.size = undefined;
  }
}

export const IDLE_FLAG = 'vf-idle';
export const IDLE_YIELD = 0.5;

const projected = new Vector3();

export class LabelLayer {
  readonly root: HTMLDivElement;
  readonly labels: Label[] = [];
  only: ReadonlySet<Label> | undefined = undefined;

  constructor(parent: HTMLElement, onRemeasure: () => void) {
    this.root = document.createElement('div');
    this.root.className = 'vf-labels';
    parent.appendChild(this.root);
    document.fonts?.ready.then(() => {
      for (const label of this.labels) label.forgetSizes();
      onRemeasure();
    });
  }

  allows(label: Label): boolean {
    return this.only === undefined || this.only.has(label);
  }

  add(kind: LabelKind, extraClass = ''): Label {
    const label = new Label(kind, extraClass, this.root);
    this.labels.push(label);
    return label;
  }

  project(camera: Camera, width: number, height: number, avoidAll = true, obstacles: readonly Box[] = []): void {
    const shown: Label[] = [];
    for (const label of this.labels) {
      let off = label.pos === undefined || !label.visible || !this.allows(label);
      if (!off) {
        projected.set(...label.pos!).project(camera);
        off = Math.abs(projected.x) > LABEL_CLIP || Math.abs(projected.y) > LABEL_CLIP;
      }
      if (off) label.setOff(true);
      else shown.push(label);
    }
    for (const label of shown) label.setOff(false);
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
        priority: label.hasFlag(IDLE_FLAG) ? label.priority - IDLE_YIELD : label.priority,
      };
    });
    const lanes = new Map<string, number[]>();
    shown.forEach((label, i) => {
      if (label.lane === undefined || label.offset !== undefined) return;
      lanes.set(label.lane, [...(lanes.get(label.lane) ?? []), i]);
    });
    for (const lane of lanes.values()) {
      const all = lane.map((i) => placements[i].box);
      const xLo = Math.min(...all.map((b) => b.x));
      const xHi = Math.max(...all.map((b) => b.x + b.w));
      const across = obstacles.filter((o) => o.x < xHi && xLo < o.x + o.w);
      const top = Math.max(LABEL_SAFE_MARGIN, ...across.filter((o) => o.y + o.h / 2 < height / 2).map((o) => o.y + o.h + LABEL_SAFE_MARGIN));
      const bottom = Math.min(height - LABEL_SAFE_MARGIN, ...across.filter((o) => o.y + o.h / 2 >= height / 2).map((o) => o.y - LABEL_SAFE_MARGIN));
      const members = lane.filter((i) => {
        const middle = placements[i].box.y + placements[i].box.h / 2;
        return middle >= 0 && middle <= height;
      });
      spreadLane(members.map((i) => placements[i].box), top, bottom, LANE_GAP);
      for (const i of members) placements[i] = { ...placements[i], movable: false, hideOnOverlap: false, priority: PRIORITY.title + 1 };
    }
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
    const avoiding = shown.flatMap((label, i) => (avoidAll || label.movable ? [i] : []));
    const adjusted = avoidOverlaps(
      avoiding.map((i) => placements[i]),
      avoidAll ? bounds : undefined,
    );
    const adjustments = placements.map(() => ({ dx: 0, dy: 0, hidden: false }));
    avoiding.forEach((i, k) => {
      adjustments[i] = adjusted[k];
    });
    shown.forEach((label, i) => {
      const { box } = placements[i];
      const { dx, dy, hidden } = adjustments[i];
      label.placed = [box.x + dx, box.y + dy];
      label.moveTo(box.x + dx, box.y + dy);
      label.setOff(hidden);
    });
  }

  dispose(): void {
    this.root.remove();
    this.labels.length = 0;
  }
}
