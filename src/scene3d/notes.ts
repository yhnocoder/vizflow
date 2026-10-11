import { Vector3 } from 'three';
import type { Node } from '../autograd';
import { renderText } from '../formula';
import type { NoteSide, StepNote } from '../steps';
import { type Box, intersects } from './avoid';
import type { SceneController } from './controller';
import { shown } from './renderable';

export const NOTE_GAP = 46;
export const NOTE_MARGIN = 10;
export const NOTE_TARGET_PAD = 8;
export const NOTE_ARROW_INSET = 5;
export const NOTE_HEAD_LENGTH = 11;
export const NOTE_HEAD_ANGLE = 0.47;
export const NOTE_BEND = 0.2;
export const NOTE_JITTER = 1.6;
export const NOTE_SAMPLES = 28;
export const NOTE_FADE_MS = 260;
export const NOTE_REACH = [1, 1.8, 2.8];
export const NOTE_SHIFTS = [-0.5, 0, -1, 0.5, 1];

const SVG = 'http://www.w3.org/2000/svg';
const CHROME_SELECTOR = ':scope > .vf-stage-tools, :scope > .vf-ovnav, :scope > .vf-key, :scope > .vf-hint, :scope > .vf-formula';
const tmpVector = new Vector3();

type Point = [number, number];

interface NoteTarget {
  nodes: Node[];
  index?: number[];
}

interface NoteGeometry {
  rect: Box;
  point?: Point;
}

interface NoteItem {
  key: string;
  text: string;
  preferred: NoteSide | undefined;
  side: NoteSide | undefined;
  target: NoteTarget;
  el: HTMLDivElement;
  ink: SVGGElement;
  strokes: SVGPathElement[];
  seed: number;
}

function hash(text: string): number {
  let h = 2166136261;
  for (let i = 0; i < text.length; i++) h = Math.imul(h ^ text.charCodeAt(i), 16777619);
  return h >>> 0;
}

function random(seed: number): () => number {
  let a = seed;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296 - 0.5;
  };
}

function overlap(a: Box, b: Box): number {
  if (!intersects(a, b)) return 0;
  return (Math.min(a.x + a.w, b.x + b.w) - Math.max(a.x, b.x)) * (Math.min(a.y + a.h, b.y + b.h) - Math.max(a.y, b.y));
}

function clampToBox(p: Point, box: Box): Point {
  return [Math.min(Math.max(p[0], box.x), box.x + box.w), Math.min(Math.max(p[1], box.y), box.y + box.h)];
}

function exitPoint(box: Box, toward: Point, pad: number): Point {
  const cx = box.x + box.w / 2;
  const cy = box.y + box.h / 2;
  const dx = toward[0] - cx;
  const dy = toward[1] - cy;
  const hx = box.w / 2 + pad;
  const hy = box.h / 2 + pad;
  const t = Math.min(dx === 0 ? Infinity : hx / Math.abs(dx), dy === 0 ? Infinity : hy / Math.abs(dy));
  return Number.isFinite(t) ? [cx + dx * t, cy + dy * t] : [cx, cy];
}

function cubic(points: Point[], t: number): Point {
  const [a, b, c, d] = points;
  const u = 1 - t;
  const w = [u * u * u, 3 * u * u * t, 3 * u * t * t, t * t * t];
  return [w[0] * a[0] + w[1] * b[0] + w[2] * c[0] + w[3] * d[0], w[0] * a[1] + w[1] * b[1] + w[2] * c[1] + w[3] * d[1]];
}

function wobble(points: Point[], rand: () => number, amplitude: number, overshoot: number): string {
  const jittered = points.map(([x, y], i): Point => (i === 0 || i === 3 ? [x + rand() * overshoot, y + rand() * overshoot] : [x + rand() * amplitude * 3, y + rand() * amplitude * 3]));
  const waves = [0, 1, 2].map((k) => ({ freq: 1.2 + k * 1.3 + rand(), phase: rand() * Math.PI * 2, amp: amplitude / (k + 1) }));
  const out: string[] = [];
  for (let i = 0; i <= NOTE_SAMPLES; i++) {
    const t = i / NOTE_SAMPLES;
    const [x, y] = cubic(jittered, t);
    const [x2, y2] = cubic(jittered, Math.min(t + 0.01, 1));
    const [x1, y1] = cubic(jittered, Math.max(t - 0.01, 0));
    const tl = Math.hypot(x2 - x1, y2 - y1) || 1;
    const taper = Math.sin(Math.PI * t) ** 0.5;
    const offset = waves.reduce((sum, w) => sum + Math.sin(t * w.freq * Math.PI * 2 + w.phase) * w.amp, 0) * taper;
    out.push(`${i === 0 ? 'M' : 'L'}${(x - ((y2 - y1) / tl) * offset).toFixed(1)},${(y + ((x2 - x1) / tl) * offset).toFixed(1)}`);
  }
  return out.join(' ');
}

function roughLine(from: Point, to: Point, rand: () => number, jitter: number): string {
  const mid: Point = [(from[0] + to[0]) / 2 + rand() * jitter, (from[1] + to[1]) / 2 + rand() * jitter];
  return `M${(from[0] + rand() * jitter).toFixed(1)},${(from[1] + rand() * jitter).toFixed(1)} Q${mid[0].toFixed(1)},${mid[1].toFixed(1)} ${(to[0] + rand() * jitter).toFixed(1)},${(to[1] + rand() * jitter).toFixed(1)}`;
}

function inside(p: Point, box: Box): boolean {
  return p[0] > box.x && p[0] < box.x + box.w && p[1] > box.y && p[1] < box.y + box.h;
}

export function arrowPaths(start: Point, end: Point, seed: number, from: Box): string[] {
  const dx = end[0] - start[0];
  const dy = end[1] - start[1];
  const length = Math.hypot(dx, dy) || 1;
  const nx = -dy / length;
  const ny = dx / length;
  const rand = random(seed);
  const magnitude = Math.min(Math.max(length * NOTE_BEND, 8), 42);
  const controls = (sign: number): Point[] => [
    start,
    [start[0] + dx * 0.3 + nx * magnitude * sign, start[1] + dy * 0.3 + ny * magnitude * sign],
    [start[0] + dx * 0.72 + nx * magnitude * sign * 0.55, start[1] + dy * 0.72 + ny * magnitude * sign * 0.55],
    end,
  ];
  const preferred = rand() > 0 ? 1 : -1;
  const curve = inside(controls(preferred)[1], from) ? controls(-preferred) : controls(preferred);
  const shafts = [wobble(curve, rand, NOTE_JITTER * 0.5, 1), wobble(curve, rand, NOTE_JITTER, 3)];
  const before = cubic(curve, 0.93);
  const angle = Math.atan2(end[1] - before[1], end[0] - before[0]);
  const heads = [-1, 1].flatMap((side) => {
    const a = angle + Math.PI - side * (NOTE_HEAD_ANGLE + rand() * 0.12);
    const len = NOTE_HEAD_LENGTH * (1 + rand() * 0.25);
    const tip: Point = [end[0] + Math.cos(a) * len, end[1] + Math.sin(a) * len];
    return [roughLine(end, tip, rand, 0.9), roughLine(end, tip, rand, 1.6)];
  });
  return [...shafts, ...heads];
}

function candidates(rect: Box, size: [number, number], width: number, height: number, preferred: NoteSide | undefined): { side: NoteSide; box: Box; cost: number }[] {
  const [w, h] = size;
  const cx = rect.x + rect.w / 2;
  const cy = rect.y + rect.h / 2;
  const room: Record<NoteSide, number> = {
    right: (width - (rect.x + rect.w)) / w,
    left: rect.x / w,
    above: rect.y / h,
    below: (height - (rect.y + rect.h)) / h,
  };
  const sides = (Object.keys(room) as NoteSide[]).sort((a, b) => room[b] - room[a]);
  const order = preferred === undefined ? sides : [preferred, ...sides.filter((s) => s !== preferred)];
  const out: { side: NoteSide; box: Box; cost: number }[] = [];
  order.forEach((side, rank) => {
    for (const [k, reach] of NOTE_REACH.entries()) {
      const gap = NOTE_GAP * reach;
      for (const [j, shift] of NOTE_SHIFTS.entries()) {
        const box: Box =
          side === 'right'
            ? { x: rect.x + rect.w + gap, y: cy - h / 2 + shift * (h + gap * 0.6), w, h }
            : side === 'left'
              ? { x: rect.x - gap - w, y: cy - h / 2 + shift * (h + gap * 0.6), w, h }
              : side === 'above'
                ? { x: cx - w / 2 + shift * (w * 0.6 + gap), y: rect.y - gap - h, w, h }
                : { x: cx - w / 2 + shift * (w * 0.6 + gap), y: rect.y + rect.h + gap, w, h };
        box.x = Math.min(Math.max(box.x, NOTE_MARGIN), width - NOTE_MARGIN - w);
        box.y = Math.min(Math.max(box.y, NOTE_MARGIN), height - NOTE_MARGIN - h);
        out.push({ side, box, cost: rank * (preferred === undefined ? 30 : 400) + k * 25 + j * 6 });
      }
    }
  });
  return out;
}

export class SceneNotes {
  private readonly svg: SVGSVGElement;
  private readonly layer: HTMLDivElement;
  private readonly items = new Map<string, NoteItem>();

  constructor(private readonly c: SceneController) {
    this.svg = document.createElementNS(SVG, 'svg');
    this.svg.setAttribute('class', 'vf-note-ink');
    this.svg.setAttribute('aria-hidden', 'true');
    this.layer = document.createElement('div');
    this.layer.className = 'vf-notes';
    c.root.append(this.svg, this.layer);
  }

  update(notes: readonly StepNote[], stepId: string): void {
    const views = notes.map((note, i) => ({ key: `${stepId}:${i}`, note }));
    const keep = new Set(views.map((v) => v.key));
    let changed = false;
    for (const [key, item] of this.items) {
      if (keep.has(key)) continue;
      this.items.delete(key);
      item.el.classList.remove('vf-note-in');
      item.ink.classList.remove('vf-note-in');
      setTimeout(() => {
        item.el.remove();
        item.ink.remove();
      }, NOTE_FADE_MS);
      changed = true;
    }
    for (const { key, note } of views) {
      const existing = this.items.get(key);
      if (existing !== undefined && existing.text === note.text) continue;
      existing?.el.remove();
      existing?.ink.remove();
      this.items.set(key, this.create(key, note, existing !== undefined));
      changed = true;
    }
    if (changed) this.c.markLabelsDirty();
  }

  private create(key: string, note: StepNote, instant: boolean): NoteItem {
    const el = document.createElement('div');
    el.className = 'vf-note vf-off';
    el.dataset.note = key;
    renderText(note.text, el);
    const ink = document.createElementNS(SVG, 'g');
    ink.setAttribute('class', 'vf-note-arrow');
    const strokes = Array.from({ length: 6 }, (_, i) => {
      const path = document.createElementNS(SVG, 'path');
      path.setAttribute('pathLength', '1');
      path.setAttribute('class', i < 2 ? 'vf-note-shaft' : 'vf-note-head');
      ink.append(path);
      return path;
    });
    this.layer.append(el);
    this.svg.append(ink);
    if (instant) {
      el.classList.add('vf-note-still');
      ink.classList.add('vf-note-still');
    }
    requestAnimationFrame(() =>
      requestAnimationFrame(() => {
        el.classList.add('vf-note-in');
        ink.classList.add('vf-note-in');
      }),
    );
    const target: NoteTarget = { nodes: note.nodes };
    if (note.index !== undefined) target.index = note.index;
    return { key, text: note.text, preferred: note.side, side: undefined, target, el, ink, strokes, seed: hash(key) };
  }

  private screenPoint(x: number, y: number, z: number): Point {
    tmpVector.set(x, y, z).project(this.c.camera);
    return [((tmpVector.x + 1) / 2) * this.c.width, ((1 - tmpVector.y) / 2) * this.c.height];
  }

  private geometry(target: NoteTarget): NoteGeometry | undefined {
    const c = this.c;
    const points: Point[] = [];
    let point: Point | undefined;
    for (const node of target.nodes) {
      if (!c.state.revealed(node)) continue;
      const found = c.renderables.filter((r) => r.block.node === node && shown(r.block));
      const renderable = found.find((r) => r.block.copyOf === undefined) ?? found[0];
      if (renderable === undefined) continue;
      renderable.block.elements.forEach((element, i) => {
        if (target.index !== undefined && element.index.some((k, axis) => k !== target.index![axis])) return;
        const h = renderable.bars === undefined ? 0 : renderable.transition.current[i];
        const top = this.screenPoint(element.x, element.y + Math.max(h, 0), element.z);
        const bottom = this.screenPoint(element.x, element.y + Math.min(h, 0), element.z);
        points.push(top, bottom);
        if (target.index !== undefined) point = h >= 0 ? top : bottom;
      });
    }
    if (points.length === 0) return undefined;
    const xs = points.map((p) => p[0]);
    const ys = points.map((p) => p[1]);
    const x = Math.min(...xs);
    const y = Math.min(...ys);
    const geometry: NoteGeometry = { rect: { x, y, w: Math.max(...xs) - x, h: Math.max(...ys) - y } };
    if (point !== undefined) geometry.point = point;
    return geometry;
  }

  private obstacles(): Box[] {
    const c = this.c;
    const origin = c.root.getBoundingClientRect();
    const chrome = [...c.root.querySelectorAll<HTMLElement>(CHROME_SELECTOR)]
      .map((el) => el.getBoundingClientRect())
      .filter((r) => r.width > 0)
      .map((r) => ({ x: r.left - origin.left - 6, y: r.top - origin.top - 6, w: r.width + 12, h: r.height + 12 }));
    const labels = c.labels.labels
      .filter((label) => label.visible && label.pos !== undefined && !label.el.classList.contains('vf-off'))
      .map((label) => {
        const [w, h] = label.measure();
        return { x: label.placed[0], y: label.placed[1], w, h };
      });
    const revealed = c.renderables.filter((r) => c.state.revealed(r.block.node) && shown(r.block)).map((r) => r.block.node);
    const blocks = [...new Set([...c.stepNodes, ...revealed])].flatMap((node) => {
      const g = this.geometry({ nodes: [node] });
      return g === undefined ? [] : [{ x: g.rect.x - 4, y: g.rect.y - 4, w: g.rect.w + 8, h: g.rect.h + 8 }];
    });
    return [...chrome, ...labels, ...blocks];
  }

  place(): void {
    if (this.items.size === 0) return;
    const { width, height } = this.c;
    this.svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
    const common = this.obstacles();
    const placed: Box[] = [];
    for (const item of this.items.values()) {
      const g = this.geometry(item.target);
      const visible = g !== undefined && g.rect.x + g.rect.w > 0 && g.rect.x < width && g.rect.y + g.rect.h > 0 && g.rect.y < height;
      item.el.classList.toggle('vf-off', !visible);
      item.ink.style.display = visible ? '' : 'none';
      if (!visible) continue;
      const size: [number, number] = [item.el.offsetWidth, item.el.offsetHeight];
      const target: Box = { x: g.rect.x - NOTE_TARGET_PAD, y: g.rect.y - NOTE_TARGET_PAD, w: g.rect.w + 2 * NOTE_TARGET_PAD, h: g.rect.h + 2 * NOTE_TARGET_PAD };
      const whole = item.target.index === undefined ? undefined : this.geometry({ nodes: item.target.nodes });
      const avoid = whole === undefined ? common : [...common, whole.rect];
      const options = candidates(target, size, width, height, item.preferred ?? item.side);
      let best = options[0];
      let bestScore = Infinity;
      for (const option of options) {
        const score = overlap(option.box, target) * 20 + [...avoid, ...placed].reduce((sum, o) => sum + overlap(option.box, o), 0) * 3 + option.cost;
        if (score < bestScore) {
          bestScore = score;
          best = option;
        }
      }
      if (item.preferred === undefined) item.side = best.side;
      placed.push(best.box);
      item.el.style.transform = `translate(${best.box.x.toFixed(1)}px, ${best.box.y.toFixed(1)}px)`;
      const center: Point = [best.box.x + best.box.w / 2, best.box.y + best.box.h / 2];
      const aim = g.point ?? clampToBox(center, g.rect);
      const start = exitPoint(best.box, aim, 4);
      const back = Math.hypot(aim[0] - start[0], aim[1] - start[1]) || 1;
      const inset = g.point === undefined ? NOTE_ARROW_INSET : NOTE_ARROW_INSET + 2;
      const end: Point = [aim[0] - ((aim[0] - start[0]) / back) * inset, aim[1] - ((aim[1] - start[1]) / back) * inset];
      arrowPaths(start, end, item.seed, best.box).forEach((d, i) => item.strokes[i].setAttribute('d', d));
    }
  }

  dispose(): void {
    this.svg.remove();
    this.layer.remove();
    this.items.clear();
  }
}
