import { renderText } from '../formula';
import type { NoteSide } from '../steps';
import { type Box, intersects } from './avoid';

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
export const NOTE_PLACEHOLDER = '写一条笔记，回车保存';

const SVG = 'http://www.w3.org/2000/svg';

export interface NoteView {
  key: string;
  text: string;
  side?: NoteSide;
  reader: boolean;
  draft?: boolean;
}

export interface NoteGeometry {
  rect: Box;
  point?: [number, number];
}

interface NoteItem {
  view: NoteView;
  el: HTMLDivElement;
  ink: SVGGElement;
  strokes: SVGPathElement[];
  seed: number;
  side: NoteSide | undefined;
}

type Point = [number, number];

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

function inside(p: Point, box: Box | undefined): boolean {
  return box !== undefined && p[0] > box.x && p[0] < box.x + box.w && p[1] > box.y && p[1] < box.y + box.h;
}

export function arrowPaths(start: Point, end: Point, seed: number, from?: Box): string[] {
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

function candidates(geometry: NoteGeometry, size: [number, number], width: number, height: number, preferred: NoteSide | undefined): { side: NoteSide; box: Box; cost: number }[] {
  const { rect } = geometry;
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

export class NoteLayer {
  readonly svg: SVGSVGElement;
  readonly root: HTMLDivElement;
  private readonly items = new Map<string, NoteItem>();
  private focus: HTMLInputElement | undefined;

  constructor(
    parent: HTMLElement,
    private readonly handlers: { remove(key: string): void; save(key: string, text: string): void; cancel(key: string): void },
  ) {
    this.svg = document.createElementNS(SVG, 'svg');
    this.svg.setAttribute('class', 'vf-note-ink');
    this.svg.setAttribute('aria-hidden', 'true');
    this.root = document.createElement('div');
    this.root.className = 'vf-notes';
    parent.append(this.svg, this.root);
  }

  sync(views: NoteView[]): boolean {
    let changed = false;
    const keep = new Set(views.map((v) => v.key));
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
    for (const view of views) {
      const existing = this.items.get(view.key);
      if (existing !== undefined && existing.view.text === view.text && existing.view.draft === view.draft) continue;
      if (existing !== undefined) {
        existing.el.remove();
        existing.ink.remove();
      }
      this.items.set(view.key, this.create(view, existing !== undefined));
      changed = true;
    }
    return changed;
  }

  private create(view: NoteView, instant: boolean): NoteItem {
    const el = document.createElement('div');
    el.className = `vf-note${view.reader ? ' vf-note-reader' : ''}`;
    el.dataset.note = view.key;
    if (view.draft) {
      const input = document.createElement('input');
      input.className = 'vf-note-input';
      input.type = 'text';
      input.placeholder = NOTE_PLACEHOLDER;
      input.setAttribute('aria-label', '笔记内容');
      input.addEventListener('keydown', (ev) => {
        ev.stopPropagation();
        if (ev.key === 'Enter' && input.value.trim() !== '') this.handlers.save(view.key, input.value.trim());
        else if (ev.key === 'Escape') this.handlers.cancel(view.key);
      });
      input.addEventListener('blur', () => {
        if (input.value.trim() === '') this.handlers.cancel(view.key);
        else this.handlers.save(view.key, input.value.trim());
      });
      el.append(input);
      this.focus = input;
    } else {
      const body = document.createElement('div');
      body.className = 'vf-note-text';
      renderText(view.text, body);
      el.append(body);
      if (view.reader) {
        const remove = document.createElement('button');
        remove.type = 'button';
        remove.className = 'vf-note-x';
        remove.textContent = '×';
        remove.setAttribute('aria-label', '删除这条笔记');
        remove.addEventListener('click', () => this.handlers.remove(view.key));
        el.append(remove);
      }
    }
    const ink = document.createElementNS(SVG, 'g');
    ink.setAttribute('class', 'vf-note-arrow');
    const strokes = Array.from({ length: 6 }, (_, i) => {
      const path = document.createElementNS(SVG, 'path');
      path.setAttribute('pathLength', '1');
      path.setAttribute('class', i < 2 ? 'vf-note-shaft' : 'vf-note-head');
      ink.append(path);
      return path;
    });
    el.classList.add('vf-off');
    this.root.append(el);
    this.svg.append(ink);
    const item: NoteItem = { view, el, ink, strokes, seed: hash(view.key), side: undefined };
    const show = () => {
      el.classList.add('vf-note-in');
      ink.classList.add('vf-note-in');
    };
    if (instant) {
      el.classList.add('vf-note-still');
      ink.classList.add('vf-note-still');
    }
    requestAnimationFrame(() => requestAnimationFrame(show));
    return item;
  }

  place(geometry: (key: string) => NoteGeometry | undefined, width: number, height: number, obstacles: (key: string) => Box[]): void {
    this.svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
    const placed: Box[] = [];
    for (const [key, item] of this.items) {
      const g = geometry(key);
      const visible = g !== undefined && g.rect.x + g.rect.w > 0 && g.rect.x < width && g.rect.y + g.rect.h > 0 && g.rect.y < height;
      item.el.classList.toggle('vf-off', !visible);
      item.ink.style.display = visible ? '' : 'none';
      if (!visible) continue;
      const size: [number, number] = [item.el.offsetWidth, item.el.offsetHeight];
      const target: Box = { x: g.rect.x - NOTE_TARGET_PAD, y: g.rect.y - NOTE_TARGET_PAD, w: g.rect.w + 2 * NOTE_TARGET_PAD, h: g.rect.h + 2 * NOTE_TARGET_PAD };
      const avoid = obstacles(key);
      const options = candidates({ ...g, rect: target }, size, width, height, item.view.side ?? item.side);
      let best = options[0];
      let bestScore = Infinity;
      for (const option of options) {
        const score = overlap(option.box, target) * 20 + [...avoid, ...placed].reduce((sum, o) => sum + overlap(option.box, o), 0) * 3 + option.cost;
        if (score < bestScore) {
          bestScore = score;
          best = option;
        }
      }
      if (item.view.side === undefined) item.side = best.side;
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
    const input = this.focus;
    if (input !== undefined && input.isConnected && !input.closest('.vf-off')) {
      this.focus = undefined;
      input.focus({ preventScroll: true });
    }
  }

  keys(): string[] {
    return [...this.items.keys()];
  }

  dispose(): void {
    this.svg.remove();
    this.root.remove();
    this.items.clear();
  }
}
