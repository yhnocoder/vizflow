import { CubicBezierCurve3, Color, Mesh, MeshBasicMaterial, TubeGeometry, Vector3 } from 'three';
import type { Node } from '../autograd';
import type { FlowEdge } from '../explain';
import type { ThemeColors } from '../theme';
import type { SceneController } from './controller';
import type { DrawnEdge } from './edges';
import { TokenFrame, elementsRect, relatedOpacity, sameToken, tokenAxes, tokenRect } from './hover';
import type { Element, Vec3 } from './layout';
import { LineBatch } from './lines';
import { type Renderable, elementOf, renderableOf } from './renderable';

export const EDGE_HOVER_DISTANCE = 6;
export const LONG_EDGE = 4.2;
export const EDGE_BOW = 0.9;

const CURVE_SEGMENTS = 14;
const CURVE_TANGENT = 0.45;
const BOW_TANGENT = 0.3;
const HIGHLIGHT_RADIUS = 0.03;
const HIGHLIGHT_SEGMENTS = 30;
const HIGHLIGHT_SIDES = 6;

const tmpVector = new Vector3();

export function controlPoints(a: Vec3, b: Vec3, flat: boolean, bow?: number): [Vec3, Vec3, Vec3, Vec3] {
  if (flat) {
    const lift = (k: number): Vec3 => [a[0] + (b[0] - a[0]) * k, a[1] + (b[1] - a[1]) * k + (bow ?? 0), a[2] + (b[2] - a[2]) * k];
    return [a, lift(1 / 3), lift(2 / 3), b];
  }
  const d = b[1] - a[1];
  const z = bow ?? (d > LONG_EDGE ? EDGE_BOW : 0);
  if (z !== 0) return [a, [a[0], a[1] + d * BOW_TANGENT, a[2] + z], [b[0], b[1] - d * BOW_TANGENT, b[2] + z], b];
  return [a, [a[0], a[1] + d * CURVE_TANGENT, a[2]], [b[0], b[1] - d * CURVE_TANGENT, b[2]], b];
}

function sampleCurve(a: Vec3, b: Vec3, flat: boolean, bow?: number): number[] {
  const c = controlPoints(a, b, flat, bow);
  const points: number[] = [];
  for (let s = 0; s <= CURVE_SEGMENTS; s++) {
    const t = s / CURVE_SEGMENTS;
    const r = 1 - t;
    const w = [r * r * r, 3 * r * r * t, 3 * r * t * t, t * t * t];
    for (let axis = 0; axis < 3; axis++) points.push(w[0] * c[0][axis] + w[1] * c[1][axis] + w[2] * c[2][axis] + w[3] * c[3][axis]);
  }
  return points;
}

function segmentDistance(px: number, py: number, ax: number, ay: number, bx: number, by: number): number {
  const dx = bx - ax;
  const dy = by - ay;
  const len = dx * dx + dy * dy;
  const t = len === 0 ? 0 : Math.max(0, Math.min(1, ((px - ax) * dx + (py - ay) * dy) / len));
  return Math.hypot(px - (ax + t * dx), py - (ay + t * dy));
}

export class EdgeLines {
  private readonly lines: LineBatch;
  private readonly highlightMaterial: MeshBasicMaterial;
  private highlight: Mesh | undefined;
  private screenCached = -1;
  private readonly tokenFrame: TokenFrame;
  private readonly sourceFrames: TokenFrame[] = [];
  private readonly tokenAxes: Map<Node, number>;
  edges: DrawnEdge[] = [];
  hover: DrawnEdge | undefined;
  bar: { renderable: Renderable; index: number } | undefined;

  constructor(private readonly c: SceneController) {
    this.lines = new LineBatch(c.theme.ink, c.theme.surface);
    c.scene.add(this.lines.object);
    this.tokenAxes = tokenAxes(c.g.nodes);
    this.tokenFrame = new TokenFrame(c.theme);
    c.scene.add(this.tokenFrame.group);
    this.highlightMaterial = new MeshBasicMaterial({ color: new Color(...c.theme.ink) });
  }

  redraw(): void {
    for (const d of this.edges) {
      const [a, b] = d.endpoints();
      d.points = sampleCurve(a, b, d.flat, d.bow);
    }
    this.restyle();
    this.updateHighlight();
  }

  private touchesBar(d: DrawnEdge): boolean {
    const bar = this.bar;
    if (bar === undefined) return false;
    const block = bar.renderable.block;
    const element = block.elements[bar.index];
    const touches = (end: FlowEdge['from']) => 'index' in end && end.node === block.node && end.index.every((v, i) => v === element.index[i]);
    return touches(d.edge.from) || touches(d.edge.to);
  }

  private restyle(): void {
    const bar = this.bar;
    const block = bar?.renderable.block;
    const element = bar === undefined ? undefined : block!.elements[bar.index];
    const related = this.edges.map((d) => element !== undefined && this.touchesBar(d));
    this.lines.setPolylines(
      this.edges.flatMap((d, i) => (d.backRow === true && !related[i] ? [] : [{ points: d.points, opacity: element === undefined ? d.style.opacity : relatedOpacity(d.style.opacity, related[i]), dashed: d.edge.op.op === 'alias' }])),
    );
    const axis = block === undefined ? undefined : this.tokenAxes.get(block.node);
    const token = block !== undefined && element !== undefined && axis !== undefined;
    this.tokenFrame.set(token ? tokenRect(block, element, axis) : undefined);
    this.setSourceFrames(token ? this.tableSources(block.node, sameToken(block, element, axis)) : []);
  }

  private tableSources(node: Node, members: readonly Element[]): { renderable: Renderable; elements: Element[] }[] {
    const wanted = new Set(members.map((e) => e.index.join(',')));
    const found = new Map<Renderable, Set<Element>>();
    for (const d of this.edges) {
      const { from, to } = d.edge;
      if (!('index' in from) || !('index' in to) || to.node !== node || !wanted.has(to.index.join(','))) continue;
      const source = renderableOf(this.c, from.node);
      if (source === undefined || source.block.table === undefined) continue;
      const set = found.get(source) ?? found.set(source, new Set()).get(source)!;
      set.add(elementOf(source.block, from.node, from.index));
    }
    return [...found].map(([renderable, elements]) => ({ renderable, elements: [...elements] }));
  }

  private setSourceFrames(sources: { renderable: Renderable; elements: Element[] }[]): void {
    while (this.sourceFrames.length < sources.length) {
      const frame = new TokenFrame(this.c.theme);
      this.c.scene.add(frame.group);
      this.sourceFrames.push(frame);
    }
    this.sourceFrames.forEach((frame, i) => frame.set(i < sources.length ? elementsRect(sources[i].renderable.block, sources[i].elements) : undefined));
  }

  private updateHighlight(): void {
    const scene = this.c.scene;
    if (this.highlight !== undefined) {
      scene.remove(this.highlight);
      this.highlight.geometry.dispose();
      this.highlight = undefined;
    }
    const d = this.hover;
    if (d === undefined) return;
    const [a, b] = d.endpoints();
    const [p0, p1, p2, p3] = controlPoints(a, b, d.flat, d.bow).map((p) => new Vector3(...p));
    const curve = new CubicBezierCurve3(p0, p1, p2, p3);
    this.highlight = new Mesh(new TubeGeometry(curve, HIGHLIGHT_SEGMENTS, HIGHLIGHT_RADIUS / Math.sqrt(this.c.camera.zoom), HIGHLIGHT_SIDES, false), this.highlightMaterial);
    scene.add(this.highlight);
  }

  edgeAt(x: number, y: number): DrawnEdge | undefined {
    const c = this.c;
    if (c.state.dir !== 'backward') return undefined;
    if (this.screenCached !== c.screenVersion) {
      for (const d of this.edges) {
        d.screen = [];
        for (let i = 0; i < d.points.length; i += 3) {
          tmpVector.set(d.points[i], d.points[i + 1], d.points[i + 2]).project(c.camera);
          d.screen.push(((tmpVector.x + 1) / 2) * c.width, ((1 - tmpVector.y) / 2) * c.height);
        }
      }
      this.screenCached = c.screenVersion;
    }
    let best: DrawnEdge | undefined;
    let bestDistance = EDGE_HOVER_DISTANCE;
    for (const d of this.edges) {
      if (!c.state.revealed(d.fromNode) || (d.backRow === true && !this.touchesBar(d))) continue;
      for (let i = 0; i + 3 < d.screen.length; i += 2) {
        const distance = segmentDistance(x, y, d.screen[i], d.screen[i + 1], d.screen[i + 2], d.screen[i + 3]);
        if (distance < bestDistance) {
          bestDistance = distance;
          best = d;
        }
      }
    }
    return best;
  }

  setHover(d: DrawnEdge | undefined): void {
    if (this.hover === d) return;
    this.hover = d;
    this.c.markGeometryDirty();
    this.c.requestFrame();
  }

  setBar(found: { renderable: Renderable; index: number } | undefined): void {
    if (found?.renderable === this.bar?.renderable && found?.index === this.bar?.index) return;
    this.bar = found;
    this.restyle();
    this.c.requestFrame();
  }

  setTheme(theme: ThemeColors): void {
    this.lines.setColor(theme.ink, theme.surface);
    this.tokenFrame.setTheme(theme);
    for (const frame of this.sourceFrames) frame.setTheme(theme);
    this.highlightMaterial.color.setRGB(...theme.ink);
  }

  dispose(): void {
    this.lines.dispose();
    this.tokenFrame.dispose();
    for (const frame of this.sourceFrames) frame.dispose();
    this.highlight?.geometry.dispose();
    this.highlightMaterial.dispose();
  }
}
