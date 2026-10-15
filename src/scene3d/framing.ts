import { Matrix4, Vector3 } from 'three';
import type { SceneController } from './controller';
import { LABEL_SAFE_MARGIN, type Label, anchorOffset } from './labels';
import { DISC_RADIUS, type Vec3 } from './layout';
import { braceTip } from './modules';
import type { Renderable } from './renderable';

export const CAMERA_DISTANCE = 150;

const FRAME_SEARCH_STEPS = 30;

const tmpVector = new Vector3();

interface FrameItem {
  x: number;
  y: number;
  left: number;
  right: number;
  up: number;
  down: number;
}

interface Frustum {
  left: number;
  right: number;
  top: number;
  bottom: number;
}

export function boundsOf(points: Vec3[]): { min: number[]; max: number[] } {
  const min = [Infinity, Infinity, Infinity];
  const max = [-Infinity, -Infinity, -Infinity];
  for (const p of points) for (let axis = 0; axis < 3; axis++) {
    min[axis] = Math.min(min[axis], p[axis]);
    max[axis] = Math.max(max[axis], p[axis]);
  }
  return { min, max };
}

function span(items: FrameItem[], k: number, at: (i: FrameItem) => number, before: (i: FrameItem) => number, after: (i: FrameItem) => number): { lo: number; hi: number; size: number } {
  let lo = Infinity;
  let hi = -Infinity;
  for (const item of items) {
    lo = Math.min(lo, at(item) * k - before(item));
    hi = Math.max(hi, at(item) * k + after(item));
  }
  return { lo, hi, size: hi - lo };
}

export function framingPoints(renderables: Renderable[]): Vec3[] {
  const points: Vec3[] = [];
  for (const renderable of renderables) {
    const { block } = renderable;
    const [w, d] = block.footprint;
    const values = block.node.value.data;
    const grad = block.node.grad;
    let high = 0;
    let low = 0;
    for (let i = 0; i < values.length; i++) {
      for (const h of [values[i] * renderable.scale.forward, grad === null ? 0 : grad.data[i] * renderable.scale.backward]) {
        high = Math.max(high, Math.min(h, renderable.cap));
        low = Math.min(low, Math.max(h, -renderable.cap));
      }
    }
    const top = renderable.nodes === undefined ? high : renderable.nodes.height;
    const bottom = renderable.nodes === undefined ? low : 0;
    const radius = block.shape === 'disc' ? DISC_RADIUS : w / 2;
    const depth = block.shape === 'disc' ? DISC_RADIUS : d / 2;
    for (const e of block.elements) {
      for (const dx of [-radius, radius]) for (const dz of [-depth, depth]) points.push([e.x + dx, e.y + bottom, e.z + dz], [e.x + dx, e.y + top, e.z + dz]);
    }
    if (block.plate !== undefined) {
      const p = block.plate;
      for (const dx of [-p.w / 2, p.w / 2]) for (const dz of [-p.d / 2, p.d / 2]) points.push([p.x + dx, p.y, p.z + dz]);
    }
    points.push([block.title.x, block.title.y, block.title.z]);
    for (const text of [...block.classLabels, ...block.rowLabels]) points.push([text.x, text.y, text.z]);
    if (block.axis !== undefined) points.push([block.axis.x - 0.5, block.elements[0].y, block.axis.z]);
    points.push(...block.extent);
  }
  return points;
}

export function graphPoints(c: SceneController): Vec3[] {
  const points = framingPoints(c.renderables);
  for (const brace of c.moduleBraces.braces) points.push([brace.x, brace.y0, brace.z], [brace.x, brace.y1, brace.z], braceTip(brace));
  return points;
}

export function allPoints(c: SceneController): Vec3[] {
  if (c.compactView?.active === true) return c.compactView.points();
  return [...graphPoints(c), ...(c.kvPool?.points() ?? [])];
}

export function center(points: Vec3[]): Vector3 {
  const { min, max } = boundsOf(points);
  return new Vector3((min[0] + max[0]) / 2, (min[1] + max[1]) / 2, (min[2] + max[2]) / 2);
}

function chromeRight(c: SceneController): number {
  const panel = c.root.querySelector<HTMLElement>('.vf-edit[open]');
  return panel === null ? 0 : c.width - panel.offsetLeft;
}

function chromeBottom(c: SceneController): number {
  let bottom = 0;
  for (const el of c.root.querySelectorAll<HTMLElement>('.vf-key, .vf-hint')) {
    if (el.offsetHeight > 0) bottom = Math.max(bottom, el.offsetTop + el.offsetHeight);
  }
  return bottom;
}

function computeFrame(c: SceneController, points: Vec3[], labels: readonly Label[], middle: Vector3, direction: Vector3): Frustum {
  const eye = middle.clone().addScaledVector(direction, CAMERA_DISTANCE);
  const view = new Matrix4().lookAt(eye, middle, c.camera.up).setPosition(eye).invert();
  const marginX = LABEL_SAFE_MARGIN;
  const marginY = LABEL_SAFE_MARGIN;
  const items: FrameItem[] = points.map((p) => {
    tmpVector.set(...p).applyMatrix4(view);
    return { x: tmpVector.x, y: tmpVector.y, left: marginX, right: marginX, up: marginY, down: marginY };
  });
  const geometry = { x: span(items, 1, (i) => i.x, () => 0, () => 0), y: span(items, 1, (i) => i.y, () => 0, () => 0) };
  for (const label of labels) {
    if (label.pos === undefined) continue;
    const [w, h] = label.frameSize();
    if (w === 0 && h === 0) continue;
    const [ax, ay] = anchorOffset(label.anchor);
    tmpVector.set(...label.pos).applyMatrix4(view);
    const [dx, dy] = label.offset ?? [0, 0];
    items.push({
      x: tmpVector.x,
      y: tmpVector.y,
      left: LABEL_SAFE_MARGIN - ax * w - dx,
      right: LABEL_SAFE_MARGIN + (ax + 1) * w + dx,
      up: LABEL_SAFE_MARGIN - ay * h - dy,
      down: (ay + 1) * h + dy + LABEL_SAFE_MARGIN,
    });
  }
  const reserve = chromeBottom(c);
  const availableH = Math.max(c.height - reserve, c.height / 2);
  const availableW = Math.max(c.width - chromeRight(c), c.width / 2);
  const spanX = (k: number) => span(items, k, (i) => i.x, (i) => i.left, (i) => i.right);
  const spanY = (k: number) => span(items, k, (i) => -i.y, (i) => i.up, (i) => i.down);
  const fits = (k: number) => spanX(k).size <= availableW && spanY(k).size <= availableH;
  let k = Math.min((availableW - 2 * marginX) / geometry.x.size, (availableH - 2 * marginY) / geometry.y.size);
  if (!fits(k) && fits(0)) {
    let lo = 0;
    let hi = k;
    for (let i = 0; i < FRAME_SEARCH_STEPS; i++) {
      const mid = (lo + hi) / 2;
      if (fits(mid)) lo = mid;
      else hi = mid;
    }
    k = lo;
  }
  const unit = 1 / k;
  const sx = spanX(k);
  const sy = spanY(k);
  const cx = ((sx.lo + sx.hi) / 2) * unit;
  const cy = -((sy.lo + sy.hi) / 2) * unit;
  const left = cx - (availableW * unit) / 2;
  const top = cy + (reserve + availableH / 2) * unit;
  return { left, right: left + c.width * unit, top, bottom: top - c.height * unit };
}

export function frameAll(c: SceneController): void {
  const points = allPoints(c);
  const frame = computeFrame(c, points, c.labels.labels.filter((label) => c.labels.allows(label)), center(points), c.rig.direction);
  const camera = c.camera;
  camera.left = frame.left;
  camera.right = frame.right;
  camera.top = frame.top;
  camera.bottom = frame.bottom;
  camera.updateProjectionMatrix();
}
