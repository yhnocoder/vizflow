import {
  BoxGeometry,
  BufferGeometry,
  Color,
  CylinderGeometry,
  DoubleSide,
  EdgesGeometry,
  Group,
  InstancedMesh,
  Matrix4,
  Mesh,
  MeshBasicMaterial,
  MeshLambertMaterial,
  Object3D,
} from 'three';
import type { LinearRgb, ThemeColors } from '../theme';
import { BOX_HEIGHT, DISC_HEIGHT, DISC_RADIUS, PLATE_THICKNESS, type Block, type Element, type Plate, type Vec3 } from './layout';
import { Segments } from './lines';
import { BREAK_POSITION, BREAK_THICKNESS } from './scales';

const BREAK_SIZE = 0.66;
const PLATE_REVEALED = 0.5;
const PLATE_HIDDEN = 0.2;
const PLATE_FAR = 0.1;
const FRAME_REVEALED = 0.95;
const FRAME_HIDDEN = 0.15;
const FRAME_LIFT = 0.005;
const AXIS_REVEALED = 0.6;
const AXIS_TICK = 0.12;
const NODE_OUTLINE_REVEALED = 0.9;
const NODE_OUTLINE_HIDDEN = 0.3;
const ANCHOR_GAP = 0.01;
const NODE_ANCHOR_X = 0.2;
const NODE_ANCHOR_Y = 0.08;
const DISC_SEGMENTS = 32;
const VISIBLE_HEIGHT = 0.004;
export const SEE_THROUGH_FILL = 0.15;
export const SEE_THROUGH_INK = 0.35;

export function seeThroughFill(material: MeshLambertMaterial, seeThrough: boolean): void {
  if (material.transparent !== seeThrough) material.needsUpdate = true;
  material.transparent = seeThrough;
  material.opacity = seeThrough ? SEE_THROUGH_FILL : 1;
  material.depthWrite = !seeThrough;
}

function inkFactor(seeThrough: boolean): number {
  return seeThrough ? SEE_THROUGH_INK : 1;
}

const matrix = new Matrix4();
const color = new Color();

export interface BarUpdate {
  heights: ArrayLike<number>;
  clipped: ArrayLike<boolean>;
  colors: LinearRgb[];
  outlineOpacity: ArrayLike<number>;
  seeThrough: boolean;
}

export class BarMesh {
  readonly group = new Group();
  readonly mesh: InstancedMesh;
  readonly outline = new Segments();
  private readonly breaks: InstancedMesh;
  private readonly breakMaterial: MeshBasicMaterial;
  private readonly geometry: BoxGeometry;
  private readonly breakGeometry: BoxGeometry;
  private readonly material: MeshLambertMaterial;

  constructor(
    readonly block: Block,
    theme: ThemeColors,
  ) {
    this.geometry = new BoxGeometry(1, 1, 1);
    this.geometry.translate(0, 0.5, 0);
    this.material = new MeshLambertMaterial({ side: DoubleSide });
    this.mesh = new InstancedMesh(this.geometry, this.material, block.elements.length);
    this.mesh.userData.block = block;
    this.breakGeometry = new BoxGeometry(BREAK_SIZE, BREAK_THICKNESS, BREAK_SIZE);
    this.breakMaterial = new MeshBasicMaterial({ color: new Color(...theme.surface) });
    this.breaks = new InstancedMesh(this.breakGeometry, this.breakMaterial, block.elements.length);
    this.group.add(this.mesh, this.breaks);
    this.setTheme(theme);
  }

  setTheme(theme: ThemeColors): void {
    this.breakMaterial.color.setRGB(...theme.surface);
  }

  update(elements: Element[], data: BarUpdate): void {
    const [w, d] = this.block.footprint;
    seeThroughFill(this.material, data.seeThrough);
    const ink = inkFactor(data.seeThrough);
    let broken = false;
    this.outline.clear();
    for (let i = 0; i < elements.length; i++) {
      const { x, y, z } = elements[i];
      const h = data.heights[i];
      const shown = Math.abs(h) > VISIBLE_HEIGHT;
      if (!shown) matrix.makeScale(0, 0, 0).setPosition(x, y, z);
      else matrix.makeScale(w, h, d).setPosition(x, y, z);
      this.mesh.setMatrixAt(i, matrix);
      this.mesh.setColorAt(i, color.setRGB(...data.colors[i]));
      if (shown && data.clipped[i]) {
        broken = true;
        matrix.makeScale(1, 1, 1).setPosition(x, y + h * BREAK_POSITION, z);
      } else matrix.makeScale(0, 0, 0).setPosition(x, y, z);
      this.breaks.setMatrixAt(i, matrix);
      if (shown) addBox(this.outline, x, y, z, w, h, d, data.outlineOpacity[i] * ink);
    }
    this.breaks.visible = broken && !data.seeThrough;
    this.mesh.instanceMatrix.needsUpdate = true;
    if (this.mesh.instanceColor) this.mesh.instanceColor.needsUpdate = true;
    this.breaks.instanceMatrix.needsUpdate = true;
    this.mesh.computeBoundingSphere();
  }

  dispose(): void {
    this.geometry.dispose();
    this.breakGeometry.dispose();
    this.material.dispose();
    this.breakMaterial.dispose();
    this.mesh.dispose();
    this.breaks.dispose();
  }
}

function addBox(out: Segments, x: number, y: number, z: number, w: number, h: number, d: number, opacity: number): void {
  const x0 = x - w / 2;
  const x1 = x + w / 2;
  const z0 = z - d / 2;
  const z1 = z + d / 2;
  const y1 = y + h;
  for (const yy of [y, y1]) {
    out.add(x0, yy, z0, x1, yy, z0, opacity);
    out.add(x1, yy, z0, x1, yy, z1, opacity);
    out.add(x1, yy, z1, x0, yy, z1, opacity);
    out.add(x0, yy, z1, x0, yy, z0, opacity);
  }
  out.add(x0, y, z0, x0, y1, z0, opacity);
  out.add(x1, y, z0, x1, y1, z0, opacity);
  out.add(x1, y, z1, x1, y1, z1, opacity);
  out.add(x0, y, z1, x0, y1, z1, opacity);
}

export function barTop(element: Element, height: number): Vec3 {
  return [element.x, element.y + Math.max(height, 0) + ANCHOR_GAP, element.z];
}

export function barBottom(element: Element, height: number): Vec3 {
  return [element.x, element.y + Math.min(height, 0) - ANCHOR_GAP, element.z];
}

export class NodeMesh {
  readonly group = new Group();
  readonly mesh: InstancedMesh;
  readonly outline = new Segments();
  private readonly geometry: BufferGeometry;
  private readonly edges: Float32Array;
  private readonly material = new MeshLambertMaterial();
  readonly height: number;
  readonly radius: number;

  constructor(readonly block: Block) {
    this.height = block.shape === 'disc' ? DISC_HEIGHT : BOX_HEIGHT;
    this.radius = DISC_RADIUS;
    this.geometry = block.shape === 'disc' ? new CylinderGeometry(this.radius, this.radius, DISC_HEIGHT, DISC_SEGMENTS) : new BoxGeometry(block.footprint[0], BOX_HEIGHT, block.footprint[1]);
    const edges = new EdgesGeometry(this.geometry, 25);
    this.edges = Float32Array.from(edges.getAttribute('position').array);
    edges.dispose();
    this.mesh = new InstancedMesh(this.geometry, this.material, block.elements.length);
    this.mesh.userData.block = block;
    this.group.add(this.mesh);
  }

  update(elements: Element[], colors: LinearRgb[], revealed: boolean[], seeThrough: boolean): void {
    seeThroughFill(this.material, seeThrough);
    const opacity = (revealed.some((r) => r) ? NODE_OUTLINE_REVEALED : NODE_OUTLINE_HIDDEN) * inkFactor(seeThrough);
    this.outline.clear();
    elements.forEach((element, i) => {
      const y = element.y + this.height / 2;
      this.mesh.setMatrixAt(i, matrix.makeTranslation(element.x, y, element.z));
      this.mesh.setColorAt(i, color.setRGB(...colors[i]));
      this.outline.addShifted(this.edges, element.x, y, element.z, opacity);
    });
    this.mesh.instanceMatrix.needsUpdate = true;
    if (this.mesh.instanceColor) this.mesh.instanceColor.needsUpdate = true;
    this.mesh.computeBoundingSphere();
  }

  anchorIn(element: Element): Vec3 {
    return [element.x - NODE_ANCHOR_X, element.y + this.height / 2 - NODE_ANCHOR_Y, element.z];
  }

  anchorOut(element: Element): Vec3 {
    return [element.x - NODE_ANCHOR_X, element.y + this.height / 2 + NODE_ANCHOR_Y, element.z];
  }

  dispose(): void {
    this.geometry.dispose();
    this.material.dispose();
    this.mesh.dispose();
  }
}

export class PlateMesh {
  readonly plates: readonly Plate[];
  readonly ink = new Segments();
  readonly axis = new Segments();
  seeThrough = false;
  private readonly faded: boolean[];

  constructor(readonly block: Block) {
    this.plates = block.slabs?.map((slab) => slab.plate) ?? block.chunks ?? [block.plate!];
    this.faded = block.slabs?.map((slab) => !slab.used) ?? this.plates.map(() => false);
  }

  update(state: { revealed: boolean; far: boolean; seeThrough: boolean }, frameXs: number[], ticks: number[], scale: number, frameZs?: number[]): void {
    const { revealed, seeThrough } = state;
    this.seeThrough = seeThrough;
    const ink = inkFactor(seeThrough);
    this.ink.clear();
    this.plates.forEach((plate, i) => {
      const opacity = (state.far || this.faded[i] ? PLATE_FAR : revealed ? PLATE_REVEALED : PLATE_HIDDEN) * ink;
      addBox(this.ink, plate.x, plate.y - PLATE_THICKNESS - FRAME_LIFT, plate.z, plate.w, PLATE_THICKNESS, plate.d, opacity);
    });
    if (this.block.frameHalf !== undefined) {
      const [hw, hd] = this.block.frameHalf;
      const opacity = (revealed ? FRAME_REVEALED : FRAME_HIDDEN) * ink;
      this.block.frameRows.forEach((row, r) => {
        const x = frameXs[Math.min(r, frameXs.length - 1)];
        const z = frameZs === undefined ? row.z : frameZs[Math.min(r, frameZs.length - 1)];
        const y = row.y + FRAME_LIFT;
        this.ink.add(x - hw, y, z - hd, x + hw, y, z - hd, opacity);
        this.ink.add(x + hw, y, z - hd, x + hw, y, z + hd, opacity);
        this.ink.add(x + hw, y, z + hd, x - hw, y, z + hd, opacity);
        this.ink.add(x - hw, y, z + hd, x - hw, y, z - hd, opacity);
      });
    }
    this.axis.clear();
    if (this.block.axis !== undefined && ticks.length > 0 && revealed && !seeThrough) {
      const { x, z } = this.block.axis;
      const y0 = this.block.elements[0].y;
      this.axis.add(x, y0 + Math.min(...ticks) * scale, z, x, y0 + Math.max(...ticks) * scale, z, AXIS_REVEALED);
      for (const v of ticks) this.axis.add(x - AXIS_TICK, y0 + v * scale, z, x, y0 + v * scale, z, AXIS_REVEALED);
    }
  }
}

export function plateFill(plate: Plate, target: Matrix4): Matrix4 {
  return target.makeScale(plate.w, 1, plate.d).setPosition(plate.x, plate.y - PLATE_THICKNESS / 2 - FRAME_LIFT, plate.z);
}

export function disposeTree(root: Object3D): void {
  root.traverse((child) => {
    const mesh = child as Mesh;
    if (mesh.geometry) mesh.geometry.dispose();
    const material = mesh.material as { dispose?: () => void } | undefined;
    if (material?.dispose) material.dispose();
  });
}
