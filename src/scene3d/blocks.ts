import {
  BoxGeometry,
  BufferGeometry,
  Color,
  CylinderGeometry,
  DoubleSide,
  EdgesGeometry,
  Group,
  InstancedMesh,
  LineBasicMaterial,
  LineLoop,
  LineSegments,
  Matrix4,
  Mesh,
  MeshBasicMaterial,
  MeshLambertMaterial,
  Object3D,
  Vector3,
} from 'three';
import type { LinearRgb, ThemeColors } from '../theme';
import { BOX_HEIGHT, DISC_HEIGHT, DISC_RADIUS, PLATE_THICKNESS, type Block, type Element, type Vec3 } from './layout';
import { LineBatch } from './lines';
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

function seeThroughFill(material: MeshLambertMaterial, seeThrough: boolean): void {
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

function inkMaterial(theme: ThemeColors): LineBasicMaterial {
  return new LineBasicMaterial({ color: new Color(...theme.ink), transparent: true });
}

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
  private readonly breaks: InstancedMesh;
  private readonly outline: LineBatch;
  private readonly breakMaterial: MeshBasicMaterial;
  private readonly geometry: BoxGeometry;
  private readonly breakGeometry: BoxGeometry;
  private readonly material: MeshLambertMaterial;

  constructor(
    count: number,
    private readonly footprint: readonly [number, number],
    theme: ThemeColors,
  ) {
    this.geometry = new BoxGeometry(1, 1, 1);
    this.geometry.translate(0, 0.5, 0);
    this.material = new MeshLambertMaterial({ side: DoubleSide });
    this.mesh = new InstancedMesh(this.geometry, this.material, count);
    this.breakGeometry = new BoxGeometry(BREAK_SIZE, BREAK_THICKNESS, BREAK_SIZE);
    this.breakMaterial = new MeshBasicMaterial({ color: new Color(...theme.surface) });
    this.breaks = new InstancedMesh(this.breakGeometry, this.breakMaterial, count);
    this.outline = new LineBatch(theme.ink, theme.surface);
    this.group.add(this.mesh, this.breaks, this.outline.object);
    this.setTheme(theme);
  }

  setTheme(theme: ThemeColors): void {
    this.breakMaterial.color.setRGB(...theme.surface);
    this.outline.setColor(theme.ink, theme.surface);
  }

  update(elements: readonly Pick<Element, 'x' | 'y' | 'z'>[], data: BarUpdate): void {
    const [w, d] = this.footprint;
    seeThroughFill(this.material, data.seeThrough);
    this.breaks.visible = !data.seeThrough;
    const ink = inkFactor(data.seeThrough);
    const lines = [];
    for (let i = 0; i < elements.length; i++) {
      const { x, y, z } = elements[i];
      const h = data.heights[i];
      const shown = Math.abs(h) > VISIBLE_HEIGHT;
      if (!shown) matrix.makeScale(0, 0, 0).setPosition(x, y, z);
      else matrix.makeScale(w, h, d).setPosition(x, y, z);
      this.mesh.setMatrixAt(i, matrix);
      this.mesh.setColorAt(i, color.setRGB(...data.colors[i]));
      if (shown && data.clipped[i]) matrix.makeScale(1, 1, 1).setPosition(x, y + h * BREAK_POSITION, z);
      else matrix.makeScale(0, 0, 0).setPosition(x, y, z);
      this.breaks.setMatrixAt(i, matrix);
      if (shown) for (const points of boxEdges(x, y, z, w, h, d)) lines.push({ points, opacity: data.outlineOpacity[i] * ink });
    }
    this.mesh.instanceMatrix.needsUpdate = true;
    if (this.mesh.instanceColor) this.mesh.instanceColor.needsUpdate = true;
    this.breaks.instanceMatrix.needsUpdate = true;
    this.mesh.computeBoundingSphere();
    this.outline.setPolylines(lines);
  }

  dispose(): void {
    this.geometry.dispose();
    this.breakGeometry.dispose();
    this.material.dispose();
    this.breakMaterial.dispose();
    this.mesh.dispose();
    this.breaks.dispose();
    this.outline.dispose();
  }
}

function boxEdges(x: number, y: number, z: number, w: number, h: number, d: number): number[][] {
  const x0 = x - w / 2;
  const x1 = x + w / 2;
  const z0 = z - d / 2;
  const z1 = z + d / 2;
  const y1 = y + h;
  const ring = (yy: number) => [x0, yy, z0, x1, yy, z0, x1, yy, z1, x0, yy, z1, x0, yy, z0];
  return [ring(y), ring(y1), [x0, y, z0, x0, y1, z0], [x1, y, z0, x1, y1, z0], [x1, y, z1, x1, y1, z1], [x0, y, z1, x0, y1, z1]];
}

export function barTop(element: Element, height: number): Vec3 {
  return [element.x, element.y + Math.max(height, 0) + ANCHOR_GAP, element.z];
}

export function barBottom(element: Element, height: number): Vec3 {
  return [element.x, element.y + Math.min(height, 0) - ANCHOR_GAP, element.z];
}

export class NodeMesh {
  readonly group = new Group();
  readonly meshes: Mesh[] = [];
  private readonly outlines: LineSegments[] = [];
  private readonly geometry: BufferGeometry;
  private readonly edges: EdgesGeometry;
  private readonly materials: MeshLambertMaterial[] = [];
  private readonly outlineMaterial: LineBasicMaterial;
  readonly height: number;
  readonly radius: number;

  constructor(
    readonly block: Block,
    theme: ThemeColors,
  ) {
    this.height = block.shape === 'disc' ? DISC_HEIGHT : BOX_HEIGHT;
    this.radius = DISC_RADIUS;
    this.geometry = block.shape === 'disc' ? new CylinderGeometry(this.radius, this.radius, DISC_HEIGHT, DISC_SEGMENTS) : new BoxGeometry(block.footprint[0], BOX_HEIGHT, block.footprint[1]);
    this.edges = new EdgesGeometry(this.geometry, 25);
    this.outlineMaterial = inkMaterial(theme);
    for (let i = 0; i < block.elements.length; i++) {
      const material = new MeshLambertMaterial();
      const mesh = new Mesh(this.geometry, material);
      mesh.userData.block = block;
      mesh.userData.element = i;
      const outline = new LineSegments(this.edges, this.outlineMaterial);
      mesh.add(outline);
      this.materials.push(material);
      this.meshes.push(mesh);
      this.outlines.push(outline);
      this.group.add(mesh);
    }
  }

  setTheme(theme: ThemeColors): void {
    this.outlineMaterial.color.setRGB(...theme.ink);
  }

  update(elements: Element[], colors: LinearRgb[], revealed: boolean[], seeThrough: boolean): void {
    elements.forEach((element, i) => {
      this.meshes[i].position.set(element.x, element.y + this.height / 2, element.z);
      this.materials[i].color.setRGB(...colors[i]);
      seeThroughFill(this.materials[i], seeThrough);
    });
    this.outlineMaterial.opacity = (revealed.some((r) => r) ? NODE_OUTLINE_REVEALED : NODE_OUTLINE_HIDDEN) * inkFactor(seeThrough);
  }

  anchorIn(element: Element): Vec3 {
    return [element.x - NODE_ANCHOR_X, element.y + this.height / 2 - NODE_ANCHOR_Y, element.z];
  }

  anchorOut(element: Element): Vec3 {
    return [element.x - NODE_ANCHOR_X, element.y + this.height / 2 + NODE_ANCHOR_Y, element.z];
  }

  dispose(): void {
    this.geometry.dispose();
    this.edges.dispose();
    this.materials.forEach((m) => m.dispose());
    this.outlineMaterial.dispose();
  }
}

export type PlateSource = Pick<Block, 'plate' | 'slabs' | 'chunks' | 'frameHalf' | 'frameRows' | 'axis' | 'elements'>;

export class PlateMesh {
  readonly group = new Group();
  readonly meshes: Mesh[] = [];
  private readonly geometries: BoxGeometry[] = [];
  private readonly edges: EdgesGeometry[] = [];
  private readonly material: MeshLambertMaterial;
  private readonly outlineMaterials: LineBasicMaterial[] = [];
  private readonly faded: boolean[];
  private readonly frames: LineLoop[] = [];
  private readonly frameMaterial: LineBasicMaterial;
  private readonly frameGeometry: BufferGeometry | undefined;
  readonly axis: LineSegments | undefined;
  private readonly axisMaterial: LineBasicMaterial;

  constructor(
    readonly block: PlateSource,
    theme: ThemeColors,
  ) {
    const plates = block.slabs?.map((slab) => slab.plate) ?? block.chunks ?? [block.plate!];
    this.faded = block.slabs?.map((slab) => !slab.used) ?? plates.map(() => false);
    this.material = new MeshLambertMaterial({ color: new Color(...theme.line) });
    for (const plate of plates) {
      const geometry = new BoxGeometry(plate.w, PLATE_THICKNESS, plate.d);
      const mesh = new Mesh(geometry, this.material);
      mesh.position.set(plate.x, plate.y - PLATE_THICKNESS / 2 - FRAME_LIFT, plate.z);
      const edges = new EdgesGeometry(geometry);
      const outlineMaterial = inkMaterial(theme);
      mesh.add(new LineSegments(edges, outlineMaterial));
      this.group.add(mesh);
      this.meshes.push(mesh);
      this.geometries.push(geometry);
      this.edges.push(edges);
      this.outlineMaterials.push(outlineMaterial);
    }
    this.frameMaterial = inkMaterial(theme);
    if (block.frameHalf !== undefined) {
      const [hw, hd] = block.frameHalf;
      this.frameGeometry = new BufferGeometry().setFromPoints([
        new Vector3(-hw, 0, -hd),
        new Vector3(hw, 0, -hd),
        new Vector3(hw, 0, hd),
        new Vector3(-hw, 0, hd),
      ]);
      for (const row of block.frameRows) {
        const frame = new LineLoop(this.frameGeometry, this.frameMaterial);
        frame.position.set(0, row.y + FRAME_LIFT, row.z);
        this.frames.push(frame);
        this.group.add(frame);
      }
    }
    this.axisMaterial = inkMaterial(theme);
    if (block.axis !== undefined) {
      this.axis = new LineSegments(new BufferGeometry(), this.axisMaterial);
      this.group.add(this.axis);
    }
  }

  setTheme(theme: ThemeColors): void {
    this.material.color.setRGB(...theme.line);
    for (const material of this.outlineMaterials) material.color.setRGB(...theme.ink);
    this.frameMaterial.color.setRGB(...theme.ink);
    this.axisMaterial.color.setRGB(...theme.ink);
  }

  setFill(color: LinearRgb): void {
    this.material.color.setRGB(...color);
  }

  update(state: { revealed: boolean; far: boolean; seeThrough: boolean }, frameXs: number[], ticks: number[], scale: number, frameZs?: number[]): void {
    const { revealed, seeThrough } = state;
    const ink = inkFactor(seeThrough);
    seeThroughFill(this.material, seeThrough);
    this.outlineMaterials.forEach((material, i) => {
      material.opacity = (state.far || this.faded[i] ? PLATE_FAR : revealed ? PLATE_REVEALED : PLATE_HIDDEN) * ink;
    });
    this.frameMaterial.opacity = (revealed ? FRAME_REVEALED : FRAME_HIDDEN) * ink;
    this.frames.forEach((frame, r) => {
      frame.position.x = frameXs[Math.min(r, frameXs.length - 1)];
      if (frameZs !== undefined) frame.position.z = frameZs[Math.min(r, frameZs.length - 1)];
    });
    if (this.axis !== undefined && this.block.axis !== undefined && ticks.length > 0) {
      const { x, z } = this.block.axis;
      const y0 = this.block.elements[0].y;
      const lo = Math.min(...ticks) * scale;
      const hi = Math.max(...ticks) * scale;
      const points = [new Vector3(x, y0 + lo, z), new Vector3(x, y0 + hi, z)];
      for (const v of ticks) points.push(new Vector3(x - AXIS_TICK, y0 + v * scale, z), new Vector3(x, y0 + v * scale, z));
      if (this.axis.geometry.getAttribute('position')?.count === points.length) this.axis.geometry.setFromPoints(points);
      else {
        this.axis.geometry.dispose();
        this.axis.geometry = new BufferGeometry().setFromPoints(points);
      }
      this.axisMaterial.opacity = AXIS_REVEALED;
      this.axis.visible = revealed && !seeThrough;
    }
  }

  dispose(): void {
    for (const geometry of this.geometries) geometry.dispose();
    for (const edges of this.edges) edges.dispose();
    this.material.dispose();
    for (const material of this.outlineMaterials) material.dispose();
    this.frameMaterial.dispose();
    this.frameGeometry?.dispose();
    this.axis?.geometry.dispose();
    this.axisMaterial.dispose();
  }
}

export function disposeTree(root: Object3D): void {
  root.traverse((child) => {
    const mesh = child as Mesh;
    if (mesh.geometry) mesh.geometry.dispose();
    const material = mesh.material as { dispose?: () => void } | undefined;
    if (material?.dispose) material.dispose();
  });
}
