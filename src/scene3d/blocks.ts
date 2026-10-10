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

const BREAK_MARGIN = 0.04;
const PLATE_REVEALED = 0.5;
const PLATE_HIDDEN = 0.15;
const FRAME_REVEALED = 0.95;
const FRAME_HIDDEN = 0.15;
const FRAME_LIFT = 0.005;
const AXIS_REVEALED = 0.6;
const AXIS_HIDDEN = 0.2;
const AXIS_TICK = 0.12;
const NODE_OUTLINE_REVEALED = 0.9;
const NODE_OUTLINE_HIDDEN = 0.3;
const ANCHOR_GAP = 0.01;

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
    readonly block: Block,
    theme: ThemeColors,
  ) {
    const [w, d] = block.footprint;
    this.geometry = new BoxGeometry(1, 1, 1);
    this.geometry.translate(0, 0.5, 0);
    this.material = new MeshLambertMaterial({ side: DoubleSide });
    this.mesh = new InstancedMesh(this.geometry, this.material, block.elements.length);
    this.mesh.userData.block = block;
    this.breakGeometry = new BoxGeometry(w + BREAK_MARGIN, BREAK_THICKNESS, d + BREAK_MARGIN);
    this.breakMaterial = new MeshBasicMaterial({ color: new Color(...theme.surface) });
    this.breaks = new InstancedMesh(this.breakGeometry, this.breakMaterial, block.elements.length);
    this.outline = new LineBatch(theme.ink);
    this.group.add(this.mesh, this.breaks, this.outline.object);
    this.setTheme(theme);
  }

  setTheme(theme: ThemeColors): void {
    this.breakMaterial.color.setRGB(...theme.surface);
    this.outline.setColor(theme.ink);
  }

  update(elements: Element[], data: BarUpdate): void {
    const [w, d] = this.block.footprint;
    const lines = [];
    for (let i = 0; i < elements.length; i++) {
      const { x, y, z } = elements[i];
      const h = data.heights[i];
      if (h === 0) matrix.makeScale(0, 0, 0).setPosition(x, y, z);
      else matrix.makeScale(w, h, d).setPosition(x, y, z);
      this.mesh.setMatrixAt(i, matrix);
      this.mesh.setColorAt(i, color.setRGB(...data.colors[i]));
      if (data.clipped[i]) matrix.makeScale(1, 1, 1).setPosition(x, y + h * BREAK_POSITION, z);
      else matrix.makeScale(0, 0, 0).setPosition(x, y, z);
      this.breaks.setMatrixAt(i, matrix);
      for (const points of boxEdges(x, y, z, w, h, d)) lines.push({ points, opacity: data.outlineOpacity[i], dashed: false });
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
  if (h === 0) return [ring(y)];
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

  constructor(
    readonly block: Block,
    theme: ThemeColors,
  ) {
    this.height = block.shape === 'disc' ? DISC_HEIGHT : BOX_HEIGHT;
    this.geometry = block.shape === 'disc' ? new CylinderGeometry(DISC_RADIUS, DISC_RADIUS, DISC_HEIGHT, 48) : new BoxGeometry(block.footprint[0], BOX_HEIGHT, block.footprint[1]);
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

  update(elements: Element[], colors: LinearRgb[], revealed: boolean[]): void {
    elements.forEach((element, i) => {
      this.meshes[i].position.set(element.x, element.y + this.height / 2, element.z);
      this.materials[i].color.setRGB(...colors[i]);
    });
    this.outlineMaterial.opacity = revealed.some((r) => r) ? NODE_OUTLINE_REVEALED : NODE_OUTLINE_HIDDEN;
  }

  anchorIn(element: Element): Vec3 {
    const cy = element.y + this.height / 2;
    return this.block.shape === 'disc' ? [element.x - 0.4, cy - 0.14, element.z] : [element.x, cy - 0.25, element.z];
  }

  anchorOut(element: Element): Vec3 {
    const cy = element.y + this.height / 2;
    return this.block.shape === 'disc' ? [element.x - 0.3, cy + 0.14, element.z] : [element.x, cy + 0.25, element.z];
  }

  dispose(): void {
    this.geometry.dispose();
    this.edges.dispose();
    this.materials.forEach((m) => m.dispose());
    this.outlineMaterial.dispose();
  }
}

export class PlateMesh {
  readonly group = new Group();
  private readonly mesh: Mesh;
  private readonly geometry: BoxGeometry;
  private readonly edges: EdgesGeometry;
  private readonly material: MeshLambertMaterial;
  private readonly outlineMaterial: LineBasicMaterial;
  private readonly frames: LineLoop[] = [];
  private readonly frameMaterial: LineBasicMaterial;
  private readonly frameGeometry: BufferGeometry | undefined;
  private readonly axis: LineSegments | undefined;
  private readonly axisMaterial: LineBasicMaterial;

  constructor(
    readonly block: Block,
    theme: ThemeColors,
  ) {
    const plate = block.plate!;
    this.geometry = new BoxGeometry(plate.w, PLATE_THICKNESS, plate.d);
    this.material = new MeshLambertMaterial({ color: new Color(...theme.line) });
    this.mesh = new Mesh(this.geometry, this.material);
    this.mesh.position.set(plate.x, plate.y - PLATE_THICKNESS / 2 - FRAME_LIFT, plate.z);
    this.edges = new EdgesGeometry(this.geometry);
    this.outlineMaterial = inkMaterial(theme);
    this.mesh.add(new LineSegments(this.edges, this.outlineMaterial));
    this.group.add(this.mesh);
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
    this.outlineMaterial.color.setRGB(...theme.ink);
    this.frameMaterial.color.setRGB(...theme.ink);
    this.axisMaterial.color.setRGB(...theme.ink);
  }

  update(revealed: boolean, frameXs: number[], ticks: number[], scale: number): void {
    this.outlineMaterial.opacity = revealed ? PLATE_REVEALED : PLATE_HIDDEN;
    this.frameMaterial.opacity = revealed ? FRAME_REVEALED : FRAME_HIDDEN;
    this.frames.forEach((frame, r) => {
      frame.position.x = frameXs[Math.min(r, frameXs.length - 1)];
    });
    if (this.axis !== undefined && this.block.axis !== undefined && ticks.length > 0) {
      const { x, z } = this.block.axis;
      const y0 = this.block.elements[0].y;
      const lo = Math.min(...ticks) * scale;
      const hi = Math.max(...ticks) * scale;
      const points = [new Vector3(x, y0 + lo, z), new Vector3(x, y0 + hi, z)];
      for (const v of ticks) points.push(new Vector3(x - AXIS_TICK, y0 + v * scale, z), new Vector3(x, y0 + v * scale, z));
      this.axis.geometry.setFromPoints(points);
      this.axisMaterial.opacity = revealed ? AXIS_REVEALED : AXIS_HIDDEN;
    }
  }

  dispose(): void {
    this.geometry.dispose();
    this.edges.dispose();
    this.material.dispose();
    this.outlineMaterial.dispose();
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
