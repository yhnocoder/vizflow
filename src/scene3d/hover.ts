import { BufferGeometry, Color, DoubleSide, Group, LineBasicMaterial, LineLoop, Mesh, MeshBasicMaterial, PlaneGeometry, Vector3 } from 'three';
import type { Node } from '../autograd';
import type { ThemeColors } from '../theme';
import type { Block, Element } from './layout';

export const EDGE_RELATED = 0.95;
export const EDGE_UNRELATED_FACTOR = 0.3;
export const TOKEN_FRAME_MARGIN = 0.12;
export const TOKEN_FRAME_LIFT = 0.012;
export const TOKEN_FRAME_OPACITY = 0.9;
export const TOKEN_FILL_OPACITY = 0.3;

export interface TokenRect {
  x: number;
  y: number;
  z: number;
  w: number;
  d: number;
}

export function tokenNodes(nodes: readonly Node[]): Set<Node> {
  const tokened = new Set<Node>();
  const lengths = new Set(nodes.filter((n) => (n.kind === 'input' || n.kind === 'label') && n.shape.length > 0).map((n) => n.shape[0]));
  for (const node of nodes) {
    if (node.shape.length === 0) continue;
    if (node.kind === 'input' || node.kind === 'label') tokened.add(node);
    else if (node.kind === 'buffer' && lengths.has(node.shape[0])) tokened.add(node);
    else if (node.kind === 'op' && node.inputs.some((input) => tokened.has(input) && input.shape[0] === node.shape[0])) tokened.add(node);
  }
  return tokened;
}

export function sameToken(block: Block, element: Element): Element[] {
  return block.elements.filter((e) => e.index[0] === element.index[0]);
}

export function tokenRect(block: Block, element: Element): TokenRect {
  const members = sameToken(block, element);
  const [fw, fd] = block.footprint;
  const xs = members.map((e) => e.x);
  const zs = members.map((e) => e.z);
  const x0 = Math.min(...xs) - fw / 2 - TOKEN_FRAME_MARGIN;
  const x1 = Math.max(...xs) + fw / 2 + TOKEN_FRAME_MARGIN;
  const z0 = Math.min(...zs) - fd / 2 - TOKEN_FRAME_MARGIN;
  const z1 = Math.max(...zs) + fd / 2 + TOKEN_FRAME_MARGIN;
  return { x: (x0 + x1) / 2, y: element.y + TOKEN_FRAME_LIFT, z: (z0 + z1) / 2, w: x1 - x0, d: z1 - z0 };
}

export function relatedOpacity(opacity: number, related: boolean): number {
  return related ? Math.max(opacity, EDGE_RELATED) : opacity * EDGE_UNRELATED_FACTOR;
}

export class TokenFrame {
  readonly group = new Group();
  private readonly fillGeometry = new PlaneGeometry(1, 1);
  private readonly fillMaterial: MeshBasicMaterial;
  private readonly lineGeometry: BufferGeometry;
  private readonly lineMaterial: LineBasicMaterial;
  private readonly fill: Mesh;
  private readonly outline: LineLoop;

  constructor(theme: ThemeColors) {
    this.fillMaterial = new MeshBasicMaterial({ color: new Color(...theme.pos), transparent: true, opacity: TOKEN_FILL_OPACITY, depthWrite: false, side: DoubleSide });
    this.fill = new Mesh(this.fillGeometry, this.fillMaterial);
    this.fill.rotation.x = -Math.PI / 2;
    this.lineGeometry = new BufferGeometry().setFromPoints([new Vector3(-0.5, 0, -0.5), new Vector3(0.5, 0, -0.5), new Vector3(0.5, 0, 0.5), new Vector3(-0.5, 0, 0.5)]);
    this.lineMaterial = new LineBasicMaterial({ color: new Color(...theme.pos), transparent: true, opacity: TOKEN_FRAME_OPACITY });
    this.outline = new LineLoop(this.lineGeometry, this.lineMaterial);
    this.group.add(this.fill, this.outline);
    this.group.visible = false;
  }

  set(rect: TokenRect | undefined): void {
    this.group.visible = rect !== undefined;
    if (rect === undefined) return;
    this.group.position.set(rect.x, rect.y, rect.z);
    this.fill.scale.set(rect.w, rect.d, 1);
    this.outline.scale.set(rect.w, 1, rect.d);
  }

  setTheme(theme: ThemeColors): void {
    this.fillMaterial.color.setRGB(...theme.pos);
    this.lineMaterial.color.setRGB(...theme.pos);
  }

  dispose(): void {
    this.fillGeometry.dispose();
    this.fillMaterial.dispose();
    this.lineGeometry.dispose();
    this.lineMaterial.dispose();
  }
}
