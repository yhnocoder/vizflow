import { BufferGeometry, Color, DoubleSide, Group, LineBasicMaterial, LineLoop, Mesh, MeshBasicMaterial, PlaneGeometry, Vector3 } from 'three';
import type { Node } from '../autograd';
import { explainOf } from '../explain';
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

const REDUCE_OPS = new Set(['sum', 'mean', 'max', 'logsumexp']);
const ALONG_OPS = new Set(['softmax', 'log_softmax']);

function axisOf(dim: unknown, rank: number): number {
  const d = Number(dim);
  return d < 0 ? d + rank : d;
}

function viewAxis(input: readonly number[], output: readonly number[], axis: number): number | undefined {
  const before = input.slice(0, axis).reduce((p, n) => p * n, 1);
  let product = 1;
  for (let k = 0; k < output.length; k++) {
    if (product === before && output[k] === input[axis]) return k;
    product *= output[k];
  }
  return undefined;
}

function opTokenAxis(node: Node, axes: Map<Node, number>): number | undefined {
  const { inputs, attrs, shape } = node;
  const first = axes.get(inputs[0]);
  const rank = (n: Node) => n.shape.length;
  switch (node.op) {
    case 'transpose': {
      if (first === undefined) return undefined;
      const d0 = axisOf(attrs.dim0, rank(inputs[0]));
      const d1 = axisOf(attrs.dim1, rank(inputs[0]));
      return first === d0 ? d1 : first === d1 ? d0 : first;
    }
    case 'matmul': {
      if (first !== undefined && first < rank(inputs[0]) - 1) return first + shape.length - rank(inputs[0]);
      const other = axes.get(inputs[1]);
      return other !== undefined && other === rank(inputs[1]) - 1 ? shape.length - 1 : undefined;
    }
    case 'select_matmul':
      return first === 0 ? 0 : axes.get(inputs[1]) === 1 ? 1 : undefined;
    case 'embedding':
      return first;
    case 'select': {
      if (first === undefined) return undefined;
      const dim = axisOf(attrs.dim, rank(inputs[0]));
      return first === dim ? undefined : first > dim ? first - 1 : first;
    }
    case 'narrow':
      return first === axisOf(attrs.dim, rank(inputs[0])) ? undefined : first;
    case 'cat': {
      const dim = axisOf(attrs.dim ?? 0, rank(inputs[0]));
      return first !== dim && inputs.every((n) => axes.get(n) === first) ? first : undefined;
    }
    case 'view':
    case 'reshape':
    case 'unsqueeze':
    case 'squeeze':
      return first === undefined ? undefined : viewAxis(inputs[0].shape, shape, first);
    case 'nll_loss':
      return attrs.reduction === 'none' && first === 0 ? 0 : undefined;
  }
  if (node.op !== undefined && ALONG_OPS.has(node.op)) return first;
  if (node.op !== undefined && REDUCE_OPS.has(node.op)) {
    if (first === undefined) return undefined;
    const r = rank(inputs[0]);
    const dims = attrs.dim === null || attrs.dim === undefined ? Array.from({ length: r }, (_, i) => i) : (Array.isArray(attrs.dim) ? attrs.dim : [attrs.dim]).map((d) => axisOf(d, r));
    if (dims.includes(first)) return undefined;
    return attrs.keepdim === true ? first : first - dims.filter((d) => d < first).length;
  }
  if (node.op === undefined || explainOf(node.op)?.pattern !== 'elementwise') return undefined;
  for (const input of inputs) {
    const axis = axes.get(input);
    if (axis === undefined) continue;
    const at = shape.length - rank(input) + axis;
    if (shape[at] === input.shape[axis]) return at;
  }
  return undefined;
}

export function tokenAxes(nodes: readonly Node[]): Map<Node, number> {
  const axes = new Map<Node, number>();
  for (const node of nodes) {
    if (node.shape.length === 0) continue;
    const axis = node.kind === 'input' || node.kind === 'label' ? 0 : node.kind === 'op' ? opTokenAxis(node, axes) : undefined;
    if (axis !== undefined) axes.set(node, axis);
  }
  return axes;
}

export function sameToken(block: Block, element: Element, axis: number): Element[] {
  return block.elements.filter((e) => e.index[axis] === element.index[axis]);
}

export function tokenRect(block: Block, element: Element, axis: number): TokenRect {
  return elementsRect(block, sameToken(block, element, axis));
}

export function elementsRect(block: Block, members: readonly Element[]): TokenRect {
  const [fw, fd] = block.footprint;
  const xs = members.map((e) => e.x);
  const zs = members.map((e) => e.z);
  const x0 = Math.min(...xs) - fw / 2 - TOKEN_FRAME_MARGIN;
  const x1 = Math.max(...xs) + fw / 2 + TOKEN_FRAME_MARGIN;
  const z0 = Math.min(...zs) - fd / 2 - TOKEN_FRAME_MARGIN;
  const z1 = Math.max(...zs) + fd / 2 + TOKEN_FRAME_MARGIN;
  return { x: (x0 + x1) / 2, y: members[0].y + TOKEN_FRAME_LIFT, z: (z0 + z1) / 2, w: x1 - x0, d: z1 - z0 };
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
