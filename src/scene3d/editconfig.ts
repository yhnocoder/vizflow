import type { Node } from '../autograd';
import type { LayoutItem, StackOptions, Vec3 } from './layout';

export const POSITION_DIGITS = 2;

export interface CodeOptions {
  layout: LayoutItem[];
  camera?: object;
  labels?: Record<string, [number, number]>;
}

export interface MoveTarget {
  item: LayoutItem;
  key: 'at' | 'offsets';
}

export function round(value: number, digits = POSITION_DIGITS): number {
  const k = 10 ** digits;
  const rounded = Math.round(value * k) / k;
  return rounded === 0 ? 0 : rounded;
}

export function editableLayout(items: readonly LayoutItem[]): LayoutItem[] {
  return items.map((item) => ({ ...item, options: structuredClone(item.options) }) as LayoutItem);
}

function nodesOf(item: LayoutItem): Node[] {
  if (item.template === 'board') return [item.a, item.b, item.out];
  if (item.template === 'stack') return item.nodes;
  return [item.node];
}

export function moveTarget(items: readonly LayoutItem[], n: Node, visible: readonly Node[]): MoveTarget | undefined {
  const placed = new Set<Node>();
  for (const item of items) {
    if (item.template === 'board') {
      if (nodesOf(item).includes(n)) return { item, key: 'at' };
      nodesOf(item).forEach((m) => placed.add(m));
    } else if (item.template === 'node') {
      if (item.node === n && !placed.has(n)) return { item, key: 'at' };
      placed.add(item.node);
    } else {
      let fromAt = true;
      for (const m of item.nodes) {
        if (placed.has(m)) {
          fromAt = false;
          continue;
        }
        if (!visible.includes(m)) continue;
        if (m === n) return { item, key: fromAt ? 'at' : 'offsets' };
        fromAt = false;
        placed.add(m);
      }
    }
  }
  return undefined;
}

export function readPosition(target: MoveTarget, n: Node): Vec3 {
  const options = target.item.options as StackOptions;
  const value = target.key === 'at' ? options.at : options.offsets?.[n.name];
  return value === undefined ? [0, 0, 0] : [...value];
}

export function writePosition(target: MoveTarget, n: Node, value: Vec3): void {
  const options = target.item.options as StackOptions;
  const rounded = value.map((v) => round(v)) as Vec3;
  if (target.key === 'at') options.at = rounded;
  else options.offsets = { ...options.offsets, [n.name]: rounded };
}

const IDENTIFIER = /^[A-Za-z_$][\w$]*$/;

function quote(text: string): string {
  return `'${text.replace(/\\/g, '\\\\').replace(/'/g, "\\'")}'`;
}

export function literal(value: unknown): string {
  if (typeof value === 'number') return Object.is(value, -0) ? '0' : String(value);
  if (typeof value === 'string') return quote(value);
  if (typeof value === 'boolean') return String(value);
  if (Array.isArray(value)) return `[${value.map(literal).join(', ')}]`;
  if (value !== null && typeof value === 'object') {
    const entries = Object.entries(value).filter(([, v]) => v !== undefined);
    if (entries.length === 0) return '{}';
    return `{ ${entries.map(([k, v]) => `${IDENTIFIER.test(k) ? k : quote(k)}: ${literal(v)}`).join(', ')} }`;
  }
  throw new Error(`editconfig: cannot write ${String(value)} as code`);
}

function nodeRef(n: Node): string {
  return `g.node(${quote(n.name)})`;
}

export function templateName(item: LayoutItem): string {
  return `${item.template}(${nodesOf(item).map((n) => n.name).join(', ')})`;
}

function templateCode(item: LayoutItem): string {
  const args = nodesOf(item).map(nodeRef);
  const options = Object.entries(item.options).filter(([, v]) => v !== undefined);
  if (options.length > 0) args.push(literal(item.options));
  return `${item.template}(${args.join(', ')})`;
}

export function layoutCode(options: CodeOptions): string {
  const lines = ['layout: [', ...options.layout.map((item) => `  ${templateCode(item)},`), '],'];
  if (options.camera !== undefined && Object.keys(options.camera).length > 0) lines.push(`camera: ${literal(options.camera)},`);
  if (options.labels !== undefined && Object.keys(options.labels).length > 0) lines.push(`labels: ${literal(options.labels)},`);
  return lines.join('\n');
}
