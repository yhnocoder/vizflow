import type { Node } from '../autograd';
import { type LayoutItem, stack } from './layout';
import type { SceneModule } from './modules';

export type LayoutEntry = { kind: 'node'; node: Node } | LayoutTree | { kind: 'parallel'; branches: LayoutTree[] };
export interface LayoutTree { kind: 'module'; title?: string; subtitle?: string; children: LayoutEntry[] }

export function autoLayout(tree: LayoutTree): { layout: LayoutItem[]; modules: SceneModule[] } {
  const nodes: Node[] = [];
  const modules: SceneModule[] = [];
  const add = (node: Node) => {
    for (const input of node.inputs) if (!nodes.includes(input)) nodes.push(input);
    if (!nodes.includes(node)) nodes.push(node);
  };
  const walk = (entry: LayoutEntry): Node[] => {
    if (entry.kind === 'node') {
      add(entry.node);
      return [entry.node];
    }
    if (entry.kind === 'parallel') return entry.branches.flatMap(walk);
    const own = entry.children.flatMap(walk);
    if (entry.title !== undefined && own.length > 0) modules.push({ title: entry.title, subtitle: entry.subtitle, nodes: own });
    return own;
  };
  tree.children.forEach(walk);
  return { layout: [stack(...nodes, { level: 4.2, back: 0 })], modules };
}
