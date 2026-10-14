import { Tensor } from '../tensor';
import type { Graph } from './graph';
import type { Node } from './node';
import type { AnyOpDef } from './op';
import { inputDifferentiable, opDef } from './registry';

export function moduleJacobian(g: Graph, module: { readonly nodes: readonly Node[] }, input: Node, output: Node): number[][] {
  return moduleJacobians(g, module, [input], output).get(input)!;
}

export function moduleJacobians(g: Graph, module: { readonly nodes: readonly Node[] }, inputs: readonly Node[], output: Node): Map<Node, number[][]> {
  const inside = new Set(module.nodes);
  const sources = new Set(inputs);
  const reached = new Set<Node>(inputs);
  for (const n of g.nodes) if (n.kind === 'op' && inside.has(n) && n.id <= output.id && n.inputs.some((source) => reached.has(source))) reached.add(n);
  const ops = g.nodes.filter((n) => !sources.has(n) && reached.has(n)).reverse();
  const rows = new Map<Node, number[][]>([...sources].map((input) => [input, []]));
  for (let i = 0; i < output.value.size; i++) {
    const grads = new Map<Node, Tensor>();
    const seed = Tensor.zeros([...output.shape]);
    seed.data[i] = 1;
    grads.set(output, seed);
    for (const node of ops) {
      const gradOut = grads.get(node);
      if (gradOut === undefined) continue;
      const def = opDef(node.op as string) as AnyOpDef;
      const inputGrads = def.backward(gradOut, node.inputs.map((n) => n.value), node.value, node.attrs);
      node.inputs.forEach((source, k) => {
        const grad = inputGrads[k];
        if (grad === null || grad === undefined || !reached.has(source) || !inputDifferentiable(def, k)) return;
        const target = grads.get(source);
        if (target === undefined) grads.set(source, grad.clone());
        else for (let j = 0; j < target.size; j++) target.data[j] += grad.data[j];
      });
    }
    for (const [input, list] of rows) {
      const row = grads.get(input);
      list.push(row === undefined ? new Array<number>(input.value.size).fill(0) : Array.from(row.data));
    }
  }
  return rows;
}
