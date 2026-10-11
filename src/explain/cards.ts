import { type Graph, type Node, opDef } from '../autograd';
import { type Tensor, broadcastIndex, formatNumber, formatShape } from '../tensor';
import { elemTex, elemTexOf, isCell, matmulOperands } from './elem';
import { type FlowEdge, contextOf } from './edges';
import type { Explainer } from './explainer';
import { explainDefOf } from './lookup';
import { explainOf } from './registry';

export interface NodeCard {
  title: string;
  tex: string[];
  note?: string;
  rows: { label: string; value: string }[];
}

export interface EdgeCard {
  title: string;
  local: { tex: string; value: number };
  upstream: { tex: string; value: number };
  product: { tex: string; value: number };
  note?: string;
}

const VALUE_PREVIEW = 8;
const NOTE_TERMS = 8;

function valueText(t: Tensor): string {
  if (t.shape.length === 0) return `$${formatNumber(t.data[0], { tex: true })}$`;
  const shown = Array.from(t.data.subarray(0, VALUE_PREVIEW), (v) => formatNumber(v, { tex: true }));
  return `$[${shown.join(',\\ ')}${t.size > VALUE_PREVIEW ? ',\\ \\ldots' : ''}]$`;
}

function squeeze(shape: readonly number[]): number[] {
  return shape.filter((d) => d !== 1);
}

function consumersOf(g: Graph, node: Node, reachesLoss: (node: Node) => boolean): Node[] {
  if (node.grad === null) return [];
  return g.nodes.filter((other) => {
    if (other.kind !== 'op' || !reachesLoss(other)) return false;
    const flags = opDef(other.op as string)?.differentiable;
    return other.inputs.some((input, arg) => input === node && (flags === undefined || flags[arg]));
  });
}

function realRow(g: Graph, node: Node): { label: string; value: string }[] {
  if (g.real === undefined) return [];
  const name = g.real.name;
  const shape = formatShape(node.realShape);
  return [{ label: '真实模型', value: name === undefined ? shape : `${name}：${shape}` }];
}

function forwardTex(node: Node): string[] {
  if (node.kind !== 'op') return [node.tex];
  const def = explainOf(node.op as string);
  if (def === undefined) return [node.tex];
  const right = def.tex.forward({ out: node.tex, in: node.inputs.map((n) => n.tex), attrs: node.attrs });
  return [`${node.tex}=${right}`];
}

function chainTex(g: Graph, node: Node, consumers: Node[]): string[] {
  const left = `\\frac{\\partial\\ell}{\\partial ${node.tex}}`;
  if (g.findLoss() === node) return [`\\frac{\\partial\\ell}{\\partial\\ell}=1`];
  if (consumers.length === 0) return [`${left}=0`];
  const terms = consumers.map(
    (y) => `\\Big(\\frac{\\partial ${y.tex}}{\\partial ${node.tex}}\\Big)^{\\top}\\frac{\\partial\\ell}{\\partial ${y.tex}}`,
  );
  return [`${left}=${terms.join('+')}`];
}

export function nodeCardOf(g: Graph, node: Node, dir: 'forward' | 'backward', reachesLoss: (node: Node) => boolean): NodeCard {
  const override = node.card?.[dir] ?? {};
  const rows: { label: string; value: string }[] = [{ label: 'shape', value: formatShape(node.shape) }, ...realRow(g, node)];
  let card: NodeCard;
  if (dir === 'forward') {
    rows.push({ label: '当前值', value: valueText(node.value) });
    card = { title: `$${node.tex}$`, tex: forwardTex(node), rows };
  } else {
    const consumers = consumersOf(g, node, reachesLoss);
    for (const y of consumers) {
      rows.push({
        label: 'Jacobian',
        value: `$\\partial ${y.tex}/\\partial ${node.tex}$：${formatShape([...squeeze(y.shape), ...squeeze(node.shape)])}`,
      });
    }
    const grad = node.grad;
    rows.push({ label: '当前梯度', value: grad === null ? '无' : valueText(grad) });
    card = { title: `$\\frac{\\partial\\ell}{\\partial ${node.tex}}$`, tex: chainTex(g, node, consumers), rows };
  }
  if (override.title !== undefined) card.title = override.title;
  if (override.tex !== undefined) card.tex = [...override.tex];
  if (override.note !== undefined) card.note = override.note;
  return card;
}

const parenthesized = (v: number) => (v < 0 ? `\\left(${formatNumber(v, { tex: true })}\\right)` : formatNumber(v, { tex: true }));
const term = (v: number) => (v < 0 ? `(${formatNumber(v, { tex: true })})` : formatNumber(v, { tex: true }));

function localTexOf(e: FlowEdge): string {
  if (isCell(e.from)) return '1';
  const from = e.from;
  const def = explainDefOf(e.op);
  const ctx = contextOf(e.op);
  const outIndex = isCell(e.to) ? matmulOperands(e.op, e.to.cell.index).out : e.to.index;
  const ins = e.op.inputs.map((input, j) => {
    if (j === e.arg) return elemTexOf(from.node, from.index);
    if (isCell(e.to)) {
      const operands = matmulOperands(e.op, e.to.cell.index);
      return elemTexOf(input, j === 0 ? operands.a : operands.b);
    }
    return input.shape.length <= outIndex.length ? elemTexOf(input, broadcastIndex(outIndex, input.shape)) : input.elem;
  });
  return def.tex.local({
    out: elemTex(e.to),
    in: ins[e.arg],
    ins,
    arg: e.arg,
    attrs: e.op.attrs,
    outIndex,
    inIndex: from.index,
    ctx,
    elem: (arg, index) => elemTexOf(arg < 0 ? e.op : e.op.inputs[arg], index),
  });
}

function sumNote(explainer: Explainer, e: FlowEdge): string | undefined {
  if (isCell(e.from)) return undefined;
  const siblings = explainer.outEdges(e.from);
  if (siblings.length <= 1) return undefined;
  const grad = e.from.node.grad;
  const total = grad === null ? 0 : grad.get(e.from.index);
  const name = `$${elemTex(e.from)}$`;
  if (siblings.length > NOTE_TERMS) {
    return `${name} 有 ${siblings.length} 条出边，各条传回的量相加得到 $${formatNumber(total, { tex: true })}$。`;
  }
  const terms = siblings.map((s) => term(explainer.contribution(s))).join('+');
  return `${name} 有 ${siblings.length} 条出边，各条传回的量相加：$${terms}=${formatNumber(total, { tex: true })}$。`;
}

export function edgeCardOf(explainer: Explainer, e: FlowEdge): EdgeCard {
  const fromTex = elemTex(e.from);
  const toTex = elemTex(e.to);
  const fromDen = isCell(e.from) ? `(${fromTex})` : fromTex;
  const toDen = isCell(e.to) ? `(${toTex})` : toTex;
  const localValue = explainer.local(e);
  const upstreamValue = explainer.upstream(e);
  const localFrame = `\\frac{\\partial ${toDen}}{\\partial ${fromDen}}`;
  const upstreamFrame = `\\frac{\\partial\\ell}{\\partial ${toDen}}`;
  const notes = [explainer.zeroGrad(e) ? '上游梯度恒为 0，这条边传回 0。' : undefined, sumNote(explainer, e)].filter(
    (n): n is string => n !== undefined,
  );
  return {
    title: `${fromTex}\\to ${toTex}`,
    local: { tex: `${localFrame}=${localTexOf(e)}`, value: localValue },
    upstream: { tex: upstreamFrame, value: upstreamValue },
    product: {
      tex: `${upstreamFrame}\\cdot${localFrame}=${parenthesized(upstreamValue)}\\times${parenthesized(localValue)}`,
      value: localValue * upstreamValue,
    },
    note: notes.length === 0 ? undefined : notes.join(' '),
  };
}
