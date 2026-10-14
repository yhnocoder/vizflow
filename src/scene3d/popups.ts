import { type Node, isIntegerDtype, moduleJacobian } from '../autograd';
import { render, renderText } from '../formula';
import type { StepModule } from '../steps';
import { forEachIndex, formatNumber, formatShape, offsetOf } from '../tensor';
import { BOX_FOLDED_TEXT, type SceneBox } from './boxes';
import { type ValueGrid, edgeCardElement, nodeCardElement, placePop, placeTip, valueGridElement } from './cards';
import type { SceneController } from './controller';
import type { DrawnEdge } from './edges';
import type { Label } from './labels';
import { plates, rowCol, rowsCols, tokenText } from './layout';
import { type Renderable, showsGrad, targetOf, valuesOf } from './renderable';

export const BOX_CARD_NOTE = '点这个标签可以展开。';
export const JACOBIAN_CARD_NOTE = '折叠后的整体 Jacobian 元素。';

const TIP_DIGITS = 3;
function texTitle(title: string): string {
  const parts = title.split('$');
  return parts.map((part, i) => (i % 2 === 1 ? part : part.trim() === '' ? '' : `\\text{${part}}`)).join('');
}

export class ScenePopups {
  private readonly tip: HTMLDivElement;
  private readonly pop: HTMLDivElement;
  private pinned: { node: Node; el: HTMLElement } | undefined;

  constructor(private readonly c: SceneController) {
    this.tip = document.createElement('div');
    this.tip.className = 'vf-tip';
    this.tip.hidden = true;
    this.pop = document.createElement('div');
    this.pop.className = 'vf-pop';
    this.pop.hidden = true;
    c.root.append(this.tip, this.pop);
  }

  bindCard(label: Label, node: Node): void {
    const show = () => this.showNodeCard(node, label.el);
    label.el.addEventListener('pointerenter', show);
    label.el.addEventListener('click', show);
    label.el.addEventListener('pointerleave', () => this.hideNodeCard());
  }

  repin(): void {
    if (this.pinned !== undefined) this.showNodeCard(this.pinned.node, this.pinned.el);
  }

  hideTip(): void {
    this.tip.hidden = true;
  }

  hideUnpinned(): void {
    if (this.pinned === undefined) this.pop.hidden = true;
  }

  private tipContent(renderable: Renderable, index: number): DocumentFragment {
    const c = this.c;
    const { block } = renderable;
    const node = block.node;
    const element = block.elements[index];
    const backward = showsGrad(c, node);
    const elemTex = c.explainer.elemTex({ node, index: element.index });
    const value = valuesOf(c, node)[index];
    const label = c.aligned.get(node);
    const token = label?.tokens?.[element.col];
    const fragment = document.createDocumentFragment();
    const head = document.createElement('div');
    head.className = 'vf-f';
    head.appendChild(render(backward ? `\\partial\\ell/\\partial ${elemTex}` : elemTex));
    if (token !== undefined) head.append(`，${tokenText(token)}`);
    fragment.appendChild(head);
    const line = document.createElement('div');
    line.className = 'vf-m';
    line.appendChild(render(`= ${value === -Infinity ? '-\\infty' : formatNumber(value, { digits: !backward && isIntegerDtype(node.dtype) ? 0 : TIP_DIGITS, tex: true })}`));
    fragment.appendChild(line);
    const name = document.createElement('div');
    name.className = 'vf-s';
    name.textContent = `${node.name} · ${node.dtype}`;
    fragment.appendChild(name);
    const target = targetOf(c, block, element.row);
    if (label !== undefined && target !== element.col && !backward) {
      const note = document.createElement('div');
      note.className = 'vf-s';
      note.textContent = label.tokens === undefined ? '点击设为目标类' : '点击设为目标 token';
      fragment.appendChild(note);
    }
    return fragment;
  }

  showTip(renderable: Renderable, index: number, ev: PointerEvent): void {
    this.tip.replaceChildren(this.tipContent(renderable, index));
    this.tip.hidden = false;
    const rect = this.c.root.getBoundingClientRect();
    placeTip(this.c.root, this.tip, ev.clientX - rect.left, ev.clientY - rect.top);
  }

  private gridOf(node: Node): ValueGrid {
    const [, cols] = rowsCols(node);
    const backward = showsGrad(this.c, node);
    const values = valuesOf(this.c, node);
    const ordered = new Float64Array(values.length);
    forEachIndex(node.shape, (index, offset) => {
      const [row, col] = rowCol(node, index);
      ordered[row * cols + col] = values[offset];
    });
    return { label: backward ? '梯度' : '当前值', values: ordered, cols, heads: node.shape.length >= 3 ? plates(node).count : undefined, digits: !backward && isIntegerDtype(node.dtype) ? 0 : undefined };
  }

  showNodeCard(node: Node, el: HTMLElement): void {
    const c = this.c;
    this.pinned = { node, el };
    const view = c.levelLayout?.drawnAs.get(node);
    const card = view === undefined ? nodeCardElement(c.explainer.nodeCard(node, c.state.dir), this.gridOf(node)) : nodeCardElement(c.explainer.nodeCard(view, c.state.dir, node), this.gridOf(view));
    this.pop.replaceChildren(card);
    this.pop.hidden = false;
    this.tip.hidden = true;
    const rect = c.root.getBoundingClientRect();
    const anchor = el.getBoundingClientRect();
    placePop(c.root, this.pop, anchor.right - 8 - rect.left, anchor.top - 22 - rect.top, anchor.width);
  }

  hideNodeCard(): void {
    this.pinned = undefined;
    this.pop.hidden = true;
  }

  showBoxCard(entry: SceneBox): void {
    const c = this.c;
    const module = entry.box.module;
    const io = c.levelLayout?.io.find((p) => p.module.id === module.id);
    const fragment = document.createDocumentFragment();
    const title = document.createElement('h4');
    title.className = 'vf-pop-title';
    renderText(`${module.title}（${BOX_FOLDED_TEXT}）`, title);
    fragment.appendChild(title);
    if (io !== undefined) {
      const J = moduleJacobian(c.g, module, io.input, io.output);
      const lab = document.createElement('div');
      lab.className = 'vf-lab';
      lab.append(render(`\\partial ${io.output.tex}/\\partial ${io.input.tex}`), '，', render(`${J.length}\\times ${J[0]?.length ?? 0}`));
      fragment.append(lab, valueGridElement({ label: '', values: J.flat(), cols: J[0]?.length ?? 1 }));
      const dl = document.createElement('dl');
      dl.className = 'vf-rows';
      const row = (name: string, build: (dd: HTMLElement) => void) => {
        const dt = document.createElement('dt');
        dt.textContent = name;
        const dd = document.createElement('dd');
        build(dd);
        dl.append(dt, dd);
      };
      row('输入 → 输出', (dd) => dd.append(render(`${io.input.tex}\\;${formatShape(io.input.shape).replace(/×/g, '\\times ')}\\to ${io.output.tex}`)));
      const params = module.params.filter((n) => n.kind === 'param');
      row('参数', (dd) => {
        if (params.length === 0) dd.textContent = '无';
        else dd.append(render(params.map((n) => n.tex).join(',\\ ')));
      });
      fragment.appendChild(dl);
    }
    const note = document.createElement('p');
    note.className = 'vf-nt';
    note.textContent = BOX_CARD_NOTE;
    fragment.appendChild(note);
    this.pinned = undefined;
    this.pop.replaceChildren(fragment);
    this.pop.hidden = false;
    this.tip.hidden = true;
    const rect = c.root.getBoundingClientRect();
    const anchor = entry.label.el.getBoundingClientRect();
    placePop(c.root, this.pop, anchor.right - 8 - rect.left, anchor.top - 22 - rect.top, anchor.width);
  }

  showEdgeCard(d: DrawnEdge, ev: PointerEvent): void {
    this.pop.replaceChildren(d.jac !== undefined ? this.jacobianCard(d, d.jac) : edgeCardElement(this.c.explainer.edgeCard(d.edge), d.edge.id, d.fromNode.dtype));
    this.pop.hidden = false;
    const rect = this.c.root.getBoundingClientRect();
    placePop(this.c.root, this.pop, ev.clientX - rect.left, ev.clientY - rect.top);
  }

  private jacobianCard(d: DrawnEdge, jac: { module: StepModule; value: number }): DocumentFragment {
    const { from, to } = d.edge;
    if ('cell' in from || 'cell' in to) return document.createDocumentFragment();
    const explainer = this.c.explainer;
    const outTex = explainer.elemTex({ node: to.node, index: to.index });
    const inTex = explainer.elemTex({ node: from.node, index: from.index });
    const grad = to.node.grad;
    const upstream = grad === null ? 0 : grad.get(to.index);
    const i = offsetOf(to.node.shape, to.index) + 1;
    const j = offsetOf(from.node.shape, from.index) + 1;
    const card = {
      title: `${inTex}\\to ${outTex}`,
      local: { tex: `\\big(J_{${texTitle(jac.module.title)}}\\big)_{${i},${j}}=\\frac{\\partial ${outTex}}{\\partial ${inTex}}`, value: jac.value },
      upstream: { tex: `\\frac{\\partial\\ell}{\\partial ${outTex}}`, value: upstream },
      product: { tex: `\\frac{\\partial ${outTex}}{\\partial ${inTex}}\\cdot\\frac{\\partial\\ell}{\\partial ${outTex}}`, value: jac.value * upstream },
      note: JACOBIAN_CARD_NOTE,
    };
    return edgeCardElement(card, d.edge.id, d.fromNode.dtype);
  }
}
