import type { EdgeCard, NodeCard } from '../explain';
import { render, renderText } from '../formula';
import { formatNumber } from '../tensor';

export const POP_OFFSET_X = 18;
export const POP_OFFSET_Y = 14;
export const TIP_OFFSET = 14;
const POP_MARGIN = 12;
const TIP_MARGIN = 8;

function element(tag: string, className: string, text?: string): HTMLElement {
  const el = document.createElement(tag);
  el.className = className;
  if (text !== undefined) el.textContent = text;
  return el;
}

function note(text: string): HTMLElement {
  const el = element('p', 'vf-nt');
  renderText(text, el);
  return el;
}

export function nodeCardElement(card: NodeCard): DocumentFragment {
  const fragment = document.createDocumentFragment();
  fragment.appendChild(element('h4', 'vf-pop-title', card.title));
  const eqs = element('div', 'vf-eqs');
  for (const tex of card.tex) eqs.appendChild(render(tex, { display: true }));
  fragment.appendChild(eqs);
  if (card.note !== undefined) fragment.appendChild(note(card.note));
  const dl = element('dl', 'vf-rows');
  for (const row of card.rows) {
    dl.appendChild(element('dt', '', row.label));
    dl.appendChild(element('dd', '', row.value));
  }
  fragment.appendChild(dl);
  return fragment;
}

function equation(label: string, tex: string, value: number): HTMLElement[] {
  const eq = element('div', 'vf-eq');
  eq.dataset.value = formatNumber(value);
  eq.appendChild(render(`${tex}=${formatNumber(value, { tex: true })}`, { display: true }));
  return [element('div', 'vf-lab', label), eq];
}

export function edgeCardElement(card: EdgeCard, edgeId: string): DocumentFragment {
  const fragment = document.createDocumentFragment();
  const title = element('h4', 'vf-pop-title');
  title.appendChild(render(card.title));
  fragment.appendChild(title);
  fragment.append(...equation('局部导数', card.local.tex, card.local.value));
  fragment.append(...equation('上游梯度', card.upstream.tex, card.upstream.value));
  fragment.append(...equation('沿这条边传回', card.product.tex, card.product.value));
  if (card.note !== undefined) fragment.appendChild(note(card.note));
  const marker = fragment.firstElementChild as HTMLElement;
  marker.dataset.edgeId = edgeId;
  return fragment;
}

export function placePop(stage: HTMLElement, pop: HTMLElement, x: number, y: number, width = 0): void {
  const pw = pop.offsetWidth;
  const ph = pop.offsetHeight;
  const sw = stage.clientWidth;
  const sh = stage.clientHeight;
  let left = x + POP_OFFSET_X;
  let top = y + POP_OFFSET_Y;
  if (left + pw > sw - POP_MARGIN) left = x - pw - POP_OFFSET_X - width;
  if (left < POP_MARGIN) left = POP_MARGIN;
  if (top + ph > sh - POP_MARGIN) top = sh - ph - POP_MARGIN;
  pop.style.left = `${left}px`;
  pop.style.top = `${Math.max(POP_MARGIN, top)}px`;
}

export function placeTip(stage: HTMLElement, tip: HTMLElement, x: number, y: number): void {
  const sw = stage.clientWidth;
  const sh = stage.clientHeight;
  let left = x + TIP_OFFSET;
  let top = y + TIP_OFFSET;
  if (left + tip.offsetWidth > sw - TIP_MARGIN) left = x - tip.offsetWidth - TIP_OFFSET;
  if (top + tip.offsetHeight > sh - TIP_MARGIN) top = y - tip.offsetHeight - TIP_OFFSET;
  tip.style.left = `${Math.max(TIP_MARGIN, left)}px`;
  tip.style.top = `${Math.max(TIP_MARGIN, top)}px`;
}
