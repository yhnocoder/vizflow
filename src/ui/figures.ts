import { type Graph, type Program, runProgram } from '../autograd';
import { renderText } from '../formula';
import { type Tensor, formatNumber } from '../tensor';
import { type FigureArgs, checkFigure, decodeProgram, figureArgs } from './figure-spec';

const LINES_WIDTH = 340;
const LINES_HEIGHT = 112;
const LINES_TICK = 6;
const LINES_DASHES = ['', '4 3', '1.5 2.5', '7 2 1.5 2'];
const LABEL_HEIGHT = 16;
const TICK_BOX = 40;

const range = (n: number) => Array.from({ length: n }, (_, i) => i);

export const texAttr = (text: string) => text.replace(/&/g, '&amp;').replace(/"/g, '&quot;').replace(/</g, '&lt;');

export function fillTex(root: Element): void {
  for (const el of root.querySelectorAll<HTMLElement>('[data-vf-text]')) renderText(el.dataset.vfText ?? '', el);
}

const svgLabel = (x: number, y: number, align: 'left' | 'center' | 'right', tex: string) => {
  const left = align === 'left' ? x : align === 'right' ? x - TICK_BOX : x - TICK_BOX / 2;
  return `<foreignObject x="${left}" y="${y - LABEL_HEIGHT / 2}" width="${TICK_BOX}" height="${LABEL_HEIGHT}"><div class="vf-fo" style="color:var(--muted);font-size:10px;text-align:${align}" data-vf-text="${texAttr(tex)}"></div></foreignObject>`;
};

const dashAttr = (column: number) => {
  const dash = LINES_DASHES[column % LINES_DASHES.length];
  return dash === '' ? '' : `stroke-dasharray="${dash}"`;
};

function lines(value: Tensor, args: FigureArgs): string {
  const [N, K] = value.shape;
  const shade = args.shade ?? 0;
  const at = (row: number, column: number) => value.data[row * K + column];
  const finite = Array.from(value.data).filter(Number.isFinite);
  const low = finite.length === 0 ? -1 : Math.min(...finite);
  const high = finite.length === 0 ? 1 : Math.max(...finite);
  const [lo, hi] = low === high ? [low - 1, high + 1] : [low, high];
  const x0 = 46;
  const x1 = LINES_WIDTH - 8;
  const y0 = 12;
  const y1 = LINES_HEIGHT - 22;
  const X = (row: number) => x0 + ((x1 - x0) * row) / (N - 1);
  const Y = (v: number) => y1 - ((v - lo) * (y1 - y0)) / (hi - lo);
  const curve = (column: number) => {
    const d = range(N)
      .filter((row) => Number.isFinite(at(row, column)))
      .map((row, i) => `${i === 0 ? 'M' : 'L'}${X(row).toFixed(1)} ${Y(at(row, column)).toFixed(1)}`)
      .join('');
    return `<path d="${d}" fill="none" stroke="var(--fg)" stroke-width="1.3" ${dashAttr(column)}/>`;
  };
  const dots = (column: number) =>
    range(N)
      .filter((row) => Number.isFinite(at(row, column)))
      .map((row) => `<circle cx="${X(row).toFixed(1)}" cy="${Y(at(row, column)).toFixed(1)}" r="${row < shade ? 2.8 : 1.6}" fill="var(--fg)" opacity="${row < shade ? 1 : 0.45}"/>`)
      .join('');
  const band = shade > 0 ? `<rect x="${Math.max(x0 - 4, X(-0.4))}" y="${y0 - 4}" width="${Math.min(x1 + 4, X(shade - 0.6)) - Math.max(x0 - 4, X(-0.4))}" height="${y1 - y0 + 8}" fill="var(--ink)" opacity="0.06"/>` : '';
  const zero = lo < 0 && hi > 0 ? `<line x1="${x0}" x2="${x1}" y1="${Y(0)}" y2="${Y(0)}" stroke="var(--line-strong)"/>` : '';
  const ticks = range(Math.ceil(N / LINES_TICK))
    .map((i) => i * LINES_TICK)
    .map((row) => svgLabel(X(row), LINES_HEIGHT - 12, 'center', `$${row}$`))
    .join('');
  const number = (v: number) => `$${formatNumber(v, { digits: 2, tex: true })}$`;
  const legend = range(K)
    .map((column) => `<span class="vf-lines-key"><svg width="22" height="8" aria-hidden="true"><line x1="1" x2="21" y1="4" y2="4" stroke="var(--fg)" stroke-width="1.3" ${dashAttr(column)}/></svg>列 ${column}</span>`)
    .join('');
  return `<svg viewBox="0 0 ${LINES_WIDTH} ${LINES_HEIGHT}" class="vf-lines" role="img" aria-label="${K} 条曲线，横轴是行号 0 到 ${N - 1}">
    ${band}${zero}
    ${svgLabel(x0 - 6, Y(hi), 'right', number(hi))}${svgLabel(x0 - 6, Y(lo), 'right', number(lo))}
    ${range(K).map(curve).join('')}${range(K).map(dots).join('')}
    ${ticks}
    </svg><div class="vf-lines-legend">${legend}</div>`;
}

const DRAW: Record<string, (value: Tensor, args: FigureArgs) => string> = { lines };

const drawn = new WeakMap<HTMLElement, string>();

export function renderFigures(root: HTMLElement, g: Graph): void {
  for (const el of root.querySelectorAll<HTMLElement>('vf-figure')) {
    const name = el.getAttribute('name') ?? '';
    const given = (el.getAttribute('args') ?? '').split(':').filter((a) => a !== '');
    let program: Program;
    try {
      program = decodeProgram(el.getAttribute('program') ?? '');
    } catch {
      throw new Error(`figure ${name}: program is not an encoded program`);
    }
    const value = runProgram(program, (nodeName, grad) => {
      const node = g.node(nodeName);
      if (node === undefined) throw new Error(`figure ${name}: no node '${nodeName}'`);
      if (!grad) return node.value;
      if (node.grad === null) throw new Error(`figure ${name}: ${node.kind} '${nodeName}' has no gradient`);
      return node.grad;
    });
    const problem = checkFigure(name, value.shape, given);
    if (problem !== undefined) throw new Error(`figure ${name}: ${problem}`);
    const markup = DRAW[name](value, figureArgs(name, given) as FigureArgs);
    if (drawn.get(el) === markup) continue;
    drawn.set(el, markup);
    el.innerHTML = markup;
    fillTex(el);
  }
}
