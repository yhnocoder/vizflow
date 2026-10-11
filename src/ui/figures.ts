import type { Graph, Node } from '../autograd';
import { renderText } from '../formula';
import { WAVE_INVERSE, checkFigure } from './figure-spec';

const WAVE_WIDTH = 340;
const WAVE_HEIGHT = 112;
const WAVE_POSITIONS = 24;
const WAVE_TICK = 6;
const WAVE_SAMPLES = 4;
const LABEL_HEIGHT = 16;
const TICK_BOX = 40;

const range = (n: number) => Array.from({ length: n }, (_, i) => i);

const WAVE_FN: Record<string, (x: number) => number> = { cos: Math.cos, sin: Math.sin };

export const texAttr = (text: string) => text.replace(/&/g, '&amp;').replace(/"/g, '&quot;').replace(/</g, '&lt;');

export function fillTex(root: Element): void {
  for (const el of root.querySelectorAll<HTMLElement>('[data-vf-text]')) renderText(el.dataset.vfText ?? '', el);
}

const svgLabel = (x: number, y: number, align: 'left' | 'center' | 'right', tex: string) => {
  const left = align === 'left' ? x : align === 'right' ? x - TICK_BOX : x - TICK_BOX / 2;
  return `<foreignObject x="${left}" y="${y - LABEL_HEIGHT / 2}" width="${TICK_BOX}" height="${LABEL_HEIGHT}"><div class="vf-fo" style="color:var(--muted);font-size:10px;text-align:${align}" data-vf-text="${texAttr(tex)}"></div></foreignObject>`;
};

function ropeWave(node: Node, args: string[]): string {
  const kind = args[0];
  const f = WAVE_FN[kind];
  const inverse = WAVE_INVERSE[kind];
  const N = args[1] === undefined ? WAVE_POSITIONS : Number(args[1]);
  const T = node.shape[0];
  const dh = node.shape[node.shape.length - 1];
  const half = dh / 2;
  const omegas = range(half).map((i) => inverse(node.value.data[dh + i]));
  const x0 = 26;
  const x1 = WAVE_WIDTH - 8;
  const y0 = 12;
  const y1 = WAVE_HEIGHT - 22;
  const X = (t: number) => x0 + ((x1 - x0) * t) / N;
  const Y = (v: number) => (y0 + y1) / 2 - (v * (y1 - y0)) / 2;
  const line = (w: number, style: string) => {
    const d = range(N * WAVE_SAMPLES + 1)
      .map((s) => `${s ? 'L' : 'M'}${X(s / WAVE_SAMPLES).toFixed(1)} ${Y(f((s / WAVE_SAMPLES) * w)).toFixed(1)}`)
      .join('');
    return `<path d="${d}" fill="none" stroke="var(--fg)" stroke-width="1.3" ${style}/>`;
  };
  const dots = (w: number) =>
    range(N + 1)
      .map((t) => `<circle cx="${X(t).toFixed(1)}" cy="${Y(f(t * w)).toFixed(1)}" r="${t < T ? 2.8 : 1.6}" fill="var(--fg)" opacity="${t < T ? 1 : 0.45}"/>`)
      .join('');
  const lines = omegas.map((w, i) => line(w, i === 0 ? '' : 'stroke-dasharray="4 3" opacity="0.7"')).join('');
  const ticks = range(Math.ceil(N / WAVE_TICK))
    .map((i) => i * WAVE_TICK)
    .map((t) => svgLabel(X(t), WAVE_HEIGHT - 12, 'center', `$t=${t}$`))
    .join('');
  const legend = omegas.map((w, i) => `${i === 0 ? '实线' : '虚线'} $\\omega = ${Number(w.toFixed(3))}$（第 ${i + 1}、${i + 1 + half} 列）`).join('，');
  const note = `横轴是 position_id $t$（从 0 开始，第 1 个 token 是 $t = 0$），阴影是这里用到的 $t = ${range(T).join(', ')}$。${legend}。`;
  return `<svg viewBox="0 0 ${WAVE_WIDTH} ${WAVE_HEIGHT}" class="vf-wave" role="img" aria-label="${kind} 表延长到 ${N} 个位置">
    <rect x="${X(-0.4)}" y="${y0 - 4}" width="${X(T - 0.6) - X(-0.4)}" height="${y1 - y0 + 8}" fill="var(--ink)" opacity="0.06"/>
    <line x1="${x0}" x2="${x1}" y1="${Y(0)}" y2="${Y(0)}" stroke="var(--line-strong)"/>
    ${svgLabel(x0 - 6, Y(1), 'right', '$1$')}${svgLabel(x0 - 6, Y(-1), 'right', '$-1$')}
    ${lines}${omegas.map(dots).join('')}
    ${ticks}
    </svg><div class="vf-wave-note" data-vf-text="${texAttr(note)}"></div>`;
}

const DRAW: Record<string, (node: Node, args: string[]) => string> = { rope_wave: ropeWave };

export function renderFigures(root: HTMLElement, g: Graph): void {
  for (const el of root.querySelectorAll<HTMLElement>('vf-figure')) {
    const name = el.getAttribute('name') ?? '';
    const nodeName = el.getAttribute('node') ?? '';
    const args = (el.getAttribute('args') ?? '').split(':').filter((a) => a !== '');
    const node = g.nodes.find((n) => n.name === nodeName);
    if (node === undefined) throw new Error(`figure ${name}: no node '${nodeName}'`);
    const problem = checkFigure(name, node.shape, args);
    if (problem !== undefined) throw new Error(`figure on '${nodeName}': ${problem}`);
    el.innerHTML = DRAW[name](node, args);
    fillTex(el);
  }
}
