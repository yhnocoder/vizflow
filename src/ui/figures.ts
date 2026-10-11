import type { Graph, Node } from '../autograd';
import { formatNumber } from '../tensor';

interface FigureDef {
  check(shape: readonly number[], args: string[]): string | undefined;
  draw(node: Node, args: string[]): string;
}

const WAVE_WIDTH = 340;
const WAVE_HEIGHT = 112;
const WAVE_POSITIONS = 24;
const WAVE_TICK = 6;
const WAVE_SAMPLES = 4;

const range = (n: number) => Array.from({ length: n }, (_, i) => i);

const WAVES: Record<string, { f: (x: number) => number; inverse: (y: number) => number }> = {
  cos: { f: Math.cos, inverse: Math.acos },
  sin: { f: Math.sin, inverse: Math.asin },
};

function ropeWave(node: Node, args: string[]): string {
  const kind = args[0];
  const { f, inverse } = WAVES[kind];
  const N = args[1] === undefined ? WAVE_POSITIONS : Number(args[1]);
  const [T, dh] = node.shape;
  const half = dh / 2;
  const omegas = range(half).map((i) => inverse(node.value.get([1, i])));
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
    .map((t) => `<text x="${X(t)}" y="${WAVE_HEIGHT - 8}" text-anchor="middle" font-size="10" fill="var(--muted)">t=${t}</text>`)
    .join('');
  const legend = omegas.map((w, i) => `${i === 0 ? '实线' : '虚线'} ω = ${Number(w.toFixed(3))}（第 ${i + 1}、${i + 1 + half} 列）`).join('，');
  return `<svg viewBox="0 0 ${WAVE_WIDTH} ${WAVE_HEIGHT}" class="vf-wave" role="img" aria-label="${kind} 表延长到 ${N} 个位置">
    <rect x="${X(-0.4)}" y="${y0 - 4}" width="${X(T - 0.6) - X(-0.4)}" height="${y1 - y0 + 8}" fill="var(--ink)" opacity="0.06"/>
    <line x1="${x0}" x2="${x1}" y1="${Y(0)}" y2="${Y(0)}" stroke="var(--line-strong)"/>
    <text x="${x0 - 6}" y="${Y(1) + 4}" text-anchor="end" font-size="10" fill="var(--muted)">1</text><text x="${x0 - 6}" y="${Y(-1) + 4}" text-anchor="end" font-size="10" fill="var(--muted)">${formatNumber(-1, { digits: 0 })}</text>
    ${lines}${omegas.map(dots).join('')}
    ${ticks}
    </svg><div class="vf-wave-note">横轴是 position_id t（从 0 开始，第 1 个 token 是 t = 0），阴影是这里用到的 t = ${range(T).join(', ')}。${legend}。</div>`;
}

const FIGURES: Record<string, FigureDef> = {
  rope_wave: {
    check(shape, args) {
      if (!(args[0] in WAVES)) return `rope_wave needs cos or sin as its first argument, got '${args[0] ?? ''}'`;
      if (args[1] !== undefined && !(Number.isInteger(Number(args[1])) && Number(args[1]) >= 1)) return `rope_wave positions must be a positive integer, got '${args[1]}'`;
      if (args.length > 2) return 'rope_wave takes at most 2 arguments';
      if (shape.length !== 2 || shape[0] < 2 || shape[1] % 2 !== 0) return 'rope_wave needs a (T, d_h) table with at least 2 rows and an even d_h';
      return undefined;
    },
    draw: ropeWave,
  },
};

export const figureNames: readonly string[] = Object.keys(FIGURES);

export function checkFigure(name: string, shape: readonly number[], args: string[]): string | undefined {
  const def = FIGURES[name];
  if (def === undefined) return `unknown figure '${name}'`;
  return def.check(shape, args);
}

export function figureTag(name: string, node: string, args: string[]): string {
  return `<vf-figure name="${name}" node="${node}" args="${args.join(':')}"></vf-figure>`;
}

export function renderFigures(root: HTMLElement, g: Graph): void {
  for (const el of root.querySelectorAll<HTMLElement>('vf-figure')) {
    const name = el.getAttribute('name') ?? '';
    const nodeName = el.getAttribute('node') ?? '';
    const args = (el.getAttribute('args') ?? '').split(':').filter((a) => a !== '');
    const node = g.nodes.find((n) => n.name === nodeName);
    if (node === undefined) throw new Error(`figure ${name}: no node '${nodeName}'`);
    const problem = checkFigure(name, node.shape, args);
    if (problem !== undefined) throw new Error(`figure on '${nodeName}': ${problem}`);
    el.innerHTML = FIGURES[name].draw(node, args);
  }
}
