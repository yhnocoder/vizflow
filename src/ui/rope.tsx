import { For, createEffect, createMemo, createSignal, type Accessor } from 'solid-js';
import type { Graph, Node } from '../autograd';
import { formatNumber } from '../tensor';
import { fillTex, texAttr } from './figures';
import { Rich } from './rich';
import type { RopeCompareSpec, RopeSpec } from './sections';


export interface RopeState {
  t: Accessor<number>;
  s: Accessor<number>;
  pair: Accessor<number>;
  setT(t: number): void;
  setS(s: number): void;
  setPair(pair: number): void;
  start: number;
}


const ARROW_HEAD = 8;
const LABEL_BOX = 40;
const LABEL_ASCENT = 14;
const DIAL_NOTE = "实线是转过之后的 $\\mathbf{q}'$、$\\mathbf{k}'$，虚线是转之前的 $\\bar{\\mathbf{q}}$、$\\bar{\\mathbf{k}}$";
const TICK_STEP = 5;

const fmt = (v: number, digits: number) => formatNumber(v, { digits });

const range = (n: number) => Array.from({ length: n }, (_, i) => i);

const frequencies = (spec: RopeSpec) => range(spec.dh / 2).map((i) => 1 / Math.pow(spec.theta, (2 * i) / spec.dh));

const rowsOf = (nodes: Node[]) => nodes[0].shape[0];

const headCount = (nodes: Node[], dh: number) => {
  if (nodes.length > 1) return nodes.length;
  return nodes[0].shape.length === 3 ? nodes[0].shape[1] : nodes[0].shape[1] / dh;
};

function headMatrix(nodes: Node[], head: number, dh: number): number[][] {
  const node = nodes.length > 1 ? nodes[head] : nodes[0];
  const split = nodes.length === 1 && node.shape.length === 3;
  const offset = nodes.length > 1 || split ? 0 : head * dh;
  return range(node.shape[0]).map((r) => range(dh).map((c) => node.value.get(split ? [r, head, c] : [r, offset + c])));
}

const kvHead = (spec: RopeSpec, head: number) => Math.floor(head / (headCount(spec.q, spec.dh) / headCount(spec.k, spec.dh)));

const resolveRow = (row: number, count: number) => (row > 0 ? row - 1 : count + row);

function rotate(v: number[], i: number, angle: number, half: number): [number, number] {
  return [v[i] * Math.cos(angle) - v[i + half] * Math.sin(angle), v[i + half] * Math.cos(angle) + v[i] * Math.sin(angle)];
}

function dotHead(spec: RopeSpec, q: number[], k: number[], distance: number): number {
  const half = spec.dh / 2;
  return frequencies(spec).reduce((sum, w, i) => {
    const a = rotate(q, i, distance * w, half);
    return sum + a[0] * k[i] + a[1] * k[i + half];
  }, 0);
}

function vectors(spec: RopeSpec): { q: number[]; k: number[] } {
  const head = spec.head - 1;
  const q = headMatrix(spec.q, head, spec.dh);
  const k = headMatrix(spec.k, kvHead(spec, head), spec.dh);
  return { q: q[resolveRow(spec.rows[0], q.length)], k: k[resolveRow(spec.rows[1], k.length)] };
}

export function createRopeState(spec: RopeSpec): RopeState {
  const start = rowsOf(spec.q) - 1;
  const [t, setT] = createSignal(start);
  const [s, setS] = createSignal(0);
  const [pair, setPair] = createSignal(0);
  return { t, s, pair, setT, setS, setPair, start };
}

function arrow(x0: number, y0: number, x1: number, y1: number, color: string, dashed: boolean, width = 2): string {
  const a = Math.atan2(y1 - y0, x1 - x0);
  const p = (d: number) => `${(x1 - ARROW_HEAD * Math.cos(a + d)).toFixed(1)} ${(y1 - ARROW_HEAD * Math.sin(a + d)).toFixed(1)}`;
  return `<line x1="${x0}" y1="${y0}" x2="${x1}" y2="${y1}" stroke="${color}" stroke-width="${width}" ${dashed ? 'stroke-dasharray="4 3"' : ''}/><path d="M${x1} ${y1}L${p(-0.4)}L${p(0.4)}Z" fill="${color}"/>`;
}

const ordinalPosition = (row: number, count: number) => {
  const index = resolveRow(row, count);
  if (index === count - 1) return '最后一个位置';
  if (index === 0) return '第一个位置';
  return `第 ${index + 1} 个位置`;
};

export function RopeDial(props: { g: Graph; version: Accessor<number>; spec: RopeSpec; state: RopeState }) {
  const { spec, state } = props;
  const omegas = frequencies(spec);
  const half = spec.dh / 2;
  const last = spec.positions - 1;
  const caption = `选一对分量：${omegas.map((w, i) => `$\\omega = ${Number(w.toFixed(3))}$ 是第 ${i + 1}、${i + 1 + half} 分量`).join('，')}`;
  let svg!: SVGSVGElement;
  let kv!: HTMLDListElement;
  createEffect(() => {
    props.version();
    const { q, k } = vectors(spec);
    const rp = state.pair();
    const rt = state.t();
    const rs = state.s();
    const w = omegas[rp];
    const qa = [q[rp], q[rp + half]];
    const ka = [k[rp], k[rp + half]];
    const qr = rotate(q, rp, rt * w, half);
    const kr = rotate(k, rp, rs * w, half);
    const L = Math.max(Math.hypot(qa[0], qa[1]), Math.hypot(ka[0], ka[1])) || 1;
    const cx = 140;
    const cy = 140;
    const R = 104;
    const P = (v: number[]) => [cx + (v[0] / L) * R, cy - (v[1] / L) * R];
    const [qx, qy] = P(qr);
    const [kx, ky] = P(kr);
    const [qx0, qy0] = P(qa);
    const [kx0, ky0] = P(ka);
    const angQ = Math.atan2(qr[1], qr[0]);
    const angK = Math.atan2(kr[1], kr[0]);
    let dA = angQ - angK;
    while (dA > Math.PI) dA -= 2 * Math.PI;
    while (dA < -Math.PI) dA += 2 * Math.PI;
    const ar = 34;
    const a0 = -angK;
    const a1 = -angQ;
    const large = Math.abs(dA) > Math.PI ? 1 : 0;
    const sweep = dA > 0 ? 0 : 1;
    const arc = `<path d="M${(cx + ar * Math.cos(a0)).toFixed(1)} ${(cy + ar * Math.sin(a0)).toFixed(1)}A${ar} ${ar} 0 ${large} ${sweep} ${(cx + ar * Math.cos(a1)).toFixed(1)} ${(cy + ar * Math.sin(a1)).toFixed(1)}" fill="none" stroke="var(--muted)" stroke-width="1.2"/>`;
    const label = (x: number, y: number, color: string, text: string) => {
      const right = x > cx;
      const left = right ? x + 6 : x - 6 - LABEL_BOX;
      const top = y + (y > cy ? 14 : -6) - LABEL_ASCENT;
      return `<foreignObject x="${left}" y="${top}" width="${LABEL_BOX}" height="${LABEL_BOX}"><div class="vf-fo" style="color:${color};font-size:13px;text-align:${right ? 'left' : 'right'}" data-vf-text="${texAttr(text)}"></div></foreignObject>`;
    };
    svg.innerHTML = `<circle cx="${cx}" cy="${cy}" r="${R}" fill="none" stroke="var(--line)"/><line x1="${cx - R - 8}" x2="${cx + R + 8}" y1="${cy}" y2="${cy}" stroke="var(--line)"/><line y1="${cy - R - 8}" y2="${cy + R + 8}" x1="${cx}" x2="${cx}" stroke="var(--line)"/>
      <text x="${cx + R + 8}" y="${cy + 14}" font-size="10.5" fill="var(--muted)" text-anchor="end">第 ${rp + 1} 分量 →</text><text x="${cx - 4}" y="${cy - R - 4}" font-size="10.5" fill="var(--muted)" text-anchor="end">第 ${rp + 1 + half} 分量 ↑</text>
      ${arrow(cx, cy, qx0, qy0, 'var(--pos)', true, 1.3)}${arrow(cx, cy, kx0, ky0, 'var(--neg)', true, 1.3)}${arc}
      ${arrow(cx, cy, qx, qy, 'var(--pos)', false)}${arrow(cx, cy, kx, ky, 'var(--neg)', false)}
      ${label(qx, qy, 'var(--pos)', `$\\mathbf{q}'$`)}
      ${label(kx, ky, 'var(--neg)', `$\\mathbf{k}'$`)}
      <foreignObject x="8" y="262" width="272" height="20"><div class="vf-fo" style="color:var(--muted);font-size:10.5px" data-vf-text="${texAttr(DIAL_NOTE)}"></div></foreignObject>`;
    fillTex(svg);
    const dp = qr[0] * kr[0] + qr[1] * kr[1];
    const dp0 = qa[0] * ka[0] + qa[1] * ka[1];
    const dh = dotHead(spec, q, k, rt - rs);
    const term = (text: string) => `<dt data-vf-text="${texAttr(text)}"></dt>`;
    kv.innerHTML = `${term('q 转过的角度 $t\\omega$')}<dd>${fmt(rt * w, 3)} rad</dd>${term('k 转过的角度 $s\\omega$')}<dd>${fmt(rs * w, 3)} rad</dd>${term('相对距离 $t-s$')}<dd>${rt - rs}</dd>
      ${term('这一对的点积')}<dd>${fmt(dp, 3)}（转之前 ${fmt(dp0, 3)}）</dd>${term("整个 head 的 $\\mathbf{q}'\\cdot\\mathbf{k}'$")}<dd>${fmt(dh, 3)}</dd>`;
    fillTex(kv);
  });
  const slider = (name: string, value: Accessor<number>, set: (v: number) => void) => (
    <label>
      <Rich text={name} />
      <input type="range" min="0" max={last} step="1" value={value()} onInput={(e) => set(Number(e.currentTarget.value))} />
      <span>{value()}</span>
    </label>
  );
  return (
    <div class="vf-rope-dial">
      <svg class="vf-rope-svg" viewBox="0 0 280 280" role="img" aria-label="一对分量在平面里的旋转" ref={(el) => (svg = el)} />
      <div class="vf-ropectl">
        <div class="vf-ropecap">
          <Rich text={caption} />
        </div>
        <div class="vf-seg" role="group" aria-label="选择哪一对分量">
          <For each={omegas}>
            {(w, i) => (
              <button type="button" aria-pressed={state.pair() === i() ? 'true' : 'false'} onClick={() => state.setPair(i())}>
                <Rich text={`$\\omega = ${w.toFixed(3)}$`} />
              </button>
            )}
          </For>
        </div>
        {slider('q 的位置 $t$', state.t, state.setT)}
        {slider('k 的位置 $s$', state.s, state.setS)}
        <div class="vf-btns">
          <button
            type="button"
            class="vf-btn"
            onClick={() => {
              if (state.t() < last && state.s() < last) {
                state.setT(state.t() + 1);
                state.setS(state.s() + 1);
              }
            }}
          >
            <Rich text="$t$ 和 $s$ 同时 +1" />
          </button>
          <button
            type="button"
            class="vf-btn"
            onClick={() => {
              state.setT(state.start);
              state.setS(0);
            }}
          >
            <Rich text={`回到 $t = ${state.start}$，$s = 0$`} />
          </button>
        </div>
        <dl class="vf-ropekv" ref={(el) => (kv = el)} />
      </div>
    </div>
  );
}

export function RopeCurve(props: { g: Graph; version: Accessor<number>; spec: RopeSpec; state: RopeState }) {
  const { spec, state } = props;
  const qCount = rowsOf(spec.q);
  const kCount = rowsOf(spec.k);
  const source = `$\\bar{\\mathbf{q}}$ 取自${ordinalPosition(spec.rows[0], qCount)}的 head ${spec.head}，$\\bar{\\mathbf{k}}$ 取自${ordinalPosition(spec.rows[1], kCount)}`;
  let svg!: SVGSVGElement;
  const note = createMemo(() => {
    const D = state.t() - state.s();
    return D >= 0
      ? `圆圈是当前的 $t-s=${D}$。$\\bar{\\mathbf{q}}$ 和 $\\bar{\\mathbf{k}}$ 的内容固定不变（${source}），变的只是它们所在的位置。所以，这条曲线只随 $t-s$ 变化。`
      : `$t-s=${D}<0$，在 causal attention 里这一对会被 mask 挡住；曲线只画了 $t-s\\ge 0$ 的部分。`;
  });
  createEffect(() => {
    props.version();
    const { q, k } = vectors(spec);
    const N = spec.positions - 1;
    const W = 340;
    const H = 150;
    const x0 = 34;
    const x1 = W - 10;
    const y0 = 14;
    const y1 = H - 26;
    const vals = range(N + 1).map((d) => dotHead(spec, q, k, d));
    const m = Math.max(...vals.map(Math.abs)) || 1;
    const X = (d: number) => x0 + ((x1 - x0) * d) / N;
    const Y = (v: number) => (y0 + y1) / 2 - ((v / m) * (y1 - y0)) / 2;
    const path = vals.map((v, d) => `${d ? 'L' : 'M'}${X(d).toFixed(1)} ${Y(v).toFixed(1)}`).join('');
    const D = state.t() - state.s();
    svg.innerHTML = `<line x1="${x0}" x2="${x1}" y1="${Y(0)}" y2="${Y(0)}" stroke="var(--line-strong)"/>
      <path d="${path}" fill="none" stroke="var(--fg)" stroke-width="1.4"/>${vals.map((v, d) => `<circle cx="${X(d).toFixed(1)}" cy="${Y(v).toFixed(1)}" r="1.8" fill="var(--fg)"/>`).join('')}
      ${D >= 0 && D <= N ? `<circle cx="${X(D)}" cy="${Y(vals[D])}" r="5" fill="none" stroke="var(--pos)" stroke-width="2"/>` : ''}
      ${range(Math.floor(N / TICK_STEP) + 1).map((i) => i * TICK_STEP).map((d) => `<text x="${X(d)}" y="${H - 10}" font-size="10" fill="var(--muted)" text-anchor="middle">${d}</text>`).join('')}
      <foreignObject x="${x1 - LABEL_BOX}" y="${H - 22}" width="${LABEL_BOX}" height="18"><div class="vf-fo" style="color:var(--muted);font-size:10px;text-align:right" data-vf-text="$t-s$"></div></foreignObject>
      <text x="${x0 - 6}" y="${y0 + 6}" font-size="10" fill="var(--muted)" text-anchor="end">${fmt(m, 2)}</text><text x="${x0 - 6}" y="${y1 + 2}" font-size="10" fill="var(--muted)" text-anchor="end">${fmt(-m, 2)}</text>`;
    fillTex(svg);
  });
  return (
    <div class="vf-rope-curve">
      <svg class="vf-rope-svg" viewBox="0 0 340 150" role="img" aria-label="点积随相对距离变化" ref={(el) => (svg = el)} />
      <p class="vf-note">
        <Rich text={note()} />
      </p>
    </div>
  );
}

const shade = (v: number, limit: number) => {
  const percent = Math.abs(v) < 1e-9 ? 0 : 8 + Math.min(Math.abs(v) / limit, 1) * 62;
  return `color-mix(in srgb, var(${v >= 0 ? '--pos' : '--neg'}) ${percent.toFixed(0)}%, transparent)`;
};

function causalSoftmax(S: number[][]): number[][] {
  return S.map((row, t) => {
    const m = Math.max(...row.slice(0, t + 1));
    const e = row.map((v, s) => (s > t ? 0 : Math.exp(v - m)));
    const z = e.reduce((a, b) => a + b, 0);
    return e.map((v) => v / z);
  });
}

const matrixOf = (node: Node) => range(node.shape[0]).map((r) => range(node.shape[1]).map((c) => node.value.get([r, c])));

export function RopeCompare(props: { version: Accessor<number>; spec: RopeSpec; compare: RopeCompareSpec }) {
  const { spec, compare } = props;
  const heads = range(compare.scores.length);
  const [head, setHead] = createSignal(0);
  const view = createMemo(() => {
    props.version();
    const h = head();
    const scale = 1 / Math.sqrt(spec.dh);
    const q = headMatrix(spec.q, h, spec.dh);
    const k = headMatrix(spec.k, kvHead(spec, h), spec.dh);
    const T = q.length;
    const S0 = range(T).map((t) => range(T).map((s) => q[t].reduce((a, v, c) => a + v * k[s][c], 0) * scale));
    const S1 = matrixOf(compare.scores[h]);
    const P1 = matrixOf(compare.weights[h]);
    const lim = Math.max(...S0.flat(), ...S1.flat().filter(Number.isFinite).map(Math.abs), ...S0.flat().map(Math.abs)) || 1;
    const tokens = Array.from(compare.ids.value.data, (id) => compare.ids.tokens?.[Math.round(id)] ?? String(Math.round(id)));
    return {
      T,
      tokens,
      grids: [
        { M: S0, lim, title: '不加 RoPE：$\\bar{\\mathbf{q}}\\bar{\\mathbf{k}}^{\\top}/\\sqrt{d_h}$' },
        { M: S1, lim, title: "加了 RoPE：$\\mathbf{q}'\\mathbf{k}'^{\\top}/\\sqrt{d_h}$（页面里的 $\\tilde{S}$）" },
        { M: causalSoftmax(S0), lim: 1, title: '不加 RoPE 的注意力权重' },
        { M: P1, lim: 1, title: '加了 RoPE 的注意力权重（页面里的 $P$）' },
      ],
      from: S0[T - 1][0],
      to: S1[T - 1][0],
    };
  });
  return (
    <div class="vf-rope-compare">
      <div class="vf-seg" role="group" aria-label="选择 head">
        <For each={heads}>
          {(h) => (
            <button type="button" aria-pressed={head() === h ? 'true' : 'false'} onClick={() => setHead(h)}>
              {`head ${h + 1}`}
            </button>
          )}
        </For>
      </div>
      <div class="vf-ropecmp">
        <For each={view().grids}>
          {(grid) => (
            <div>
              <div class="vf-cap">
                <Rich text={grid.title} />
              </div>
              <div class="vf-cmpg" style={{ 'grid-template-columns': `34px repeat(${view().T}, minmax(0, 1fr))` }}>
                <div class="vf-h" />
                <For each={view().tokens}>{(token) => <div class="vf-h">{token}</div>}</For>
                <For each={grid.M}>
                  {(row, r) => (
                    <>
                      <div class="vf-h">{view().tokens[r()]}</div>
                      <For each={row}>
                        {(v, c) =>
                          c() > r() ? (
                            <div class="vf-h">·</div>
                          ) : (
                            <div classList={{ 'vf-dg': r() === c() }} style={{ background: shade(v, grid.lim) }}>
                              {fmt(v, 2)}
                            </div>
                          )
                        }
                      </For>
                    </>
                  )}
                </For>
              </div>
            </div>
          )}
        </For>
      </div>
      <p class="vf-note vf-cmpnote">
        <Rich text="先看框出的对角线。$t=s$ 时，q 和 k 转了同样的角度，所以两边完全相同。" />
        <br />
        {`再看对角线以外。距离越大，分数改得越多。比如 “${view().tokens[view().T - 1]}” 看 “${view().tokens[0]}”，距离是 ${view().T - 1}，分数从 ${fmt(view().from, 2)} 变成了 ${fmt(view().to, 2)}。`}
        <br />
        {'第一行是例外。它只能看自己，权重永远是 1。'}
        <br />
        {'总之，没有 RoPE，同一个词无论远近，分数都一样；有了 RoPE，模型才能区分远近。（causal mask 也隐含一点位置信息，比如越靠后的位置能看到的词越多，但很间接。）'}
      </p>
    </div>
  );
}
