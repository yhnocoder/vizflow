import { For, Index, Show, createMemo, createSignal, onCleanup, type Accessor } from 'solid-js';
import { render } from 'solid-js/web';
import { simulateSglang } from './sglang';
import { describeEvent, ENGINE_LABEL, graphText, MODE_LABEL, narrate } from './text';
import type { BatchRow, EngineName, EngineRun, EngineStep, RequestView, ServingConfig } from './types';
import { simulateVllm } from './vllm';

export type TokenState = 'wait' | 'done' | 'now' | 'hit' | 'duplicate' | 'inflight';

export function tokenState(view: RequestView, row: BatchRow | undefined, pos: number): TokenState | null {
  const known = view.prompt + view.outputs;
  if (view.status === 'finished') return pos < known ? (pos < view.hit ? 'hit' : 'done') : null;
  if (view.status === 'pending') return pos < view.prompt ? 'wait' : null;
  const total = known + view.inflight;
  if (pos >= Math.max(total, view.computed)) return null;
  const from = row?.from ?? view.computed;
  const n = row?.n ?? 0;
  if (pos >= from && pos < from + n) return row!.duplicate.includes(pos) ? 'duplicate' : 'now';
  if (pos < from) return pos < view.hit ? 'hit' : 'done';
  return pos >= known ? 'inflight' : 'wait';
}

export function cellSize(tokens: number, maxCells = 56): number {
  let cell = 1;
  while (tokens / cell > maxCells) cell *= 2;
  return cell;
}

const CELL_W = 9;
const CELL_GAP = 1;

function TokenRow(props: { view: RequestView; row: BatchRow | undefined; capacity: number }) {
  const cell = () => cellSize(props.capacity);
  const cells = () => Math.ceil(props.capacity / cell());
  const width = () => cells() * (CELL_W + CELL_GAP);
  const runs = createMemo(() => {
    const out: { x: number; w: number; state: TokenState }[] = [];
    const c = cell();
    for (let i = 0; i < cells(); i++) {
      const t0 = i * c;
      let start = -1;
      let current: TokenState | null = null;
      const flush = (end: number) => {
        if (current !== null && current !== 'wait') out.push({ x: i * (CELL_W + CELL_GAP) + ((start - t0) / c) * CELL_W, w: Math.max(((end - start) / c) * CELL_W, 1.3), state: current });
      };
      for (let t = t0; t < t0 + c; t++) {
        const s = tokenState(props.view, props.row, t);
        if (s !== current) {
          flush(t);
          current = s;
          start = t;
        }
      }
      flush(t0 + c);
    }
    return out;
  });
  const exists = (i: number) => tokenState(props.view, props.row, i * cell()) !== null;
  const status = () => ({ pending: '未到', waiting: '等待', preempted: '等待', running: '', finished: '完成' })[props.view.status];
  return (
    <div class="vf-sv-req">
      <span class="vf-sv-req-label" title={`${props.view.id}：提示词 ${props.view.prompt}，已生成 ${props.view.outputs}，每格 ${cell()} 个 token`}>
        <b>{props.view.id}</b> {props.view.prompt}+{props.view.outputs}
        <Show when={status() !== ''}> <i>{status()}</i></Show>
        <Show when={props.row !== undefined}> <em>+{props.row!.n}</em></Show>
      </span>
      <svg width={width()} height={14} viewBox={`0 0 ${width()} 14`} aria-hidden="true">
        <Index each={Array.from({ length: cells() })}>
          {(_, i) => <rect x={i * (CELL_W + CELL_GAP)} y={1} width={CELL_W} height={12} class={exists(i) ? 'vf-sv-cell' : 'vf-sv-cell vf-sv-cell-none'} />}
        </Index>
        <For each={runs()}>{(r) => <rect x={r.x} y={1} width={r.w} height={12} class={`vf-sv-${r.state}`} />}</For>
      </svg>
      <span class="vf-sv-scale">×{cell()}</span>
    </div>
  );
}

function BatchStrip(props: { step: EngineStep; budget: number; engine: EngineName }) {
  const geometry = createMemo(() => {
    const step = props.step;
    const decodeRows = props.engine === 'sglang' && step.mode === 'decode';
    const slots = decodeRows ? step.rows.length + step.padded : step.tokens + step.padded;
    const discrete = slots <= 16;
    const unit = discrete ? 24 : 560 / Math.max(props.budget, slots);
    let x = 2;
    const rows = step.rows.map((r) => {
      const w = r.n * unit;
      const item = { r, x, w };
      x += w;
      return item;
    });
    const pad = { x, w: step.padded * unit };
    return { rows, pad, end: x + pad.w, width: Math.max(x + pad.w, 40) + 6 };
  });
  return (
    <svg class="vf-sv-batch" width={geometry().width} height={34} viewBox={`0 0 ${geometry().width} 34`} role="img" aria-label="这一步的扁平批">
      <For each={geometry().rows}>
        {(g) => (
          <g>
            <rect x={g.x} y={4} width={Math.max(g.w - 1, 1.5)} height={18} class={g.r.n === 1 && g.r.duplicate.length === 0 ? 'vf-sv-decode' : 'vf-sv-now'} />
            <Show when={g.r.duplicate.length > 0}>
              <rect x={g.x} y={18} width={Math.max((g.r.duplicate.length / g.r.n) * g.w - 1, 1)} height={4} class="vf-sv-duplicate" />
            </Show>
            <line x1={g.x} y1={0} x2={g.x} y2={26} class="vf-sv-tick" />
            <Show when={g.w >= 18}>
              <text x={g.x + g.w / 2} y={17} text-anchor="middle" class="vf-sv-batch-label">{g.w > 44 ? `${g.r.id} ${g.r.n}` : g.r.id}</text>
            </Show>
            <Show when={g.r.sampled}>
              <circle cx={g.x + g.w - Math.min(4, g.w / 2)} cy={30} r={2.4} class="vf-sv-dot" />
            </Show>
          </g>
        )}
      </For>
      <Show when={geometry().pad.w > 0}>
        <rect x={geometry().pad.x} y={4} width={Math.max(geometry().pad.w - 1, 1.5)} height={18} class="vf-sv-pad" />
      </Show>
      <line x1={geometry().end} y1={0} x2={geometry().end} y2={26} class="vf-sv-tick" />
    </svg>
  );
}

function Overview(props: { run: EngineRun; current: number; total: number; budget: number; onPick: (step: number) => void }) {
  const H = 46;
  const bw = () => Math.max(2, Math.min(10, Math.floor(900 / props.total)));
  const width = () => props.total * bw() + 2;
  const line = createMemo(() => props.run.steps.map((s, i) => `${i * bw() + 1 + bw() / 2},${H - Math.min(1, s.kvUsed / props.run.kvCapacity) * (H - 6)}`).join(' '));
  return (
    <svg width={width()} height={H + 4} viewBox={`0 0 ${width()} ${H + 4}`} role="img" aria-label="每步的 token 数与 KV 占用">
      <For each={props.run.steps}>
        {(s, i) => {
          const h = () => (s.tokens === 0 ? 0 : Math.max(2, Math.min(1, s.tokens / props.budget) * (H - 6)));
          return (
            <rect
              x={i() * bw() + 1} y={H - h()} width={bw() - (bw() > 3 ? 1 : 0)} height={h()}
              class={`vf-sv-bar ${s.mode === 'decode' ? 'vf-sv-decode' : 'vf-sv-now'}`}
              opacity={i() + 1 === props.current ? 1 : 0.5}
              onClick={() => props.onPick(i() + 1)}
            />
          );
        }}
      </For>
      <polyline points={line()} class="vf-sv-kvline" />
      <line x1={(props.current - 1) * bw() + 1 + bw() / 2} y1={0} x2={(props.current - 1) * bw() + 1 + bw() / 2} y2={H} class="vf-sv-cursor" />
    </svg>
  );
}

function Timeline(props: { engine: EngineName; current: number; overlap: boolean; last: number }) {
  const W = 92;
  const steps = createMemo(() => {
    const out: number[] = [];
    for (let s = Math.max(1, props.current - 2); s <= Math.min(props.last, props.current + 1); s++) out.push(s);
    return out;
  });
  const unit = () => (props.engine === 'vllm' ? '' : '#');
  const x0 = () => steps()[0];
  return (
    <svg class="vf-sv-timeline" width={steps().length * W + 70} height={58} viewBox={`0 0 ${steps().length * W + 70} 58`} role="img" aria-label="CPU 与 GPU 的时间线">
      <text x={0} y={18} class="vf-sv-lane">CPU</text>
      <text x={0} y={46} class="vf-sv-lane">GPU</text>
      <For each={steps()}>
        {(s) => {
          const x = () => 40 + (s - x0()) * W;
          const active = () => s === props.current;
          return (
            <g opacity={active() ? 1 : 0.55}>
              <rect x={x()} y={6} width={props.overlap ? W * 0.42 : W * 0.3} height={16} rx={3} class="vf-sv-cpu" />
              <text x={x() + 4} y={18} class="vf-sv-tl-label">调度{unit()}{s}</text>
              <Show when={props.overlap && s > 1}>
                <rect x={x() + W * 0.46} y={6} width={W * 0.48} height={16} rx={3} class="vf-sv-proc" />
                <text x={x() + W * 0.5} y={18} class="vf-sv-tl-label">处理{unit()}{s - 1}</text>
              </Show>
              <Show when={!props.overlap}>
                <rect x={x() + W * 0.72} y={6} width={W * 0.26} height={16} rx={3} class="vf-sv-proc" />
                <text x={x() + W * 0.74} y={18} class="vf-sv-tl-label">处理</text>
              </Show>
              <rect x={props.overlap ? x() + W * 0.44 : x() + W * 0.32} y={34} width={props.overlap ? W : W * 0.38} height={16} rx={3} class="vf-sv-gpu" />
              <text x={(props.overlap ? x() + W * 0.44 : x() + W * 0.32) + 4} y={46} class="vf-sv-tl-label">前向{unit()}{s}</text>
            </g>
          );
        }}
      </For>
    </svg>
  );
}

function EnginePane(props: { run: EngineRun; step: Accessor<number>; total: Accessor<number>; config: ServingConfig; onPick: (step: number) => void }) {
  const run = () => props.run;
  const index = () => Math.min(props.step(), run().steps.length) - 1;
  const step = () => run().steps[index()];
  const finishedEarly = () => props.step() > run().steps.length;
  const rowOf = (id: string) => step().rows.find((r) => r.id === id);
  const capacityOf = (id: string) => {
    const r = props.config.requests.find((q) => q.id === id)!;
    return r.prompt + r.out;
  };
  const totals = createMemo(() => {
    const reqs = run().requests;
    return {
      prefill: reqs.reduce((s, r) => s + r.prefillTokens, 0),
      duplicate: reqs.reduce((s, r) => s + r.duplicateTokens, 0),
      evictions: reqs.reduce((s, r) => s + r.evictions, 0),
    };
  });
  return (
    <section class="vf-sv-pane" data-engine={run().engine}>
      <h3>{ENGINE_LABEL[run().engine]}</h3>
      <div class="vf-sv-stats">
        <span>全部完成 <b>{run().steps.length}</b> 步</span>
        <span>prefill 计算 <b>{totals().prefill}</b> token</span>
        <span>其中重复 <b>{totals().duplicate}</b></span>
        <span>{run().engine === 'vllm' ? '抢占' : '撤回'} <b>{totals().evictions}</b> 次</span>
      </div>
      <div class="vf-sv-firsts">
        首 token 所在的步 / 完成的步：{run().requests.map((r) => `${r.id} ${r.firstToken ?? '-'}/${r.finished ?? '-'}`).join('  ')}
      </div>
      <div class="vf-sv-sub">每步 token 数（橙：含 prefill，蓝：纯 decode），折线是 KV 占用（满格 = {run().kvCapacity} token）。点柱子跳到那一步。</div>
      <div class="vf-sv-scroll">
        <Overview run={run()} current={Math.min(props.step(), run().steps.length)} total={props.total()} budget={props.config.budget} onPick={props.onPick} />
      </div>
      <div class="vf-sv-head">
        <Show when={!finishedEarly()} fallback={<>这个引擎在第 {run().steps.length} 步已全部完成，下面是最后一步。</>}>
          第 {props.step()} 步 · {MODE_LABEL[step().mode]} · {step().tokens} 个 token · {graphText(step(), run().engine)}
        </Show>
      </div>
      <ul class="vf-sv-narration">
        <For each={narrate(run(), index())}>{(line) => <li>{line}</li>}</For>
      </ul>
      <div class="vf-sv-scroll">
        <For each={step().requests}>{(v) => <TokenRow view={v} row={rowOf(v.id)} capacity={capacityOf(v.id)} />}</For>
      </div>
      <div class="vf-sv-sub">扁平批：竖线是每个请求的起点，灰色是 CUDA graph 的填充，圆点表示这一行的采样结果会保留。</div>
      <div class="vf-sv-scroll"><BatchStrip step={step()} budget={props.config.budget} engine={run().engine} /></div>
      <div class="vf-sv-sub">CPU 与 GPU：{props.config.overlap ? '重叠调度，CPU 先排下一批，再处理上一批的结果' : '不重叠，每批算完再排下一批'}</div>
      <div class="vf-sv-scroll"><Timeline engine={run().engine} current={Math.min(props.step(), run().steps.length)} overlap={props.config.overlap} last={run().steps.length} /></div>
      <ul class="vf-sv-events">
        <For each={step().events}>{(e) => <li>{describeEvent(e, run().engine)}</li>}</For>
        <Show when={step().newTokenRatio !== undefined}><li>新 token 比例 r = {step().newTokenRatio!.toFixed(3)}</li></Show>
      </ul>
    </section>
  );
}

export interface ServingViewOptions {
  config: ServingConfig;
  engines?: EngineName[];
  step?: number;
}

export interface ServingHandle {
  setConfig(config: ServingConfig): void;
  setStep(step: number): void;
  readonly step: number;
  readonly runs: EngineRun[];
  dispose(): void;
}

const SIMULATORS: Record<EngineName, (c: ServingConfig) => EngineRun> = { vllm: simulateVllm, sglang: simulateSglang };

export function mountServing(target: HTMLElement, options: ServingViewOptions): ServingHandle {
  const engines = options.engines ?? ['vllm', 'sglang'];
  const [config, setConfig] = createSignal(options.config);
  const runs = createMemo(() => engines.map((e) => SIMULATORS[e](config())));
  const total = createMemo(() => Math.max(...runs().map((r) => r.steps.length)));
  const [step, setStepRaw] = createSignal(options.step ?? 1);
  const clamped = createMemo(() => Math.min(Math.max(1, step()), total()));
  const setStep = (s: number) => setStepRaw(Math.min(Math.max(1, Math.round(s)), total()));

  const dispose = render(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.target instanceof Element && event.target.closest('input, textarea, select')) return;
      if (event.key === 'ArrowRight') setStep(clamped() + 1);
      if (event.key === 'ArrowLeft') setStep(clamped() - 1);
    };
    document.addEventListener('keydown', onKey);
    onCleanup(() => document.removeEventListener('keydown', onKey));
    return (
      <div class="vf-sv">
        <div class="vf-sv-stepper">
          <button type="button" onClick={() => setStep(clamped() - 1)} disabled={clamped() <= 1}>上一步</button>
          <input type="range" min={1} max={total()} value={clamped()} aria-label="步骤" onInput={(e) => setStep(+e.currentTarget.value)} />
          <button type="button" onClick={() => setStep(clamped() + 1)} disabled={clamped() >= total()}>下一步</button>
          <span class="vf-sv-stepnum">第 {clamped()} / {total()} 步</span>
        </div>
        <div class="vf-sv-legend">
          <span><i class="vf-sv-wait" />还没算</span>
          <span><i class="vf-sv-done" />之前算过</span>
          <span><i class="vf-sv-now" />这一步算</span>
          <span><i class="vf-sv-hit" />前缀缓存命中</span>
          <span><i class="vf-sv-duplicate" />这一步算，但别处已有同样的 KV</span>
          <span><i class="vf-sv-inflight" />已采样，还在 GPU 上</span>
        </div>
        <div class="vf-sv-panes">
          <For each={runs()}>{(run) => <EnginePane run={run} step={clamped} total={total} config={config()} onPick={setStep} />}</For>
        </div>
      </div>
    );
  }, target);

  return {
    setConfig: (c) => setConfig(c),
    setStep,
    get step() {
      return clamped();
    },
    get runs() {
      return runs();
    },
    dispose,
  };
}
