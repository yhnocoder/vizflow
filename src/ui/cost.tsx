import { For, Show, createMemo, createSignal } from 'solid-js';
import type { Graph } from '../autograd';
import { FLOP_RULES, type FlopEntry, MEMORY_KINDS, type MemoryEntry, type MemoryKind, type StepCost, costTimeline } from '../cost';
import type { Steps } from '../steps';
import { formatShape } from '../tensor';
import { Rich } from './rich';

export interface CostPanelOptions {
  title?: string;
  dims?: 'toy' | 'real';
}

const KIND_NAMES: Readonly<Record<MemoryKind, string>> = {
  param: '参数',
  input: '输入',
  buffer: '缓冲区',
  saved: '为反向保存的激活',
  temp: '临时激活',
  param_grad: '参数梯度',
  grad: '激活梯度',
};

const BYTE_UNITS = ['B', 'KiB', 'MiB', 'GiB', 'TiB'];
const FLOP_UNITS = ['', 'K', 'M', 'G', 'T', 'P', 'E'];
const LIST_LIMIT = 8;
const TIMELINE_HEIGHT = 64;
const TIMELINE_BAR = 0.7;

function scaled(value: number, base: number, units: string[]): { value: number; unit: string } {
  let at = 0;
  while (Math.abs(value) >= base && at < units.length - 1) {
    value /= base;
    at++;
  }
  return { value, unit: units[at] };
}

export function formatBytes(bytes: number): string {
  const { value, unit } = scaled(bytes, 1024, BYTE_UNITS);
  return `${unit === 'B' ? value : value.toFixed(value < 10 ? 2 : 1)} ${unit}`;
}

export function formatFlops(flops: number): string {
  const { value, unit } = scaled(flops, 1000, FLOP_UNITS);
  return `${unit === '' ? value : value.toFixed(value < 10 ? 2 : 1)} ${unit}FLOP`;
}

export function flopTex(entry: FlopEntry): string {
  return entry.terms.map((t) => t.join('\\times ')).join(' + ');
}

function grouped(memory: readonly MemoryEntry[]): { kind: MemoryKind; bytes: number; entries: MemoryEntry[] }[] {
  return MEMORY_KINDS.map((kind) => {
    const entries = memory.filter((m) => m.kind === kind).sort((a, b) => b.bytes - a.bytes);
    return { kind, bytes: entries.reduce((sum, m) => sum + m.bytes, 0), entries };
  }).filter((group) => group.entries.length > 0);
}

function TensorList(props: { entries: MemoryEntry[] }) {
  const shown = () => props.entries.slice(0, LIST_LIMIT);
  const rest = () => props.entries.slice(LIST_LIMIT);
  return (
    <div class="vf-costlist">
      <For each={shown()}>
        {(m) => (
          <>
            <Rich class="vf-costname" text={`$${m.node.tex}$`} />
            <span class="vf-costshape">{formatShape(m.shape)}</span>
            <span class="vf-costnum">{formatBytes(m.bytes)}</span>
          </>
        )}
      </For>
      <Show when={rest().length > 0}>
        <span class="vf-costmore">其余 {rest().length} 个</span>
        <span class="vf-costnum">{formatBytes(rest().reduce((sum, m) => sum + m.bytes, 0))}</span>
      </Show>
    </div>
  );
}

function OpList(props: { ops: FlopEntry[] }) {
  const sorted = () => [...props.ops].sort((a, b) => b.flops - a.flops);
  const shown = () => sorted().slice(0, LIST_LIMIT);
  const rest = () => sorted().slice(LIST_LIMIT);
  return (
    <div class="vf-costlist">
      <For each={shown()}>
        {(op) => (
          <>
            <span class="vf-costname vf-costwide">
              <Rich text={`$${op.node.tex}$`} /> <span class="vf-costop">{op.node.op}</span>
            </span>
            <span class="vf-costnum">{formatFlops(op.flops)}</span>
            <Rich class="vf-costshape vf-costformula" text={`$${flopTex(op)}$`} />
          </>
        )}
      </For>
      <Show when={rest().length > 0}>
        <span class="vf-costmore">其余 {rest().length} 个运算</span>
        <span class="vf-costnum">{formatFlops(rest().reduce((sum, op) => sum + op.flops, 0))}</span>
      </Show>
    </div>
  );
}

function Timeline(props: { steps: StepCost[]; peak: number; current: number; go: (i: number) => void; labels: string[] }) {
  const most = () => Math.max(1, ...props.steps.map((s) => s.flops));
  const width = () => props.steps.length;
  const y = (fraction: number) => TIMELINE_HEIGHT - fraction * (TIMELINE_HEIGHT - 4);
  const line = () =>
    props.steps
      .flatMap((s, i) => (s.overview ? [] : [`${i + 0.5},${y(props.peak === 0 ? 0 : s.bytes / props.peak).toFixed(2)}`]))
      .join(' ');
  return (
    <svg class="vf-costline" viewBox={`0 0 ${width()} ${TIMELINE_HEIGHT}`} preserveAspectRatio="none" role="img" aria-label="每一步的计算量与显存">
      <For each={props.steps}>
        {(s, i) => (
          <g class="vf-coststep" classList={{ 'vf-on': i() === props.current }} onClick={() => props.go(i())}>
            <rect x={i()} y="0" width="1" height={TIMELINE_HEIGHT} class="vf-costhit" />
            <Show when={!s.overview}>
              <rect x={i() + (1 - TIMELINE_BAR) / 2} width={TIMELINE_BAR} y={y(s.flops / most())} height={TIMELINE_HEIGHT - y(s.flops / most())} class="vf-costbar" />
            </Show>
            <title>{`${props.labels[i()]}：${formatFlops(s.flops)}，${formatBytes(s.bytes)}`}</title>
          </g>
        )}
      </For>
      <polyline points={line()} class="vf-costmem" />
    </svg>
  );
}

export function CostPanel(props: { g: Graph; steps: Steps | undefined; stepsVersion: () => number; options: CostPanelOptions }) {
  const { g } = props;
  const [real, setReal] = createSignal(props.options.dims !== 'toy' && g.real !== undefined);
  const [tip, setTip] = createSignal<MemoryKind | 'flops' | null>(null);
  const [pinned, setPinned] = createSignal(false);
  const timeline = createMemo(() => {
    props.stepsVersion();
    return costTimeline(g, props.steps?.list ?? [], { real: real() });
  });
  const current = () => {
    props.stepsVersion();
    return props.steps?.current ?? 0;
  };
  const step = (): StepCost | undefined => timeline().steps[current()];
  const groups = () => grouped(step()?.memory ?? []);
  const peak = () => timeline().peak;
  const hover = (key: MemoryKind | 'flops') => ({
    onMouseEnter: () => !pinned() && setTip(key),
    onMouseLeave: () => !pinned() && setTip(null),
    onClick: () => {
      const same = tip() === key && pinned();
      setPinned(!same);
      setTip(same ? null : key);
    },
  });
  const labels = () => {
    props.stepsVersion();
    const seen = { forward: 0, backward: 0 };
    return (props.steps?.list ?? []).map((info) => `${info.dir === 'forward' ? '前向' : '反向'}第 ${++seen[info.dir]} 步`);
  };
  const tipGroup = () => groups().find((group) => group.kind === tip());
  const realName = () => (typeof g.real?.name === 'string' ? g.real.name : '真实模型');
  return (
    <div class="vf-card vf-cost">
      <h2>
        <Rich text={props.options.title ?? '显存与计算量'} />
      </h2>
      <Show when={g.real !== undefined}>
        <div class="vf-seg" role="group" aria-label="按哪种尺寸计算">
          <button type="button" aria-pressed={!real()} onClick={() => setReal(false)}>
            页面尺寸
          </button>
          <button type="button" aria-pressed={real()} onClick={() => setReal(true)}>
            {realName()}
          </button>
        </div>
      </Show>
      <div class="vf-costhead">
        <span class="vf-costlabel">已分配</span>
        <span class="vf-costtotal">
          {formatBytes(step()?.bytes ?? 0)}
          <span class="vf-costpeak">
            {' '}
            / 峰值 {formatBytes(peak())}
            <Show when={timeline().peakStep >= 0}>（{labels()[timeline().peakStep]}）</Show>
          </span>
        </span>
      </div>
      <div class="vf-costbarwrap">
        <div class="vf-costmeter" classList={{ 'vf-overview': step()?.overview === true }}>
          <For each={groups()}>
            {(group) => (
              <button
                type="button"
                class={`vf-costseg vf-k-${group.kind}`}
                classList={{ 'vf-on': tip() === group.kind }}
                style={{ width: `${peak() === 0 ? 0 : (group.bytes / peak()) * 100}%` }}
                aria-label={`${KIND_NAMES[group.kind]} ${formatBytes(group.bytes)}`}
                {...hover(group.kind)}
              />
            )}
          </For>
        </div>
        <Show when={tipGroup()}>
          {(group) => (
            <div class="vf-costtip" role="tooltip">
              <div class="vf-costtiphead">
                <span class={`vf-costdot vf-k-${group().kind}`} />
                {KIND_NAMES[group().kind]}，{group().entries.length} 个张量，{formatBytes(group().bytes)}
              </div>
              <TensorList entries={group().entries} />
            </div>
          )}
        </Show>
      </div>
      <div class="vf-costlegend">
        <For each={groups()}>
          {(group) => (
            <button type="button" class="vf-costkey" {...hover(group.kind)}>
              <span class={`vf-costdot vf-k-${group.kind}`} />
              {KIND_NAMES[group.kind]}
            </button>
          )}
        </For>
      </div>
      <div class="vf-costflops">
        <button type="button" class="vf-costkey" classList={{ 'vf-on': tip() === 'flops' }} {...hover('flops')}>
          本步 <b>{formatFlops(step()?.flops ?? 0)}</b>
        </button>
        <span class="vf-costpeak">
          前向累计 {formatFlops(step()?.forward ?? 0)}，反向累计 {formatFlops(step()?.backward ?? 0)}
        </span>
      </div>
      <Show when={tip() === 'flops' && (step()?.ops.length ?? 0) > 0}>
        <div class="vf-costtip vf-costtip-flat" role="tooltip">
          <OpList ops={step()?.ops ?? []} />
        </div>
      </Show>
      <Timeline steps={timeline().steps} peak={peak()} current={current()} go={(i) => props.steps?.go(i)} labels={labels()} />
      <div class="vf-costaxis">
        <span>柱：每步 FLOP</span>
        <span>线：已分配显存</span>
      </div>
      <Show when={timeline().uncounted.length > 0}>
        <div class="vf-note">
          这些运算没有定义 cost，FLOP 未计入，反向按保存全部输入统计：{[...new Set(timeline().uncounted.map((n) => n.op))].join('、')}。
        </div>
      </Show>
      <Show when={step()?.overview === true}>
        <div class="vf-note">总览一步显示峰值时的显存和整次前向、反向的计算量。</div>
      </Show>
      <details class="vf-costrules">
        <summary>怎么统计的</summary>
        <ul>
          <li>只统计本页计算图里的张量，图里收成占位盒子的层不计入。一个张量的大小是元素个数乘以 dtype 的字节数，不计优化器状态、分配器预留和算子内部的临时空间。</li>
          <li>按 PyTorch eager 模式的 autograd 推算每一步还活着的张量：前向的结果在后面的运算都用过、且没有被反向保存时释放；反向按步骤倒序执行运算的反向，执行后释放这个运算保存的张量；激活的梯度传给输入以后释放，参数梯度一直保留。</li>
          <li>每个运算的反向保存哪些张量写在运算定义里。view、transpose、narrow、select、identity 与输入共用存储，不另计显存。本页把 softmax 等运算拆成了几步，保存的张量按拆开后的运算统计，可能多于 PyTorch 的融合算子。</li>
          <li>
            <Rich text={String.raw`FLOP 按下表计数，反向只计需要梯度的输入，每个输入各计一次。$|y|$ 是输出的元素个数，$|x|$ 是第一个输入的元素个数，$|t|$ 是目标的个数，矩阵乘是 $(m, k)$ 乘 $(k, n)$。`} />
          </li>
        </ul>
        <table class="vf-costrule">
          <thead>
            <tr>
              <th>运算</th>
              <th>前向</th>
              <th>反向</th>
            </tr>
          </thead>
          <tbody>
            <For each={FLOP_RULES}>
              {(rule) => (
                <tr>
                  <td>{rule.ops.join(', ')}</td>
                  <td>
                    <Rich text={`$${rule.forward}$`} />
                  </td>
                  <td>
                    <Rich text={`$${rule.backward}$`} />
                  </td>
                </tr>
              )}
            </For>
          </tbody>
        </table>
      </details>
    </div>
  );
}
