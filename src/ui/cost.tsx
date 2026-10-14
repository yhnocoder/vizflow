import { For, Show, createMemo, createSignal } from 'solid-js';
import type { Graph } from '../autograd';
import { FLOP_RULES, type FlopEntry, MEMORY_KINDS, type MemoryEntry, type MemoryKind, type Scale, type StepCost, costTimeline } from '../cost';
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
const SEQUENCE_LOG2 = { min: 7, max: 17 };
const BATCH_LOG2 = { min: 0, max: 8 };

function copiesText(copies: { drawn: number; real: number }): string {
  return copies.real % copies.drawn === 0 ? `${copies.real / copies.drawn}` : `${copies.real}/${copies.drawn}`;
}

function ScaleTags(props: { scale: Scale }) {
  return (
    <>
      <Show when={props.scale.copies}>
        {(copies) => (
          <span class="vf-costtag" title={`循环在图里画出 ${copies().drawn} 份，真实模型有 ${copies().real} 份，按 ${copies().real}/${copies().drawn} 倍计`}>
            ×{copiesText(copies())}
          </span>
        )}
      </Show>
    </>
  );
}

function scaleTex(scale: Scale): string[] {
  const factors: string[] = [];
  if (scale.copies !== undefined) factors.push(scale.copies.real % scale.copies.drawn === 0 ? `${scale.copies.real / scale.copies.drawn}` : `\tfrac{${scale.copies.real}}{${scale.copies.drawn}}`);
  if (scale.batch !== 1) factors.push(`${scale.batch}`);
  return factors;
}

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
  const sum = entry.terms.map((t) => t.join('\\times ')).join(' + ');
  const factors = scaleTex(entry.scale);
  if (factors.length === 0) return sum;
  return [...factors, entry.terms.length > 1 ? `(${sum})` : sum].join('\\times ');
}

const total = (memory: readonly MemoryEntry[]) => memory.reduce((sum, m) => sum + m.bytes, 0);

function grouped(memory: readonly MemoryEntry[]): { kind: MemoryKind; bytes: number; entries: MemoryEntry[] }[] {
  return MEMORY_KINDS.map((kind) => {
    const entries = memory.filter((m) => m.kind === kind).sort((a, b) => b.bytes - a.bytes);
    return { kind, bytes: total(entries), entries };
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
            <span class="vf-costname">
              <Rich text={`$${m.node.tex}$`} />
              <ScaleTags scale={m.scale} />
            </span>
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
              <ScaleTags scale={op.scale} />
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
  const realT = typeof g.real?.T === 'number' ? g.real.T : undefined;
  const [sequence, setSequence] = createSignal(realT);
  const [batch, setBatch] = createSignal(1);
  const [tip, setTip] = createSignal<MemoryKind | 'flops' | null>(null);
  const [pinned, setPinned] = createSignal(false);
  const list = createMemo(() => {
    props.stepsVersion();
    return props.steps?.list ?? [];
  });
  const timeline = createMemo(() => {
    const T = sequence();
    return costTimeline(g, list(), {
      real: real(),
      batch: batch(),
      ...(T === undefined ? {} : { dims: { T } }),
    });
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
  const labels = createMemo(() => {
    const seen = { forward: 0, backward: 0 };
    return list().map((info) => `${info.dir === 'forward' ? '前向' : '反向'}第 ${++seen[info.dir]} 步`);
  });
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
      <Show when={real()}>
        <div class="vf-costdims">
          <label>
            <span>
              <i>B</i> = {batch()}
            </span>
            <input
              type="range"
              min={BATCH_LOG2.min}
              max={BATCH_LOG2.max}
              step="1"
              value={Math.log2(batch())}
              aria-label="批大小 B"
              onInput={(event) => setBatch(2 ** Number(event.currentTarget.value))}
            />
          </label>
          <Show when={sequence()}>
            {(T) => (
              <label>
                <span>
                  <i>T</i> = {T()}
                </span>
                <input
                  type="range"
                  min={SEQUENCE_LOG2.min}
                  max={SEQUENCE_LOG2.max}
                  step="1"
                  value={Math.round(Math.log2(T()))}
                  aria-label="序列长度 T"
                  onInput={(event) => setSequence(2 ** Number(event.currentTarget.value))}
                />
              </label>
            )}
          </Show>
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
          <li>页面尺寸按图里画出的张量统计。真实尺寸按 real 里的维度推导每个张量的 shape：循环在图里画出 h 份、真实模型有 H 份时（例如注意力的 head、MoE 的专家），每份按 H/h 倍计，列表里标出倍数；占位盒子里的层按页面画出的同类层推算，横条上用斜线、列表里用「×N 层」标出；B 是批大小，依赖输入的张量和运算乘以 B，参数和 buffer 不变；改 T 后按新的 T 重新推导所有 shape。</li>
          <li>一个张量的大小是元素个数乘以 dtype 的字节数。不计优化器状态、分配器预留与碎片、算子内部的临时空间。显存是每一步结束时的快照，算子执行过程中输入和输出同时存在的那一刻不计入峰值。</li>
          <li>按 PyTorch eager 模式的 autograd 推算每一步还活着的张量。参数、输入、label、buffer 一直在显存里。前向的结果在后面的运算都读过、且没有被反向保存时释放；没有后继的节点算作图的输出，一直保留。反向按步骤倒序执行运算的反向，执行后释放这个运算保存的张量。一个节点的梯度在读它的运算中最先执行反向的那个（前向里最后一个读它的运算）执行反向时出现，激活的梯度在节点自己的反向执行后释放，参数梯度一直保留。</li>
          <li>「需要梯度」指节点有一条可导的路径通到损失，运算只在它需要梯度、并且有需要梯度的输入时保存张量。这和 PyTorch 的 requires_grad 不同：requires_grad 为真、但通不到损失的分支，PyTorch 也会保存张量，本面板不计。</li>
          <li>每个运算的反向保存哪些张量写在运算定义里。view、transpose、narrow、select、identity 与输入共用存储，不另计显存。本页把 softmax 等运算拆成了几步，保存的张量按拆开后的运算统计，可能多于 PyTorch 的融合算子。页面用稠密写法表示 MoE（每个专家处理全部 token，再按路由权重相加）时，显存和 FLOP 按这个稠密计算统计，远大于只把 token 发给选中专家的实现。</li>
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
