import { For, Show, createMemo, createSignal, type Accessor } from 'solid-js';
import { type PagedEngine, type PagedStep, type RadixNodeView, simulatePages } from '../sim';
import { type KvBlocksSpec, PAGE_REQUEST, type StepState, engineStepAt, kvScenario, kvTokenText } from '../steps';

const ENGINE_LABEL: Record<PagedEngine, string> = { vllm: 'vLLM', sglang: 'SGLang' };

const ENGINE_NOTE: Record<PagedEngine, (blockSize: number) => string> = {
  vllm: (b) =>
    `vLLM 的 PagedAttention 把缓存切成每块 ${b} 个位置的块，一个请求用块表记录它的第几块放在哪个物理块。块写满以后，用前一块的哈希和这块的 token 算出块哈希，登记到前缀缓存里。新请求到达时按同样的规则算哈希，从第一块开始查，命中的块直接引用，引用计数加 1，只给没命中的部分分配新块。0 号块是 vLLM 保留的空块，不存数据。`,
  sglang: () =>
    'SGLang 的 RadixAttention 按 token 分配槽位（page_size 为 1），用 radix 树记录已经算过的 token 序列和它们所在的槽位。新请求到达时在树上逐个 token 匹配最长前缀，匹配到的槽位直接复用，只给剩下的 token 分配槽位。prefill 算完以后，prompt 插入树中；decode 生成的 token 在请求结束时才插入。SGLang 默认一步只跑 prefill 或只跑 decode，新请求的 prefill 那一步，正在 decode 的请求等一步。',
};

function scheduledText(step: PagedStep): string {
  return step.scheduled
    .map((s) => (s.hit > 0 ? `${s.id} 命中前缀 ${s.hit} 个 token，算剩下的 ${s.tokens} 个` : s.tokens === 1 ? `${s.id} 算 1 个位置` : `${s.id} 算 ${s.tokens} 个位置`))
    .join('；');
}

function ownerClass(owners: readonly string[]): string {
  if (owners.length > 1) return 'vf-kv-shared';
  if (owners[0] === PAGE_REQUEST) return 'vf-kv-a';
  return owners.length === 1 ? 'vf-kv-b' : 'vf-kv-free';
}

function VllmView(props: { step: PagedStep; blockSize: number }) {
  const blocks = () => props.step.blocks ?? [];
  return (
    <>
      <div class="vf-kv-grid">
        <For each={blocks()}>
          {(block) => (
            <div class={`vf-kv-block ${block.id === 0 ? 'vf-kv-null' : ownerClass(block.owners)}`}>
              <div class="vf-kv-head">
                <span>{block.id === 0 ? '空块' : `块 ${block.id}`}</span>
                <span>{block.refCnt > 0 ? `引用 ${block.refCnt}` : block.hashed ? '可复用' : ''}</span>
              </div>
              <div class="vf-kv-cells" style={{ 'grid-template-columns': `repeat(${props.blockSize}, minmax(0, 1fr))` }}>
                <For each={block.tokens}>{(token) => <span class="vf-kv-cell">{kvTokenText(token)}</span>}</For>
              </div>
            </div>
          )}
        </For>
      </div>
      <table class="vf-gen-cache">
        <thead>
          <tr>
            <th>请求</th>
            <th>块表</th>
            <th>已算</th>
          </tr>
        </thead>
        <tbody>
          <For each={props.step.requests}>
            {(r) => (
              <tr>
                <td>{r.id}</td>
                <td>[{r.table.join(', ')}]</td>
                <td>{r.computed}</td>
              </tr>
            )}
          </For>
        </tbody>
      </table>
    </>
  );
}

function TreeRows(props: { nodes: readonly RadixNodeView[]; id: number; depth: number }) {
  const node = () => props.nodes.find((n) => n.id === props.id)!;
  return (
    <>
      <Show when={node().parent !== null}>
        <div class="vf-kv-node" style={{ 'padding-left': `${(props.depth - 1) * 16}px` }}>
          <span class="vf-kv-key">{node().key.map(kvTokenText).join(' ')}</span>
          <span class="vf-kv-meta">
            槽位 {node().slots.join(',')}；{node().lockRef > 0 ? `${node().lockRef} 个请求在用` : '可被驱逐'}
          </span>
        </div>
      </Show>
      <For each={node().children}>{(child) => <TreeRows nodes={props.nodes} id={child} depth={props.depth + 1} />}</For>
    </>
  );
}

function SglangView(props: { step: PagedStep }) {
  const slots = () => (props.step.slots ?? []).slice(1);
  const ownerOf = (k: number) => {
    const slot = k + 1;
    const users = props.step.requests.filter((r) => r.table.slice(0, r.computed).includes(slot)).map((r) => r.id);
    return users;
  };
  const tree = () => props.step.tree!;
  const root = () => tree().nodes.find((n) => n.parent === null)!.id;
  return (
    <>
      <div class="vf-kv-slots">
        <For each={slots()}>
          {(slot, k) => (
            <div class={`vf-kv-slot ${ownerClass(ownerOf(k()))} ${slot.cached ? 'vf-kv-cached' : ''}`}>
              <span class="vf-kv-index">{k() + 1}</span>
              <span class="vf-kv-cell">{kvTokenText(slot.token)}</span>
            </div>
          )}
        </For>
      </div>
      <div class="vf-kv-tree">
        <div class="vf-kv-tree-title">radix 树</div>
        <TreeRows nodes={tree().nodes} id={root()} depth={0} />
      </div>
      <table class="vf-gen-cache">
        <thead>
          <tr>
            <th>请求</th>
            <th>槽位</th>
            <th>已算</th>
          </tr>
        </thead>
        <tbody>
          <For each={props.step.requests}>
            {(r) => (
              <tr>
                <td>{r.id}</td>
                <td>[{r.table.slice(0, r.computed).join(', ')}]</td>
                <td>{r.computed}</td>
              </tr>
            )}
          </For>
        </tbody>
      </table>
    </>
  );
}

export function KvBlocksPanel(props: { options: KvBlocksSpec; version: Accessor<number>; state: Accessor<StepState> }) {
  const { blockSize, others } = props.options;
  const [engine, setEngine] = createSignal<PagedEngine>('vllm');
  const runs = createMemo(() => {
    props.version();
    const scenario = kvScenario(props.options);
    return { vllm: simulatePages('vllm', scenario), sglang: simulatePages('sglang', scenario) };
  });
  const generation = () => props.state().generation;
  const current = () => {
    const state = generation();
    if (state === undefined) return undefined;
    const steps = runs()[engine()];
    return steps[engineStepAt(steps, state.position)];
  };
  return (
    <>
      <div class="vf-kv-engines" role="tablist">
        <For each={['vllm', 'sglang'] as const}>
          {(name) => (
            <button type="button" role="tab" aria-selected={engine() === name} class={engine() === name ? 'vf-kv-on' : ''} onClick={() => setEngine(name)}>
              {ENGINE_LABEL[name]}
            </button>
          )}
        </For>
      </div>
      <p class="vf-note">
        这一页的请求记为 {PAGE_REQUEST}。
        {others.map((o) => `请求 ${o.id} 的 prompt 是 ${o.prompt.join(' ')}，在 ${PAGE_REQUEST} 的第 ${o.arrive} 个 decode 步到达。`).join('')}
        {others.length > 0 ? '这一页不计算其他请求的值，它们生成的 token 画成 ·。' : ''}
      </p>
      <Show when={current()} fallback={<p class="vf-note">选一个 prefill 或 decode 步骤查看缓存的分配。</p>}>
        {(step) => (
          <>
            <div class="vf-gen-phase">
              引擎第 {step().step + 1} 步：{scheduledText(step())}
            </div>
            <Show when={engine() === 'vllm'} fallback={<SglangView step={step()} />}>
              <VllmView step={step()} blockSize={blockSize} />
            </Show>
          </>
        )}
      </Show>
      <div class="vf-kv-legend">
        <span class="vf-kv-a">{PAGE_REQUEST} 独占</span>
        <span class="vf-kv-b">其他请求独占</span>
        <span class="vf-kv-shared">共享</span>
      </div>
      <p class="vf-note">{ENGINE_NOTE[engine()](blockSize)}</p>
    </>
  );
}
