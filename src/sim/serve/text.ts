import type { BatchRow, EngineEvent, EngineName, EngineRun, EngineStep, RequestView } from './types';

export const ENGINE_LABEL: Record<EngineName, string> = { vllm: 'vLLM', sglang: 'SGLang' };

function list(ids: string[]): string {
  return ids.join('、');
}

export function describeEvent(e: EngineEvent, engine: EngineName): string {
  const unit = engine === 'vllm' ? '步' : '批';
  switch (e.kind) {
    case 'launch':
      return `调度第 ${e.batch} ${unit}，发给 GPU`;
    case 'process':
      return e.ids.length === 0
        ? `处理第 ${e.batch} ${unit}的结果：没有输出 token`
        : `处理第 ${e.batch} ${unit}的结果：${list(e.ids)}${e.ids.length > 1 ? ' 各' : ' '}得到一个 token`;
    case 'preempt':
      return `${list(e.ids)} 被抢占：释放全部 KV 块，游标归零，回到等待队列最前`;
    case 'retract':
      return `${list(e.ids)} 被撤回：KV 不够下一步 decode，释放 KV，回到等待队列`;
    case 'resume':
      return `${list(e.ids)} 重新开始，前缀缓存命中 ${e.hit} 个 token`;
    case 'blocked':
      return `${list(e.ids)} 放不下，留在等待队列`;
    case 'reject':
      return `${list(e.ids)} 的提示词超过 KV 容量允许的输入长度，请求被中止，提示词换成 1 个 token 走完一次 prefill 后结束`;
    case 'abort':
      return `${list(e.ids)} 被中止：撤回其他请求后 KV 仍不够它 decode，释放 KV`;
    case 'finish':
      return `${list(e.ids)} 生成完毕，释放 KV`;
  }
}

function rowSentence(row: BatchRow, view: RequestView, engine: EngineName, admitted: boolean): string {
  const known = view.prompt + view.outputs;
  const prefixEnd = row.from + row.n;
  const parts: string[] = [];
  if (admitted && row.from > 0) parts.push(`前缀缓存命中 ${row.from} 个 token`);
  if (prefixEnd < view.prompt) parts.push(`算位置 ${row.from} 到 ${prefixEnd - 1}，提示词没算完，这一块采样的 token 丢掉`);
  else parts.push(`算位置 ${row.from} 到 ${prefixEnd - 1}，提示词算完，采样第一个 token`);
  if (row.duplicate.length > 0) parts.push(`其中 ${row.duplicate.length} 个 token 别的请求或缓存里已有，这里重算`);
  if (engine === 'vllm' && prefixEnd > known) parts.push('包括一个还在 GPU 上的已生成 token');
  return `${row.id} ${parts.join('；')}`;
}

export function narrate(run: EngineRun, index: number): string[] {
  const step: EngineStep = run.steps[index];
  const before = index > 0 ? run.steps[index - 1] : null;
  const viewOf = new Map(step.requests.map((v) => [v.id, v]));
  const lines = step.rows.filter((r) => r.from < viewOf.get(r.id)!.prompt).map((row) => {
    const prevView = before?.requests.find((v) => v.id === row.id);
    return rowSentence(row, viewOf.get(row.id)!, run.engine, prevView === undefined || prevView.status !== 'running');
  });
  const decodes = step.rows.filter((r) => r.from >= viewOf.get(r.id)!.prompt);
  if (decodes.length > 0) lines.push(`${list(decodes.map((r) => r.id))} 各 decode 1 个 token`);
  if (run.engine === 'sglang' && step.mode === 'prefill') {
    const stalled = step.requests.filter((v) => v.status === 'running' && !step.rows.some((r) => r.id === v.id) && v.outputs + v.inflight > 0);
    if (stalled.length > 0) lines.push(`${list(stalled.map((v) => v.id))} 已经在 decode，但这一步有 prefill，prefill 优先，它们等一步`);
  }
  const waitingIds = step.requests.filter((v) => v.status === 'waiting' || v.status === 'preempted').map((v) => v.id);
  if (waitingIds.length > 0) lines.push(`等待队列：${list(waitingIds)}`);
  return lines;
}

export function graphText(step: EngineStep, engine: EngineName): string {
  const size = step.mode === 'decode' && engine === 'sglang' ? step.rows.length : step.tokens;
  const unit = step.mode === 'decode' && engine === 'sglang' ? '行' : '个 token';
  switch (step.graph) {
    case 'none':
      return '没有可算的 token';
    case 'eager':
      return `不用 CUDA graph（${size} ${unit}没有合适的捕获尺寸）`;
    case 'full':
      return `整图，${size} ${unit}填充到 ${size + step.padded}`;
    case 'piecewise':
      return `分段图，${size} ${unit}填充到 ${size + step.padded}`;
    case 'decode':
      return `decode 图，${size} ${unit}填充到 ${size + step.padded}`;
    case 'prefill':
      return `prefill 图，${size} ${unit}填充到 ${size + step.padded}`;
  }
}

export const MODE_LABEL: Record<EngineStep['mode'], string> = { prefill: 'prefill 批', mixed: '混合批', decode: 'decode 批', idle: '空闲' };
