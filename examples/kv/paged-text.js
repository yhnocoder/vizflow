const ids = (list) => list.map((id) => `#${id}`).join('、');
const short = (hash) => hash.slice(0, 4);
const req = (id) => `请求 ${id}`;

export function tokensOf(spec, view) {
  return [...spec.prompt, ...spec.outputs.slice(0, view.numOutputTokens)];
}

function commonPrefix(a, b) {
  let i = 0;
  while (i < a.length && i < b.length && a[i] === b[i]) i++;
  return i;
}

export function bestSharedPrefix(spec, others) {
  let best = { length: 0, with: null };
  for (const other of others) {
    if (other.id === spec.id) continue;
    const length = commonPrefix(spec.prompt, other.tokens);
    if (length > best.length) best = { length, with: other.id };
  }
  return best;
}

export const V4_GROUP_NAMES = ['CSA 组', 'HCA 组', 'SWA 组'];

function groupLabel(ctx, group) {
  return ctx.groupNames ? `${ctx.groupNames[group]}的` : '';
}

function admitText(event, snap, ctx) {
  const spec = ctx.specs.find((s) => s.id === event.req);
  const view = snap.requests.find((r) => r.id === event.req);
  const total = view.numPromptTokens + view.numOutputTokens;
  const lines = [];
  if (event.resumed) {
    lines.push(
      `调度器从 waiting 队头重新取出被抢占过的${req(event.req)}。抢占时它的 num_computed_tokens 被置 0，所以这次要把 prompt 和已经生成的 ${view.numOutputTokens} 个 token 一起重算，共 ${total} 个 token（scheduler.py 中恢复被抢占请求的路径）。`,
    );
  } else {
    lines.push(`调度器从 waiting 队头取出${req(event.req)}，它的 prompt 有 ${spec.prompt.length} 个 token。`);
  }
  lines.push(
    `get_computed_blocks 按块哈希从第 0 块开始逐块查前缀缓存，遇到第一个查不到的块就停止，最多查到 num_tokens − 1 = ${total - 1} 个 token，因为最后一个 token 必须重新计算才能得到下一个 token 的 logits。结果命中 ${event.hitTokens} 个 token，本步为它计算剩下的 ${event.numNewTokens} 个。`,
  );
  if (ctx.groupNames && snap.schedulerBlockSize > 1) {
    lines.push(
      `这个模型有多个 KV cache 组，块大小分别是 ${ctx.groupBlockSizes.join('、')}。命中长度必须同时落在每个组的块边界上，所以只能是调度块大小 ${snap.schedulerBlockSize}（各组块大小的最小公倍数）的整数倍；哈希按最大公约数 ${snap.hashBlockSize} 个 token 一段计算，各组再按自己的块大小取对应的哈希。`,
    );
  }
  if (event.resumed) {
    lines.push(
      `被抢占时释放的块里，带哈希的块接到了空闲队列队尾。之后没被分配出去的块还保留哈希，现在可以直接命中；已经分配给其他请求的块哈希被删掉了，这些位置要重新计算。`,
    );
  }
  if (!event.resumed) {
    const shared = bestSharedPrefix(spec, ctx.seenTokens(snap));
    const bs = snap.schedulerBlockSize;
    const aligned = Math.floor(shared.length / bs) * bs;
    if (shared.length > event.hitTokens && ctx.groupNames) {
      lines.push(
        `${req(event.req)}的前 ${shared.length} 个 token 与${req(shared.with)}相同，按 ${bs} 对齐向下取整是 ${aligned}，所以命中 ${event.hitTokens} 个 token。各组的块表在命中长度处截断：压缩组取覆盖这 ${event.hitTokens} 个 token 的块，滑动窗口组只需要命中位置之前窗口所覆盖的那几块，更早的位置填 null 块 #0。`,
      );
    } else if (shared.length > event.hitTokens && aligned === event.hitTokens && shared.length % bs !== 0) {
      lines.push(
        `${req(event.req)}的前 ${shared.length} 个 token 与${req(shared.with)}相同，但第 ${shared.length} 个 token（从 0 数）落在 token ${aligned} 到 ${aligned + bs - 1} 这一块的中间。块哈希由父块哈希和整块 ${bs} 个 token 一起算出，这一块两边的内容不同，哈希不同，所以这一块不能复用，只命中前 ${aligned} 个 token。SGLang 的 RadixCache 按 token 匹配，同样的输入能复用到第 ${shared.length} 个 token，切换到 radix 模式可以对照。`,
      );
    } else if (shared.length > event.hitTokens && aligned > event.hitTokens) {
      lines.push(
        `${req(event.req)}的前 ${shared.length} 个 token 与${req(shared.with)}相同，按块对齐最多能命中 ${aligned} 个 token。实际只命中 ${event.hitTokens} 个，因为后面那些块的哈希已经不在缓存表里：块被重新分配时删掉了哈希。`,
      );
    }
  }
  return lines.join('');
}

function hitText(event, snap, ctx, events) {
  const touched = new Set(events.filter((e) => e.kind === 'touch' && e.req === event.req && e.group === event.group).flatMap((e) => e.blockIds));
  const live = event.blockIds.filter((id) => !touched.has(id));
  const revived = event.blockIds.filter((id) => touched.has(id));
  const parts = [`${groupLabel(ctx, event.group)}命中块是 ${ids(event.blockIds)}，按顺序填进${req(event.req)}的块表开头，这些位置的 KV 不用再算。`];
  if (live.length > 0) {
    const users = live.map((id) => snap.blocks[id].refs.filter((r) => r.req !== event.req).map((r) => r.req)).flat();
    parts.push(
      `${ids(live)} 此时正被${[...new Set(users)].map(req).join('、') || '其他请求'}使用，refCnt 加 1，现在是 ${live.map((id) => snap.blocks[id].refCnt).join('、')}，两个请求读同一份物理块。`,
    );
  }
  if (revived.length > 0) {
    parts.push(
      `${ids(revived)} 的 refCnt 原来是 0，它们在空闲队列里但还保留着哈希。touch 把它们从空闲队列中取出并把 refCnt 加 1（block_pool.py touch），它们就不会再被分配给别的请求。`,
    );
  }
  return parts.join('');
}

function allocateText(event, ctx, events) {
  const evicted = events.filter((e) => e.kind === 'evict' && e.reason === 'reuse' && event.blockIds.includes(e.blockId));
  let text = `${req(event.req)}${ctx.groupNames ? `在${ctx.groupNames[event.group]}` : ''}需要 ${event.blockIds.length} 个新块，get_new_blocks 从空闲队列队头依次取出 ${ids(event.blockIds)}，refCnt 置 1，追加到块表末尾。`;
  if (evicted.length === 0) text += '这些块没有哈希，取出时不需要做别的事。';
  return text;
}

function evictText(event, ctx) {
  if (event.reason === 'reuse') {
    return `#${event.blockId} 被取出时还带着哈希 ${short(event.hash)}，里面是以前算好、可以被命中的 KV。块要装新内容，_maybe_evict_cached_block 先把这个哈希从缓存表中删掉，之后的请求再也命中不到这份内容。这就是 vLLM 的驱逐：没有单独的驱逐步骤，分配新块时顺带发生，驱逐的顺序就是空闲队列的顺序。`;
  }
  return `#${event.blockId} 原来的哈希 ${short(event.hash)}（${groupLabel(ctx, event.hashGroup)}）对应更短的前缀，现在它写满了，换成整块的哈希。`;
}

function hashText(event, ctx) {
  if (event.partial) {
    return `${groupLabel(ctx, event.group)}#${event.blockId} 额外登记一个前 ${event.numTokens} 个 token 的哈希 ${short(event.hash)}：prompt 末尾不足一个组块的那一段按哈希块大小也登记一次（_cache_partial_tail_block），这一块因此带两个哈希。`;
  }
  return `${groupLabel(ctx, event.group)}#${event.blockId} 写满了，cache_full_blocks 为它算出块哈希 ${short(event.hash)}（覆盖前 ${event.numTokens} 个 token），登记到缓存表。哈希由父块哈希和本块的 token 一起决定，所以只有从第 0 个 token 起完全相同的序列才能命中它。没写满的块没有哈希。`;
}

function scheduleText(event) {
  const start = event.numComputedTokens;
  const end = start + event.numTokens - 1;
  const what = event.numTokens === 1 ? `位置 ${start} 的 1 个 token（decode）` : `位置 ${start} 到 ${end} 的 ${event.numTokens} 个 token`;
  return `本步计算${req(event.req)}${what}，把它们的 KV 写进块表对应的槽位。`;
}

function freeText(event, snap, ctx) {
  const why = { finish: '结束', preempt: '被抢占', out_of_window: '有块完全移出了滑动窗口' }[event.reason];
  const kept = event.blockIds.filter((id) => !event.toHead.includes(id) && !event.toTail.includes(id));
  if (event.reason === 'out_of_window') {
    let text = `${req(event.req)}已经算到的位置让${groupLabel(ctx, event.group)}${ids(event.blockIds)} 完全落在窗口之外，之后的 attention 不会再读它们。remove_skipped_blocks 释放这些块，块表里对应的位置换成 null 块 #0。`;
    if (event.toTail.length > 0) text += `其中 ${ids(event.toTail)} 有哈希，接到空闲队列队尾，以后同样前缀的请求还可能命中它；`;
    if (event.toHead.length > 0) text += `${ids(event.toHead)} 没有哈希，放到空闲队列队头，最先被重新分配。`;
    return text;
  }
  let text = `${req(event.req)}${why}，free_blocks 按块表倒序释放${groupLabel(ctx, event.group)}块 ${ids(event.blockIds)}（尾部的块先处理），每块 refCnt 减 1。`;
  if (event.toHead.length > 0) text += `refCnt 降到 0 且没有哈希的 ${ids(event.toHead)} 放到空闲队列队头，下次最先被分配。`;
  if (event.toTail.length > 0) text += `refCnt 降到 0 且带哈希的 ${ids(event.toTail)} 按这个顺序接到队尾，尽量晚被重新分配，在这之前它们都还能被命中。`;
  if (kept.length > 0) text += `${ids(kept)} 还被其他请求使用，refCnt 减 1 后不进入空闲队列。`;
  return text;
}

function preemptText(event, snap) {
  const self = event.req === event.for;
  return `${req(event.for)}这一步需要一个新块，但空闲队列已经空了，allocate_slots 返回失败。调度器抢占 running 队列的最后一个请求（running[-1]，在 running 里排得最晚）${self ? `，也就是${req(event.req)}自己` : `，也就是${req(event.req)}`}：释放它的全部块，num_computed_tokens 置 0，状态改为 preempted，放回 waiting 队头，之后按 FCFS 最先恢复。这一步发生了抢占，调度器不再从 waiting 接纳新请求。现在 waiting 是 [${snap.waiting.join('、')}]。`;
}

export function explainVllmEvents(snap, ctx) {
  const events = snap.events;
  const out = [];
  for (const event of events) {
    const blocks = 'blockIds' in event ? [...event.blockIds] : 'blockId' in event ? [event.blockId] : [];
    let text = null;
    if (event.kind === 'arrive') {
      const spec = ctx.specs.find((s) => s.id === event.req);
      text = `${req(event.req)}到达，排到 waiting 队尾。prompt 是「${spec.prompt.join('')}」，共 ${spec.prompt.length} 个 token，最多生成 ${spec.maxTokens} 个 token。`;
    } else if (event.kind === 'admit') text = admitText(event, snap, ctx);
    else if (event.kind === 'hit') text = hitText(event, snap, ctx, events);
    else if (event.kind === 'touch') text = null;
    else if (event.kind === 'allocate') text = allocateText(event, ctx, events);
    else if (event.kind === 'evict') text = evictText(event, ctx);
    else if (event.kind === 'hash') text = hashText(event, ctx);
    else if (event.kind === 'schedule') text = scheduleText(event);
    else if (event.kind === 'free') text = freeText(event, snap, ctx);
    else if (event.kind === 'preempt') text = preemptText(event, snap);
    else if (event.kind === 'finish') {
      text =
        event.reason === 'length'
          ? `${req(event.req)}已经生成 ${ctx.specs.find((s) => s.id === event.req).maxTokens} 个 token，达到 max_tokens，结束。`
          : `${req(event.req)}达到 max_model_len，结束。`;
    }
    if (text !== null) out.push({ kind: event.kind, req: event.req, text, blocks });
  }
  return out;
}

function slotRuns(slots) {
  const out = [];
  let i = 0;
  while (i < slots.length) {
    let j = i;
    while (j + 1 < slots.length && slots[j + 1] === slots[j] + 1) j++;
    out.push(j > i ? `${slots[i]}–${slots[j]}` : `${slots[i]}`);
    i = j + 1;
  }
  return out.join('、');
}

export { slotRuns };

function radixEventText(event, action, nodeKey) {
  switch (event.kind) {
    case 'match':
      return `match_prefix 沿树从根往下逐 token 比较 ${event.tokens} 个 prompt token，匹配到 ${event.matched} 个，停在节点 n${event.lastNode}${nodeKey(event.lastNode)}。匹配过的节点 last_access 更新为当前时钟。`;
    case 'split':
      return `匹配或插入在节点 n${event.node} 的 key 中间结束（第 ${event.at} 个 token 处），_split_node 新建节点 n${event.newNode} 存前 ${event.at} 个 token 和对应槽位，n${event.node} 只留下后半段并成为它的子节点。树的分叉点因此可以落在任意 token 上。`;
    case 'insert':
      return `insert 在节点 n${event.parent} 下新建叶子 n${event.node}，存 ${event.tokens} 个 token 的 key 和它们的槽位。之后前缀相同的请求可以匹配到这些 token。`;
    case 'lock':
      return event.path.length === 0
        ? `inc_lock_ref 加在根节点上，根节点永远不会被驱逐，所以这里没有节点被锁。`
        : `inc_lock_ref 把 ${event.path.map((n) => `n${n}`).join('、')} 这条到根的路径 lock_ref 各加 1。lock_ref 大于 0 的节点不会被驱逐，请求正在用的前缀因此受到保护。`;
    case 'unlock':
      return event.path.length === 0
        ? `dec_lock_ref 作用在根节点上，没有节点被解锁。`
        : `dec_lock_ref 把 ${event.path.map((n) => `n${n}`).join('、')} 的 lock_ref 各减 1。减到 0 且没有子节点的节点成为可驱逐的叶子。`;
    case 'evict':
      return `空闲槽位不够，evict 在 lock_ref 为 0 的叶子里选 last_access 最早的节点 n${event.node} 驱逐，释放槽位 ${slotRuns(event.slots)}。父节点如果因此变成无子节点且未加锁的叶子，也会进入候选。驱逐只删掉树上的这一段，前面的公共前缀还在。`;
    case 'alloc':
      return `从 free_pages 队头取出 ${event.slots.length} 个槽位 ${slotRuns(event.slots)}${action.kind === 'decode' ? `，存「${action.token}」的 KV` : ''}。page size 是 1，每个 token 占一个槽位，没有块内的空位浪费。`;
    case 'free':
      return event.req === null
        ? `被驱逐节点的槽位 ${slotRuns(event.slots)} 追加到 free_pages 队尾。`
        : `请求 ${event.req} 的槽位 ${slotRuns(event.slots)} 与树上已有的内容重复（同样的 token 已经由别的槽位存过），释放后追加到 free_pages 队尾，请求改用树上的那一份。`;
    default:
      return '';
  }
}

export function explainRadixActions(step, specs, nodeKey) {
  const out = [];
  for (const action of step.actions) {
    const spec = specs.find((s) => s.id === action.req);
    let head;
    if (action.kind === 'admit') {
      head = `${req(action.req)}到达并被接纳，prompt 是「${spec.prompt.join('')}」，共 ${action.promptLength} 个 token。树上已有的前缀命中 ${action.prefixLength} 个 token，只需为剩下的 ${action.promptLength - action.prefixLength} 个 token 分配槽位并计算。`;
    } else if (action.kind === 'checkpoint') {
      head = `${req(action.req)}的 prefill 算完，它还没有结束。SGLang 在 prefill 之后调用 cache_unfinished_req，把已经算好的 token 插入树，之后到达的请求马上就能匹配到；请求的锁从原来的末端节点移到新的末端节点。`;
    } else if (action.kind === 'decode') {
      head = `${req(action.req)} decode 一步，计算上一步采样出的「${action.token}」并为它分配 1 个槽位。`;
    } else {
      head = `${req(action.req)}已生成 ${spec.maxTokens} 个 token，结束。release_kv_cache 调用 cache_finished_req：把请求写过 KV 的全部 token 插入树（与树上重复的槽位释放），再在 prompt 边界处插入一次，然后解锁。之后这些节点只要不被别的请求锁住，就会按 LRU 被驱逐。`;
    }
    const nodes = new Set();
    const slots = new Set();
    const ordered = [...action.events];
    for (let i = 0; i + 1 < ordered.length; i++) {
      if (ordered[i].kind === 'free' && ordered[i].req === null && ordered[i + 1].kind === 'evict') [ordered[i], ordered[i + 1]] = [ordered[i + 1], ordered[i]];
    }
    const details = ordered.map((e) => {
      for (const key of ['node', 'newNode', 'lastNode', 'parent']) if (key in e) nodes.add(e[key]);
      if ('path' in e) e.path.forEach((n) => nodes.add(n));
      if ('slots' in e) e.slots.forEach((s) => slots.add(s));
      return { kind: e.kind, text: radixEventText(e, action, nodeKey) };
    });
    out.push({ req: action.req, kind: action.kind, text: head, details, nodes: [...nodes], slots: [...slots] });
  }
  return out;
}
