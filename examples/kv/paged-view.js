import { slotRuns, tokensOf } from './paged-text.js';

const esc = (text) => String(text).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]);
const color = (req) => `var(--req-${req}, var(--muted))`;

export function blockOwners(snaps) {
  let last = new Map();
  return snaps.map((snap) => {
    const next = new Map(last);
    for (const block of snap.blocks) {
      if (block.refs.length > 0) next.set(block.id, block.refs[0]);
      else if (block.hash === null) next.delete(block.id);
    }
    last = next;
    return next;
  });
}

const BADGES = {
  allocate: '分配',
  hit: '命中',
  touch: '移出空闲队列',
  evict: '删除哈希',
  hash: '登记哈希',
  free: '释放',
};

export function blockBadges(snap) {
  const badges = new Map();
  const add = (id, text) => badges.set(id, [...(badges.get(id) ?? []), text]);
  for (const e of snap.events) {
    if (e.kind === 'free') {
      for (const id of e.blockIds) add(id, e.toHead.includes(id) ? '释放→队头' : e.toTail.includes(id) ? '释放→队尾' : '引用减 1');
    } else if ('blockIds' in e) e.blockIds.forEach((id) => add(id, BADGES[e.kind]));
    else if ('blockId' in e) add(e.blockId, BADGES[e.kind]);
  }
  return badges;
}

function groupTokens(spec) {
  return spec.kind === 'compressed' ? spec.tokensPerBlock : spec.blockSize;
}

function slotCells(block, owner, ctx, snap) {
  const spec = block.group === null ? null : ctx.config.groups[block.group];
  if (spec === null || owner === undefined) {
    const n = ctx.config.groups.length === 1 ? groupTokens(ctx.config.groups[0]) : 4;
    return Array.from({ length: n }, () => '<i class="slot"></i>').join('');
  }
  const ownerSpec = ctx.specs.find((s) => s.id === owner.req);
  const ownerView = snap.requests.find((r) => r.id === owner.req);
  const tokens = tokensOf(ownerSpec, ownerView);
  const per = groupTokens(spec);
  const start = owner.index * per;
  const capacity = block.capacity ?? per;
  const reqs = [...new Set(block.refs.map((r) => r.req))];
  const fill = reqs.length > 1 ? `linear-gradient(135deg, ${reqs.map((r, i) => `${color(r)} ${(i * 100) / reqs.length}% ${((i + 1) * 100) / reqs.length}%`).join(', ')})` : color(owner.req);
  return Array.from({ length: capacity }, (_, j) => {
    if (j >= block.filled) return '<i class="slot"></i>';
    if (spec.kind === 'compressed') {
      const a = start + j * spec.ratio;
      return `<i class="slot on entry" style="--fill:${fill}" title="条目 ${j}：压缩 token ${a}–${a + spec.ratio - 1}">${j}</i>`;
    }
    const token = tokens[start + j] ?? '';
    return `<i class="slot on" style="--fill:${fill}" title="位置 ${start + j}">${esc(token)}</i>`;
  }).join('');
}

export function renderPool(snap, owners, ctx) {
  const badges = blockBadges(snap);
  return snap.blocks
    .map((block) => {
      if (block.isNull) {
        return `<div class="blk null" data-block="0"><div class="blk-h"><b>#0</b><span>null 块</span></div><div class="blk-note">保留不用，块表里被跳过的位置指向它</div></div>`;
      }
      const owner = owners.get(block.id);
      const free = block.refCnt === 0;
      const cached = free && block.hash !== null;
      const cls = ['blk', free ? 'free' : 'used', cached ? 'cached' : '', badges.has(block.id) ? 'hl' : ''].join(' ');
      const group = block.group !== null && ctx.groupNames ? `<span class="grp g${block.group}">${ctx.groupShort[block.group]}</span>` : '';
      const refs = block.refs.length > 0 ? block.refs.map((r) => `<span class="who" style="--c:${color(r.req)}">${r.req}[${r.index}]</span>`).join('') : `<span class="muted">${cached ? `空闲，上次属于 ${owner?.req ?? '?'}` : '空闲'}</span>`;
      const hashes = [block.hash, ...block.extraHashes].filter(Boolean);
      const hash = hashes.length > 0 ? hashes.map((h, i) => `<span class="hash" title="${h}${i === 0 && block.hashTokens !== null ? `，覆盖前 ${block.hashTokens} 个 token` : ''}">${h.slice(0, 4)}</span>`).join('') : '<span class="muted">无哈希</span>';
      const badge = (badges.get(block.id) ?? []).map((b) => `<span class="badge">${b}</span>`).join('');
      return `<div class="${cls}" data-block="${block.id}">
<div class="blk-h"><b>#${block.id}</b>${group}<span class="ref">ref ${block.refCnt}</span></div>
<div class="slots">${slotCells(block, owner, ctx, snap)}</div>
<div class="blk-f">${refs}</div>
<div class="blk-f">${hash}${badge}</div>
</div>`;
    })
    .join('');
}

export function renderFreeQueue(snap) {
  const badges = blockBadges(snap);
  if (snap.freeQueue.length === 0) return '<span class="muted">空闲队列为空，下一次分配会失败。</span>';
  return snap.freeQueue
    .map((id) => {
      const block = snap.blocks[id];
      return `<span class="fq ${block.hash ? 'cached' : ''} ${badges.has(id) ? 'hl' : ''}" data-block="${id}" title="${block.hash ? `带哈希 ${block.hash}` : '无哈希'}">#${id}</span>`;
    })
    .join('');
}

const STATUS = { pending: '未到达', waiting: '等待', running: '运行', preempted: '被抢占', finished: '已结束' };

export function renderQueues(snap, specs) {
  const chip = (id) => `<span class="rq" style="--c:${color(id)}">${id}</span>`;
  const list = (ids) => (ids.length > 0 ? ids.map(chip).join('') : '<span class="muted">空</span>');
  const finished = snap.requests.filter((r) => r.status === 'finished').map((r) => r.id);
  const pending = snap.requests.filter((r) => r.status === 'pending').map((r) => `${r.id}（第 ${specs.find((s) => s.id === r.id).arrival} 步）`);
  const scheduled = snap.scheduled.map((s) => `${s.req} ${s.numTokens} 个`).join('，');
  return `<div class="ql"><span class="ql-k">waiting</span>${list(snap.waiting)}</div>
<div class="ql"><span class="ql-k">running</span>${list(snap.running)}</div>
<div class="ql"><span class="ql-k">本步计算</span><span>${scheduled || '无'}</span></div>
<div class="ql"><span class="ql-k">已结束</span>${list(finished)}${pending.length > 0 ? `<span class="muted">　未到达：${pending.join('、')}</span>` : ''}</div>`;
}

export function renderTables(snap, ctx) {
  const rows = snap.requests.filter((r) => r.status !== 'pending' && r.status !== 'finished');
  if (rows.length === 0) return '<p class="muted">没有持有块的请求。</p>';
  return rows
    .map((r) => {
      const spec = ctx.specs.find((s) => s.id === r.id);
      const tokens = tokensOf(spec, r);
      const hit = r.prefixHitTokens ?? 0;
      const text = tokens.map((t, i) => `<span class="${i < hit ? 'hit' : ''} ${i >= r.numComputedTokens ? 'pend' : ''}">${esc(t)}</span>`).join('');
      const tables = r.blockTables
        .map((table, g) => {
          const cells = table.length > 0 ? table.map((id, i) => `<span class="bt ${id === 0 ? 'nullb' : ''}" data-block="${id}"><small>${i}</small>#${id}</span>`).join('') : '<span class="muted">没有块</span>';
          return `<div class="bt-row">${ctx.groupNames ? `<span class="grp g${g}">${ctx.groupShort[g]}</span>` : ''}${cells}</div>`;
        })
        .join('');
      return `<div class="req" style="--c:${color(r.id)}">
<div class="req-h"><span class="rq">${r.id}</span><span>${STATUS[r.status]}</span><span class="muted">已算 ${r.numComputedTokens}/${r.numPromptTokens + r.numOutputTokens} 个 token${r.prefixHitTokens !== null ? `，最近一次命中 ${r.prefixHitTokens}` : ''}${r.numPreemptions > 0 ? `，被抢占 ${r.numPreemptions} 次` : ''}</span></div>
<div class="toks">${text}</div>
${tables}
</div>`;
    })
    .join('');
}

export function renderTree(snapshot, hlNodes) {
  const byId = new Map(snapshot.nodes.map((n) => [n.id, n]));
  const reqAt = new Map();
  for (const r of snapshot.requests) reqAt.set(r.lastNode, [...(reqAt.get(r.lastNode) ?? []), r.id]);
  const node = (n) => {
    const isRoot = n.parent === null;
    const owners = (reqAt.get(n.id) ?? []).map((id) => `<span class="rq" style="--c:${color(id)}">${id}</span>`).join('');
    const label = isRoot
      ? `<b>n${n.id}</b> 根`
      : `<b>n${n.id}</b><span class="key">${esc(n.key.join(''))}</span><span class="meta">lock ${n.lockRef} · t ${n.lastAccess} · 槽 ${slotRuns(n.slots)}${n.evictable ? ' · 可驱逐' : ''}</span>`;
    const kids = n.children.map((c) => byId.get(c)).filter(Boolean);
    return `<li><div class="tn ${hlNodes.has(n.id) ? 'hl' : ''} ${n.lockRef > 0 && !isRoot ? 'locked' : ''}" data-node="${n.id}">${label}${owners ? `<span class="meta">末端节点：</span>${owners}` : ''}</div>${kids.length > 0 ? `<ul>${kids.map(node).join('')}</ul>` : ''}</li>`;
  };
  const root = snapshot.nodes.find((n) => n.parent === null);
  return `<ul class="tree">${node(root)}</ul>`;
}

export function renderSlots(snapshot, numSlots, hlSlots) {
  const owner = new Map();
  for (const r of snapshot.requests) r.slots.forEach((s, i) => owner.set(s, { req: r.id, token: r.tokens[i] }));
  const cached = new Map();
  for (const n of snapshot.nodes) n.slots.forEach((s, i) => cached.set(s, n.key[i]));
  const free = new Set([...snapshot.freeSlots, ...snapshot.releasedSlots]);
  return Array.from({ length: numSlots }, (_, k) => {
    const slot = k + 1;
    const o = owner.get(slot);
    const hl = hlSlots.has(slot) ? 'hl' : '';
    if (o) return `<i class="rs on ${cached.has(slot) ? 'intree' : ''} ${hl}" style="--fill:${color(o.req)}" data-slot="${slot}" title="槽 ${slot}：${o.req}">${esc(o.token)}<small>${slot}</small></i>`;
    if (cached.has(slot)) return `<i class="rs cachedslot ${hl}" data-slot="${slot}" title="槽 ${slot}：只在树上">${esc(cached.get(slot))}<small>${slot}</small></i>`;
    return `<i class="rs ${free.has(slot) ? '' : 'lost'} ${hl}" data-slot="${slot}" title="槽 ${slot}：空闲"><small>${slot}</small></i>`;
  }).join('');
}

export function renderFreeSlots(snapshot) {
  if (snapshot.freeSlots.length === 0) return '<span class="muted">free_pages 为空，下一次分配要先驱逐。</span>';
  return snapshot.freeSlots.map((s) => `<span class="fq" data-slot="${s}">${s}</span>`).join('');
}

export function renderRadixRequests(snapshot) {
  if (snapshot.requests.length === 0) return '<p class="muted">没有进行中的请求。</p>';
  return snapshot.requests
    .map(
      (r) => `<div class="req" style="--c:${color(r.id)}">
<div class="req-h"><span class="rq">${r.id}</span><span class="muted">${r.tokens.length} 个 token 有 KV，前 ${r.cacheProtectedLen} 个在树上并被锁住，末端节点 n${r.lastNode}</span></div>
<div class="rtoks">${r.tokens.map((t, i) => `<span class="${i < r.cacheProtectedLen ? 'hit' : ''}">${esc(t)}<small>${r.slots[i]}</small></span>`).join('')}</div>
</div>`,
    )
    .join('');
}
