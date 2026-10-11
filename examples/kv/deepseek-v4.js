import { graph } from '@yhnocoder/vizflow/core';

export const KINDS = {
  swa: { title: '滑动窗口层', short: 'SWA', layers: '第 0、1、42 层' },
  csa: { title: 'CSA 层（压缩比 4，带 indexer）', short: 'CSA', layers: '第 2、4、…、40 层' },
  hca: { title: 'HCA 层（压缩比 128）', short: 'HCA', layers: '第 3、5、…、41 层' },
};

const decode = (v) =>
  v === 'inf' ? Infinity : v === '-inf' ? -Infinity : Array.isArray(v) ? v.map(decode) : v && typeof v === 'object' ? Object.fromEntries(Object.entries(v).map(([k, x]) => [k, decode(x)])) : v;
const range = (n) => [...Array(n).keys()];

export function windowIdxs(win, t) {
  if (t >= win - 1) {
    const s = t % win;
    return [...range(win).slice(s + 1), ...range(s + 1)];
  }
  return [...range(t + 1), ...Array(win - t - 1).fill(-1)];
}

export function hadamard(n) {
  let h = [[1]];
  while (h.length < n) h = [...h.map((r) => [...r, ...r]), ...h.map((r) => [...r, ...r.map((v) => -v)])];
  return h.map((r) => r.map((v) => v / Math.sqrt(n)));
}

export function stateBefore(fixture, kind, t) {
  const layer = fixture.layers[kind];
  const i = t - fixture.prefill;
  return decode(i === 0 ? layer.initial : layer.steps[i - 1].after);
}

export function stepOf(fixture, kind, t) {
  return fixture.layers[kind].steps[t - fixture.prefill];
}

export function buildStep(fixture, kind, t) {
  const c = fixture.config;
  const layer = fixture.layers[kind];
  const ratio = layer.compress_ratio;
  const W = layer.weights;
  const before = stateBefore(fixture, kind, t);
  const D = c.dim, H = c.n_heads, d = c.head_dim, rd = c.rope_head_dim, R = c.q_lora_rank, win = c.window_size;
  const G = c.o_groups, Or = c.o_lora_rank;
  const Hi = c.index_n_heads, di = c.index_head_dim;
  const overlap = ratio === 4;
  const coff = overlap ? 2 : 1;
  const emits = ratio > 0 && (t + 1) % ratio === 0;
  const nCache = win + (ratio ? c.max_seq_len / ratio : 0);
  const cos = layer.cos[t], sin = layer.sin[t];

  const g = graph({
    dims: { D, H, d, R, win, n: nCache, G },
  });
  const N = {};
  const tree = [];
  const steps = [];
  const leaf = (node) => ({ kind: 'node', node });
  const mod = (title, subtitle, nodes) => ({ kind: 'module', title, subtitle, children: nodes.map((n) => (n.kind ? n : leaf(n))) });
  const T = (node) => node.T;

  function rms(x, weight, name, tex) {
    const sq = g.pow(x, 2, { name: `${name}²`, tex: `{${tex}}^{2}`, elem: `{${tex}}^{2}` });
    const ms = g.mean(sq, -1, { keepdim: true, name: `ms(${name})`, tex: `\\mathrm{ms}(${tex})`, elem: `\\mathrm{ms}` });
    const r = g.rsqrt(ms, { eps: c.norm_eps ?? 1e-6, name: `r(${name})`, tex: `r_{${tex}}`, elem: 'r' });
    const hat = g.mul(x, r, { name: `${name}̂`, tex: `\\hat{${tex}}`, elem: `\\hat{${tex}}` });
    const out = weight ? g.mul(hat, weight, { name: `norm(${name})`, tex: `\\overline{${tex}}`, elem: `\\overline{${tex}}` }) : hat;
    return { nodes: [sq, ms, r, hat, ...(weight ? [weight, out] : [])], out };
  }

  function rope(x, cs, sn, name, tex, width) {
    const cosB = g.buffer(`cos(${name})`, [1, rd], [Array(rd).fill(cs[0])], { tex: '\\cos', elem: '\\cos' });
    const sinB = g.buffer(`sin(${name})`, [1, rd], [Array(rd).fill(sn[0])], { tex: '\\sin', elem: '\\sin' });
    const nope = g.narrow(x, -1, 0, width - rd, { name: `${name} nope`, tex: `${tex}^{\\mathrm{nope}}`, elem: `${tex}^{\\mathrm{nope}}` });
    const pe = g.narrow(x, -1, width - rd, rd, { name: `${name} pe`, tex: `${tex}^{\\mathrm{pe}}`, elem: `${tex}^{\\mathrm{pe}}` });
    const a = g.mul(pe, cosB, { name: `${name} pe·cos`, tex: `${tex}^{\\mathrm{pe}}\\cos`, elem: `(${tex}^{\\mathrm{pe}}\\cos)` });
    const rh = g.rotate_half(pe, { name: `rot(${name} pe)`, tex: `\\mathrm{rot}(${tex}^{\\mathrm{pe}})`, elem: `\\mathrm{rot}` });
    const b = g.mul(rh, sinB, { name: `${name} rot·sin`, tex: `\\mathrm{rot}\\sin`, elem: `(\\mathrm{rot}\\sin)` });
    const roped = g.add(a, b, { name: `${name} pe roped`, tex: `\\tilde{${tex}}^{\\mathrm{pe}}`, elem: `\\tilde{${tex}}^{\\mathrm{pe}}` });
    const out = g.cat([nope, roped], -1, { name: `${name} roped`, tex: `\\tilde{${tex}}`, elem: `\\tilde{${tex}}` });
    return { nodes: [nope, pe, cosB, a, rh, sinB, b, roped, out], out };
  }

  const x = g.input('x', [1, 'D'], [fixture.x[t]], { tex: '\\mathbf{x}_t', elem: 'x' });
  const cache0 = g.input('kv_cache', ['n', 'd'], before.kv_cache, { tex: '\\mathcal{C}', elem: '\\mathcal{C}' });
  N.x = x;
  N.cache0 = cache0;
  steps.push({ show: [x, cache0], kind: 'inputs', key: 'inputs' });

  const wqa = g.param('wq_a', ['R', 'D'], W['wq_a.weight'], { tex: 'W^{qa}', elem: 'W^{qa}' });
  const qa = g.matmul(x, T(wqa), { name: 'q_a', tex: '\\mathbf{q}^{a}', elem: 'q^{a}' });
  const qn = rms(qa, g.param('q_norm', ['R'], W['q_norm.weight'], { tex: '\\gamma^{q}', elem: '\\gamma^{q}' }), 'q_a', '\\mathbf{q}^{a}');
  const qr = qn.out;
  const wqb = g.param('wq_b', [H * d, 'R'], W['wq_b.weight'], { tex: 'W^{qb}', elem: 'W^{qb}' });
  const qflat = g.matmul(qr, T(wqb), { name: 'q_b', tex: '\\mathbf{q}^{b}', elem: 'q^{b}' });
  const qh = g.view(qflat, [H, d], { name: 'q heads', tex: 'Q', elem: 'Q' });
  const qhn = rms(qh, null, 'q', 'Q');
  const qrope = rope(qhn.out, cos, sin, 'q', 'Q', d);
  const q = qrope.out;
  Object.assign(N, { qa, qr, qflat, qh, q });
  tree.push(mod('q', 'wq_a → q_norm → wq_b → 逐 head 归一化 → RoPE', [wqa, qa, ...qn.nodes, wqb, qflat, qh, ...qhn.nodes, ...qrope.nodes]));
  steps.push({ show: [wqa, qa, ...qn.nodes], key: 'qa' });
  steps.push({ show: [wqb, qflat, qh], key: 'qb' });
  steps.push({ show: qhn.nodes, key: 'qnorm' });
  steps.push({ show: qrope.nodes, key: 'qrope' });

  const wkv = g.param('wkv', ['d', 'D'], W['wkv.weight'], { tex: 'W^{kv}', elem: 'W^{kv}' });
  const kv0 = g.matmul(x, T(wkv), { name: 'kv', tex: '\\mathbf{kv}', elem: 'kv' });
  const kvn = rms(kv0, g.param('kv_norm', ['d'], W['kv_norm.weight'], { tex: '\\gamma^{kv}', elem: '\\gamma^{kv}' }), 'kv', '\\mathbf{kv}');
  const kvr = rope(kvn.out, cos, sin, 'kv', '\\mathbf{kv}', d);
  const kv = kvr.out;
  N.kv = kv;
  tree.push(mod('kv', 'wkv → kv_norm → RoPE，K 与 V 共用', [wkv, kv0, ...kvn.nodes, ...kvr.nodes]));
  steps.push({ show: [wkv, kv0, ...kvn.nodes], key: 'kv' });
  steps.push({ show: kvr.nodes, key: 'kvrope' });

  const widx = windowIdxs(win, t);
  const winIdx = g.label('window idxs', [widx], { classes: nCache, tex: 'I^{\\mathrm{win}}', elem: 'I^{\\mathrm{win}}' });
  N.winIdx = winIdx;
  steps.push({ show: winIdx, key: 'widx' });

  function compressor(prefix, wts, dim, stateKv, stateScore, cache, cacheRow, rotate, texName) {
    const P = (k) => wts[`${prefix}${k}`];
    const nodes = [];
    const st = [];
    const wkvC = g.param(`${prefix}wkv`, [coff * dim, 'D'], P('wkv.weight'), { tex: `W^{${texName}}_{kv}`, elem: `W^{${texName}}_{kv}` });
    const wgate = g.param(`${prefix}wgate`, [coff * dim, 'D'], P('wgate.weight'), { tex: `W^{${texName}}_{z}`, elem: `W^{${texName}}_{z}` });
    const ape = g.param(`${prefix}ape`, [ratio, coff * dim], P('ape'), { tex: `B^{${texName}}`, elem: `B^{${texName}}` });
    const ckv = g.matmul(x, T(wkvC), { name: `${prefix}kv`, tex: `\\mathbf{c}^{${texName}}`, elem: `c^{${texName}}` });
    const cz0 = g.matmul(x, T(wgate), { name: `${prefix}score`, tex: `\\mathbf{z}^{${texName}}`, elem: `z^{${texName}}` });
    const apeRow = g.narrow(ape, 0, t % ratio, 1, { name: `${prefix}ape[t mod m]`, tex: `B^{${texName}}_{${t % ratio}}`, elem: `B^{${texName}}_{${t % ratio}}` });
    const cz = g.add(cz0, apeRow, { name: `${prefix}score+ape`, tex: `\\mathbf{z}^{${texName}}+B`, elem: `z^{${texName}}` });
    const sKv = g.input(`${prefix}kv_state`, [coff * ratio, coff * dim], stateKv, { tex: `S^{${texName}}_{kv}`, elem: `S^{${texName}}_{kv}` });
    const sZ = g.input(`${prefix}score_state`, [coff * ratio, coff * dim], stateScore, { tex: `S^{${texName}}_{z}`, elem: `S^{${texName}}_{z}` });
    const row = (overlap ? ratio : 0) + (t % ratio);
    const at = g.label(`${prefix}state row`, [row], { classes: coff * ratio, tex: `${row}`, elem: `${row}` });
    const sKv1 = g.index_copy(sKv, 0, at, ckv, { name: `${prefix}kv_state'`, tex: `S^{${texName}\\prime}_{kv}`, elem: `S^{${texName}\\prime}_{kv}` });
    const sZ1 = g.index_copy(sZ, 0, at, cz, { name: `${prefix}score_state'`, tex: `S^{${texName}\\prime}_{z}`, elem: `S^{${texName}\\prime}_{z}` });
    nodes.push(wkvC, ckv, wgate, cz0, ape, apeRow, cz, sKv, sZ, at, sKv1, sZ1);
    st.push({ show: [wkvC, ckv, wgate, cz0, ape, apeRow, cz], key: `${prefix}proj` });
    st.push({ show: [sKv, sZ, at, sKv1, sZ1], key: `${prefix}state` });
    let cacheOut = cache;
    let entry = null;
    if (emits) {
      let pool, poolZ;
      const windowNodes = [];
      if (overlap) {
        const halves = (s, nm, tx) => {
          const prevRows = g.narrow(s, 0, 0, ratio, { name: `${nm} prev rows`, tex: `${tx}[:m]`, elem: tx });
          const prev = g.narrow(prevRows, 1, 0, dim, { name: `${nm} prev a`, tex: `${tx}[:m,:d]`, elem: `${tx}^{a}` });
          const curRows = g.narrow(s, 0, ratio, ratio, { name: `${nm} cur rows`, tex: `${tx}[m:]`, elem: tx });
          const cur = g.narrow(curRows, 1, dim, dim, { name: `${nm} cur b`, tex: `${tx}[m:,d:]`, elem: `${tx}^{b}` });
          const both = g.cat([prev, cur], 0, { name: `${nm} window`, tex: `\\hat{${tx}}`, elem: `\\hat{${tx}}` });
          nodes.push(prevRows, prev, curRows, cur, both);
          windowNodes.push(prevRows, prev, curRows, cur, both);
          return both;
        };
        pool = halves(sKv1, `${prefix}kv`, `S_{kv}`);
        poolZ = halves(sZ1, `${prefix}score`, `S_{z}`);
      } else {
        pool = sKv1;
        poolZ = sZ1;
      }
      const w = g.softmax(poolZ, 0, { name: `${prefix}weights`, tex: `\\mathrm{softmax}(\\hat{S}_{z})`, elem: 'w' });
      const prod = g.mul(pool, w, { name: `${prefix}weighted`, tex: `w\\odot\\hat{S}_{kv}`, elem: '(w\\hat{S})' });
      const pooled = g.sum(prod, 0, { keepdim: true, name: `${prefix}pooled`, tex: `\\mathbf{e}^{${texName}}`, elem: `e^{${texName}}` });
      const en = rms(pooled, g.param(`${prefix}norm`, [dim], P('norm.weight'), { tex: `\\gamma^{${texName}}`, elem: `\\gamma^{${texName}}` }), `${prefix}entry`, `\\mathbf{e}^{${texName}}`);
      const pos = t + 1 - ratio;
      const er = rope(en.out, layer.cos[pos], layer.sin[pos], `${prefix}entry`, `\\mathbf{e}^{${texName}}`, dim);
      entry = er.out;
      const extra = [];
      if (rotate) {
        const Hm = g.buffer(`${prefix}hadamard`, [dim, dim], hadamard(dim), { tex: 'H', elem: 'H' });
        entry = g.matmul(er.out, Hm, { name: `${prefix}entry rotated`, tex: `\\mathbf{e}^{${texName}}H`, elem: `(eH)` });
        extra.push(Hm, entry);
      }
      const slot = g.label(`${prefix}entry slot`, [cacheRow], { classes: cache.shape[0], tex: `${cacheRow}`, elem: `${cacheRow}` });
      cacheOut = g.index_copy(cache, 0, slot, entry, { name: `${cache.name}'`, tex: `${cache.tex}'`, elem: `${cache.tex}'` });
      nodes.push(w, prod, pooled, ...en.nodes, ...er.nodes, ...extra, slot, cacheOut);
      st.push({ show: [...windowNodes, w], key: `${prefix}softmax` });
      st.push({ show: [prod, pooled, ...en.nodes], key: `${prefix}pool` });
      st.push({ show: [...er.nodes, ...extra], key: `${prefix}rope` });
      st.push({ show: [slot, cacheOut], key: `${prefix}write` });
      return { nodes, steps: st, cache: cacheOut, entry, sKv1, sZ1 };
    }
    return { nodes, steps: st, cache: cacheOut, entry, sKv1, sZ1 };
  }

  let compIdx = null;
  let idxParts = null;
  if (ratio === 4) {
    const idxCache0 = g.input('indexer kv_cache', [c.max_seq_len / ratio, di], before.index_cache, { tex: '\\mathcal{K}^{I}', elem: '\\mathcal{K}^{I}' });
    const wqbI = g.param('indexer.wq_b', [Hi * di, 'R'], W['indexer.wq_b.weight'], { tex: 'W^{Iq}', elem: 'W^{Iq}' });
    const qI0 = g.matmul(qr, T(wqbI), { name: 'indexer q', tex: '\\mathbf{q}^{I}', elem: 'q^{I}' });
    const qIh = g.view(qI0, [Hi, di], { name: 'indexer q heads', tex: 'Q^{I}', elem: 'Q^{I}' });
    const qIr = rope(qIh, cos, sin, 'indexer q', 'Q^{I}', di);
    const Hm = g.buffer('indexer hadamard q', [di, di], hadamard(di), { tex: 'H', elem: 'H' });
    const qI = g.matmul(qIr.out, Hm, { name: 'indexer q rotated', tex: 'Q^{I}H', elem: '(Q^{I}H)' });
    const comp = compressor('indexer.compressor.', W, di, before.index_kv_state, before.index_score_state, idxCache0, Math.floor(t / ratio), true, 'I');
    const idxCache = comp.cache;
    const visible = Math.floor((t + 1) / ratio);
    const seen = g.narrow(idxCache, 0, 0, visible, { name: 'indexer keys', tex: '\\mathcal{K}^{I}[:\\lfloor (t+1)/m\\rfloor]', elem: '\\mathcal{K}^{I}' });
    const wproj = g.param('indexer.weights_proj', [Hi, 'D'], W['indexer.weights_proj.weight'], { tex: 'W^{Iw}', elem: 'W^{Iw}' });
    const hw0 = g.matmul(x, T(wproj), { name: 'indexer head weights', tex: '\\mathbf{w}^{I}', elem: 'w^{I}' });
    const hw = g.mul(hw0, Math.pow(di, -0.5) * Math.pow(Hi, -0.5), { name: 'indexer head weights scaled', tex: '\\mathbf{w}^{I}/\\sqrt{d_I H_I}', elem: 'w' });
    const dots = g.matmul(qI, T(seen), { name: 'indexer dots', tex: 'Q^{I}H\\,\\mathcal{K}^{I\\top}', elem: '(qk)' });
    const relu = g.relu(dots, { name: 'indexer relu', tex: '\\mathrm{ReLU}(\\cdot)', elem: '\\mathrm{ReLU}' });
    const weighted = g.mul(relu, T(hw), { name: 'indexer weighted', tex: 'w_h\\,\\mathrm{ReLU}', elem: '(w\\,\\mathrm{ReLU})' });
    const score = g.sum(weighted, 0, { keepdim: true, name: 'index score', tex: '\\mathbf{s}^{I}', elem: 's^{I}' });
    const k = Math.min(c.index_topk, visible);
    const top = g.topk_indices(score, k, -1, { name: 'indexer topk', tex: '\\operatorname{topk}(\\mathbf{s}^{I})', elem: 'j' });
    compIdx = g.add(top, win, { name: 'compressed idxs', tex: 'I^{\\mathrm{cmp}}', elem: 'I^{\\mathrm{cmp}}' });
    idxParts = { idxCache0, wqbI, qI0, qIh, qIr, Hm, qI, comp, seen, wproj, hw0, hw, dots, relu, weighted, score, top };
    Object.assign(N, { idxCache: comp.cache, indexScore: score, compIdx, indexerEntry: comp.entry });
    tree.push(mod('indexer', 'Lightning Indexer：自己的压缩 key cache 与 top-k', [idxCache0, wqbI, qI0, qIh, ...qIr.nodes, Hm, qI, ...comp.nodes, seen, wproj, hw0, hw, dots, relu, weighted, score, top, compIdx]));
    steps.push({ show: [wqbI, qI0, qIh, ...qIr.nodes, Hm, qI], key: 'iq' });
    steps.push({ show: idxCache0, key: 'icache' });
    for (const s of comp.steps) steps.push({ ...s, key: `i.${s.key}` });
    steps.push({ show: [seen, wproj, hw0, hw, dots, relu, weighted, score], key: 'iscore' });
    steps.push({ show: [top, compIdx], key: 'itopk' });
  } else if (ratio > 0) {
    const n = Math.floor((t + 1) / ratio);
    compIdx = g.label('compressed idxs', [range(n).map((i) => i + win)], { classes: nCache, tex: 'I^{\\mathrm{cmp}}', elem: 'I^{\\mathrm{cmp}}' });
    N.compIdx = compIdx;
    steps.push({ show: compIdx, key: 'hidx' });
  }

  const idx = compIdx ? g.cat([winIdx, compIdx], -1, { name: 'topk_idxs', tex: 'I', elem: 'I' }) : winIdx;
  N.idx = idx;
  if (compIdx) steps.push({ show: idx, key: 'idx' });

  const wslot = g.label('window slot', [t % win], { classes: nCache, tex: `${t % win}`, elem: `${t % win}` });
  const cache1 = g.index_copy(cache0, 0, wslot, kv, { name: "kv_cache'", tex: "\\mathcal{C}'", elem: "\\mathcal{C}'" });
  N.cache1 = cache1;
  steps.push({ show: [wslot, cache1], key: 'wwrite' });

  let cacheF = cache1;
  let mainComp = null;
  if (ratio > 0) {
    mainComp = compressor('compressor.', W, d, before.kv_state, before.score_state, cache1, win + Math.floor(t / ratio), false, 'C');
    cacheF = mainComp.cache;
    for (const s of mainComp.steps) steps.push({ ...s, key: `c.${s.key}` });
    N.entry = mainComp.entry;
  }
  N.cacheF = cacheF;

  const idxFlat = g.view(idx, [-1], { name: 'idx list', tex: 'I', elem: 'I' });
  const rows = g.embedding(idxFlat, cacheF, { name: 'gathered kv', tex: '\\mathcal{C}[I]', elem: '\\mathcal{C}[I]' });
  const scores0 = g.matmul(q, T(rows), { name: 'scores', tex: 'Q\\,\\mathcal{C}[I]^{\\top}', elem: '(QK^{\\top})' });
  const scores = g.mul(scores0, Math.pow(d, -0.5), { name: 'scaled scores', tex: '\\mathbf{s}', elem: 's' });
  const sink = g.param('attn_sink', ['H', 1], W.attn_sink.map((v) => [v]), { tex: '\\mathbf{b}^{\\mathrm{sink}}', elem: 'b^{\\mathrm{sink}}' });
  const withSink = g.cat([scores, sink], -1, { name: 'scores with sink', tex: '[\\mathbf{s};\\,b^{\\mathrm{sink}}]', elem: '[s;b]' });
  const pAll = g.softmax(withSink, -1, { name: 'softmax with sink', tex: '\\mathrm{softmax}([\\mathbf{s};b])', elem: 'p' });
  const K = idxFlat.shape[0];
  const p = g.narrow(pAll, -1, 0, K, { name: 'probs', tex: 'P', elem: 'P' });
  const o0 = g.matmul(p, rows, { name: 'o', tex: 'O', elem: 'O' });
  const negSin = sin.map((v) => -v);
  const oInv = rope(o0, cos, negSin, 'o', 'O', d);
  Object.assign(N, { rows, scores, sink, pAll, p, o0, o: oInv.out });
  steps.push({ show: [idxFlat, rows], key: 'gather' });
  steps.push({ show: [scores0, scores], key: 'scores' });
  steps.push({ show: [sink, withSink, pAll, p], key: 'softmax' });
  steps.push({ show: o0, key: 'o' });
  steps.push({ show: oInv.nodes, key: 'orope' });

  const oGroups = g.view(oInv.out, [G, (H * d) / G], { name: 'o groups', tex: 'O^{g}', elem: 'O^{g}' });
  const woa = g.param('wo_a', [G * Or, (H * d) / G], W['wo_a.weight'], { tex: 'W^{oa}', elem: 'W^{oa}' });
  const groupOuts = range(G).map((gi) => {
    const og = g.narrow(oGroups, 0, gi, 1, { name: `o group ${gi}`, tex: `O^{(${gi})}`, elem: `O^{(${gi})}` });
    const wg = g.narrow(woa, 0, gi * Or, Or, { name: `wo_a group ${gi}`, tex: `W^{oa}_{(${gi})}`, elem: `W^{oa}_{(${gi})}` });
    const yg = g.matmul(og, T(wg), { name: `o_a ${gi}`, tex: `\\mathbf{u}^{(${gi})}`, elem: `u^{(${gi})}` });
    return { og, wg, yg };
  });
  const u = g.cat(groupOuts.map((o) => o.yg), -1, { name: 'o_a', tex: '\\mathbf{u}', elem: 'u' });
  const wob = g.param('wo_b', ['D', G * Or], W['wo_b.weight'], { tex: 'W^{ob}', elem: 'W^{ob}' });
  const y = g.matmul(u, T(wob), { name: 'y', tex: '\\mathbf{y}', elem: 'y' });
  N.y = y;
  steps.push({ show: [oGroups, woa, ...groupOuts.flatMap((o) => [o.og, o.wg, o.yg]), u], key: 'oa' });
  steps.push({ show: [wob, y], key: 'ob' });

  tree.splice(2, 0, mod('window', 'get_window_topk_idxs', [winIdx]));
  tree.push(mod('kv_cache', '写入窗口槽位', [...(compIdx && ratio !== 4 ? [compIdx] : []), ...(compIdx ? [idx] : []), wslot, cache1]));
  if (mainComp) tree.push(mod(ratio === 4 ? 'compressor（CSA）' : 'compressor（HCA）', ratio === 4 ? '重叠窗口：上一窗口的 a 半与本窗口的 b 半' : '不重叠窗口', mainComp.nodes));
  tree.push(mod('sparse_attn', '按下标取行，带 sink 的 softmax', [idxFlat, rows, scores0, scores, sink, withSink, pAll, p, o0, ...oInv.nodes]));
  tree.push(mod('o', '分组低秩输出投影 wo_a → wo_b', [oGroups, woa, ...groupOuts.flatMap((o) => [o.og, o.wg, o.yg]), u, wob, y]));

  return { g, N, steps, tree: { kind: 'module', title: `Attention（${KINDS[kind].short}）`, subtitle: `t = ${t}`, children: tree }, info: { ratio, emits, win, nCache, widx, idxParts, mainComp, K } };
}
