import { type Graph, type Node, graph } from '../../autograd';
import { BATCH_ROW_SPACING, BOARD_BACK_OFFSET, BOARD_FRONT, type LayoutItem, type Vec3, board, stack } from '../../scene3d/layout';
import type { ForwardStep, StepsSpec } from '../../steps';
import type { PanelItem } from '../../ui';
import { mulberry32 } from '../../tensor';

export interface PackedRow {
  id: string;
  cached: number;
  n: number;
}

export interface PackedOptions {
  d?: number;
  seed?: number;
}

export interface PackedRequest extends PackedRow {
  start: number;
  keyStart: number;
  last: number;
}

export interface PackedNodes {
  x: Node;
  q: Node;
  k: Node;
  v: Node;
  K: Node;
  V: Node;
  S: Node;
  M: Node;
  P: Node;
  O: Node;
  logits: Node;
  kCache: Map<string, Node>;
  vCache: Map<string, Node>;
}

export interface PackedPage {
  graph: Graph;
  steps: StepsSpec;
  layout: LayoutItem[];
  panels: PanelItem[];
  requests: PackedRequest[];
  nodes: PackedNodes;
}

const DEFAULT_D = 4;
const DEFAULT_SEED = 1;
const SCALE = 0.65;
const LEVEL = 4.2;
const K_X = -15;
const Q_X = 0;
const V_X = 15;
const PIECE_SHIFT = 3.5;
const LAST_SHIFT = 8;


function uniformMatrix(random: () => number, rows: number, cols: number): number[][] {
  return Array.from({ length: rows }, () => Array.from({ length: cols }, () => Math.round((random() * 2 - 1) * 100) / 100));
}

function linear(x: number[][], W: number[][]): number[][] {
  return x.map((row) => W.map((w) => w.reduce((sum, wi, i) => sum + wi * row[i], 0)));
}

export function packedRequests(rows: readonly PackedRow[]): PackedRequest[] {
  if (rows.length === 0) throw new Error('packed: the batch needs at least one request');
  const seen = new Set<string>();
  let start = 0;
  let keyStart = 0;
  return rows.map((row) => {
    if (!/^[A-Za-z0-9]+$/.test(row.id)) throw new Error(`packed: request id '${row.id}' must be letters or digits`);
    if (seen.has(row.id)) throw new Error(`packed: request id '${row.id}' appears twice`);
    seen.add(row.id);
    if (!Number.isInteger(row.cached) || row.cached < 0) throw new Error(`packed: request '${row.id}' has cached = ${row.cached}, expected an integer >= 0`);
    if (!Number.isInteger(row.n) || row.n < 1) throw new Error(`packed: request '${row.id}' has n = ${row.n}, expected an integer >= 1`);
    const request = { ...row, start, keyStart, last: start + row.n - 1 };
    start += row.n;
    keyStart += row.cached + row.n;
    return request;
  });
}

export function packedMask(requests: readonly PackedRequest[]): number[][] {
  const total = requests.reduce((sum, r) => sum + r.cached + r.n, 0);
  return requests.flatMap((r) =>
    Array.from({ length: r.n }, (_, j) => {
      const row = new Array<number>(total).fill(-Infinity);
      for (let p = 0; p <= r.cached + j; p++) row[r.keyStart + p] = 0;
      return row;
    }),
  );
}

const range = (from: number, to: number) => (from === to ? `第 ${from}` : `第 ${from} 到 ${to}`);

function kindOf(r: PackedRow): string {
  if (r.cached === 0) return 'prefill';
  return r.n === 1 ? 'decode' : '分块 prefill 的后续块';
}

function requestLine(r: PackedRequest): string {
  const rows = `占打包后的${range(r.start + 1, r.last + 1)} 行`;
  const tokens = `${range(r.cached + 1, r.cached + r.n)} 个 token`;
  const cache = r.cached === 0 ? 'KV cache 里还没有它的 token' : `KV cache 里已有 ${r.cached} 个 token`;
  return `<li>请求 ${r.id}（${kindOf(r)}）：${cache}，这一步新增 ${r.n} 个，即它的${tokens}，${rows}。</li>`;
}

function visibility(r: PackedRequest): string {
  const cache = r.cached === 0 ? '没有缓存列' : `${r.cached} 个缓存列全部可见`;
  const fresh = r.n === 1 ? '新列只有它自己' : `第 $j$ 个新 token 看到前 $j$ 个新列`;
  return `<li>请求 ${r.id}：${r.n} 行 query，${cache}，${fresh}。</li>`;
}

function maskTable(requests: readonly PackedRequest[]): string {
  const columns = `r|${requests.map((r) => `${'c'.repeat(r.cached)}${r.cached > 0 ? ':' : ''}${'c'.repeat(r.n)}`).join('|')}`;
  const header = ['', ...requests.flatMap((r) => Array.from({ length: r.cached + r.n }, (_, p) => `${r.id}_{${p + 1}}`))].join('&');
  const blocks = requests.map((r) =>
    Array.from({ length: r.n }, (_, j) => {
      const position = r.cached + j;
      const cells = requests.flatMap((other) => Array.from({ length: other.cached + other.n }, (_, p) => (other === r && p <= position ? '0' : '-\\infty')));
      return [`${r.id}_{${position + 1}}`, ...cells].join('&');
    }).join('\\\\'),
  );
  return `{\\scriptsize\\begin{array}{${columns}}${header}\\\\\\hline ${blocks.join('\\\\\\hline ')}\\end{array}}`;
}

function pmatrix(name: string, rows: number, cols: number): string {
  const lines = Array.from({ length: rows }, (_, r) => Array.from({ length: cols }, (_, c) => `{{${name}[${r}, ${c}]}}`).join('&'));
  return `\\begin{pmatrix}${lines.join('\\\\')}\\end{pmatrix}`;
}

export function packedAttention(rows: readonly PackedRow[], options: PackedOptions = {}): PackedPage {
  const d = options.d ?? DEFAULT_D;
  const random = mulberry32(options.seed ?? DEFAULT_SEED);
  const requests = packedRequests(rows);
  const N = requests.reduce((sum, r) => sum + r.n, 0);
  const total = requests.reduce((sum, r) => sum + r.cached + r.n, 0);
  const cachedRequests = requests.filter((r) => r.cached > 0);

  const g = graph({ dims: { d, N, Nall: total } });
  const weights = { q: uniformMatrix(random, d, d), k: uniformMatrix(random, d, d), v: uniformMatrix(random, d, d) };
  const xValues = uniformMatrix(random, N, d);
  const x = g.input('x', ['N', 'd'], xValues, { tex: '\\mathbf{x}', elem: 'x' });
  const Wq = g.param('Wq', ['d', 'd'], weights.q, { tex: 'W_{q}', elem: 'W_{q}' });
  const Wk = g.param('Wk', ['d', 'd'], weights.k, { tex: 'W_{k}', elem: 'W_{k}' });
  const Wv = g.param('Wv', ['d', 'd'], weights.v, { tex: 'W_{v}', elem: 'W_{v}' });

  const kCache = new Map<string, Node>();
  const vCache = new Map<string, Node>();
  for (const r of cachedRequests) {
    const earlier = uniformMatrix(random, r.cached, d);
    kCache.set(r.id, g.buffer(`K_cache_${r.id}`, [r.cached, 'd'], linear(earlier, weights.k), { tex: `K^{(${r.id})}_{\\mathrm{cache}}`, elem: `{K^{(${r.id})}_{\\mathrm{cache}}}` }));
    vCache.set(r.id, g.buffer(`V_cache_${r.id}`, [r.cached, 'd'], linear(earlier, weights.v), { tex: `V^{(${r.id})}_{\\mathrm{cache}}`, elem: `{V^{(${r.id})}_{\\mathrm{cache}}}` }));
  }

  const q = g.matmul(x, Wq.T, { name: 'q', tex: '\\mathbf{q}', elem: 'q' });
  const k = g.matmul(x, Wk.T, { name: 'k', tex: '\\mathbf{k}', elem: 'k' });
  const v = g.matmul(x, Wv.T, { name: 'v', tex: '\\mathbf{v}', elem: 'v' });
  const kNew = requests.map((r) => g.narrow(k, 0, r.start, r.n, { name: `k_${r.id}`, tex: `\\mathbf{k}^{(${r.id})}`, elem: `k^{(${r.id})}` }));
  const vNew = requests.map((r) => g.narrow(v, 0, r.start, r.n, { name: `v_${r.id}`, tex: `\\mathbf{v}^{(${r.id})}`, elem: `v^{(${r.id})}` }));
  const pieces = (cache: Map<string, Node>, fresh: Node[]) => requests.flatMap((r, i) => [...(cache.has(r.id) ? [cache.get(r.id)!] : []), fresh[i]]);
  const K = g.cat(pieces(kCache, kNew), 0, { name: 'K', tex: 'K', elem: 'K' });
  const V = g.cat(pieces(vCache, vNew), 0, { name: 'V', tex: 'V', elem: 'V' });
  const S = g.matmul(q, K.T, { name: 'S', tex: 'S', elem: 'S' });
  const Ss = g.mul(S, 1 / Math.sqrt(d), { name: 'Ss', tex: '\\tilde{S}', elem: '\\tilde{S}' });
  const M = g.buffer('M', ['N', 'Nall'], packedMask(requests), { tex: 'M', elem: 'M' });
  const Sm = g.add(Ss, M, { name: 'Sm', tex: '\\bar{S}', elem: '\\bar{S}' });
  const P = g.softmax(Sm, -1, { name: 'P', tex: 'P', elem: 'P' });
  const O = g.matmul(P, V, { name: 'O', tex: 'O', elem: 'O' });
  const index = g.label(
    'logits_idx',
    requests.map((r) => r.last),
    { classes: N, tex: '\\mathrm{idx}', elem: '\\mathrm{idx}' },
  );
  const logits = g.embedding(index, O, { name: 'O_last', tex: 'O_{\\mathrm{last}}', elem: 'O^{\\mathrm{last}}' });

  const R = requests.length;
  const first = requests[0];
  const tail = requests[requests.length - 1];
  const order = requests.map((r) => r.id).join('、');
  const hasDecode = requests.some((r) => r.cached > 0 && r.n === 1);
  const hasPrefill = requests.some((r) => r.cached === 0);
  const hasChunk = requests.some((r) => r.cached > 0 && r.n > 1);
  const cacheIds = cachedRequests.map((r) => r.id).join('、');
  const kAll = requests.map((r) => (r.cached > 0 ? `K^{(${r.id})}_{\\mathrm{cache}};\\ \\mathbf{k}^{(${r.id})}` : `\\mathbf{k}^{(${r.id})}`)).join(';\\ ');
  const lastRows = requests.map((r) => `${r.id} 的第 ${r.last + 1} 行`).join('、');

  const forward: ForwardStep[] = [
    {
      show: [x, Wq.T, Wk.T, Wv.T],
      label: '$\\mathbf{x}$',
      title: '打包的输入',
      text: String.raw`
        推理引擎每一步只做一次前向。这一步的 ${R} 个请求按 ${order} 的顺序沿 token 维拼成一块，共 $N=${N}$ 行，不补齐：

        <ul>${requests.map(requestLine).join('')}</ul>

        $\mathbf{x}$ 的每一行是一个新 token 在这一层的输入，$W_q$、$W_k$、$W_v$ 是所有请求共用的权重。`,
    },
    {
      show: [q, k, v],
      label: '$\\mathbf{q},\\mathbf{k},\\mathbf{v}$',
      title: '对整块算 q、k、v',
      text: String.raw`
        三个投影对 $N=${N}$ 行一次算完。矩阵乘的每一行只用到 $\mathbf{x}$ 的同一行，所以不同请求拼在一起不会互相影响，也不需要知道行属于哪个请求。

        $$\mathbf{q}=\mathbf{x}W_q^{\top},\quad \mathbf{k}=\mathbf{x}W_k^{\top},\quad \mathbf{v}=\mathbf{x}W_v^{\top}$$`,
    },
    {
      show: kNew,
      label: '切开 $\\mathbf{k}$',
      title: '按请求切开新的 k',
      text: String.raw`
        按每个请求在打包中的起止行，把新算出的 $\mathbf{k}$ 切成 ${R} 段：${requests.map((r) => `$\mathbf{k}^{(${r.id})}$ 有 ${r.n} 行`).join('，')}。图里每段放在右边一列。

        vLLM 里这一步对应把新 token 的 k、v 写进它们在 KV cache 里的槽位，槽位由每个请求的 block table 决定。`,
    },
  ];
  if (cachedRequests.length > 0) {
    forward.push({
      show: [...kCache.values()],
      label: '读 KV cache',
      title: '从 KV cache 读出之前的 k',
      text: String.raw`
        请求 ${cacheIds} 在之前的步骤里已经算过前面的 token，那些 token 的 k、v 存在 KV cache 里。这一步只读出它们，不再计算。图里它们放在左边一列，是常量叶子：没有梯度，也不连回 $\mathbf{x}$ 和 $W_k$。

        ${cachedRequests.map((r) => `$K^{(${r.id})}_{\mathrm{cache}}$ 有 ${r.cached} 行`).join('，')}。这些数值是用同一个 $W_k$ 从更早的输入算出的，和这一步新算的行没有区别，区别只在于这一步不用再算它们。`,
    });
  }
  forward.push(
    {
      show: K,
      label: '$K$',
      title: '按请求拼出 K',
      text: String.raw`
        每个请求先放缓存里的行，再放这一步新算的行，按打包顺序拼成 $K$，共 $N_{\mathrm{all}}=${total}$ 行：

        $$K=\begin{pmatrix}${kAll}\end{pmatrix}$$

        实际的 attention kernel 不做这次拼接，它按 block table 直接从 KV cache 的各个 block 读取。这里画成拼接，是为了把每个 query 能读到的 key 放在同一张表里。`,
    },
    {
      show: [K.T, S],
      label: '$\\mathbf{q}K^{\\top}$',
      title: '打分：q 乘 K 转置',
      text: String.raw`
        $S$ 的形状是 $N\times N_{\mathrm{all}}=${N}\times ${total}$，第 $r$ 行第 $c$ 列是第 $r$ 个 query 和第 $c$ 个 key 的点积。这张表把所有请求的 query 和所有请求的 key 都乘了一遍，其中跨请求的格子没有意义，下一步由 mask 去掉。`,
    },
    {
      show: Ss,
      label: '$\\div\\sqrt{d}$',
      title: '缩放',
      text: String.raw`乘上 $1/\sqrt{d}=${(1 / Math.sqrt(d)).toFixed(2)}$。这一页只有一个 head，head 的维度就是 $d=${d}$。`,
    },
    {
      show: [M, Sm],
      notes: [{ nodes: [M], index: [tail.start, tail.keyStart + tail.cached], text: `请求 ${tail.id} 的第一行 query 只在自己的块里是 $0$`, side: 'right' }],
      label: '$+M$',
      title: '加上分块的因果 mask',
      text: String.raw`
        $M$ 的第 $r$ 行第 $c$ 列在两个条件同时成立时是 $0$，否则是 $-\infty$：第 $c$ 列的 key 和第 $r$ 行的 query 属于同一个请求，并且这个 key 在请求里的位置不晚于这个 query 的位置。key 的位置从请求的第一个缓存 token 算起，所以缓存列总是在 query 之前。

        $$M=${maskTable(requests)}$$

        行标 $${first.id}_{${first.cached + 1}}$ 表示请求 ${first.id} 的第 ${first.cached + 1} 个 token 的 query，列标同理表示 key。实线分开不同的请求，虚线左边是 KV cache 里的列，右边是这一步新算的列。图里 $M$ 画在 $\tilde{S}$ 和 $\bar{S}$ 之间，$-\infty$ 的格子画成截断的蓝色柱子。

        第一个条件让 $M$ 成为分块对角的：每个请求一块，块与块之间全是 $-\infty$，一个请求的 query 读不到别的请求的 key。第二个条件让每一块是因果的，并且把缓存列包括进来：

        <ul>${requests.map(visibility).join('')}</ul>

        ${[
          hasDecode ? 'decode 请求只有一行 query，它的位置在所有缓存 token 之后，所以这一行看到全部缓存列和它自己，块是一整行 0。' : '',
          hasPrefill ? 'prefill 请求没有缓存列，它的块就是普通的下三角 causal mask。' : '',
          hasChunk ? '分块 prefill 的后续块先看到全部缓存列，再在新列上是下三角。' : '',
        ].join('')}`,
    },
    {
      show: P,
      notes: [{ nodes: [P], index: [tail.last, tail.keyStart], text: `请求 ${tail.id} 的最后一行：权重只落在它自己的 ${tail.cached + tail.n} 列上`, side: 'right' }],
      label: '$P$',
      title: 'softmax',
      text: String.raw`
        按行做 softmax。$e^{-\infty}=0$，所以每一行的权重只落在同一个请求、位置不晚于自己的 key 上，每行加起来是 1。结果和把每个请求单独算一次注意力完全相同。`,
    },
    {
      show: [...vNew, ...vCache.values(), V],
      label: '$V$',
      title: '同样拼出 V',
      text: String.raw`
        $\mathbf{v}$ 按同样的起止行切开，${cachedRequests.length > 0 ? `请求 ${cacheIds} 的缓存行从 KV cache 读出，` : ''}再按打包顺序拼成 $V$，行的顺序和 $K$ 相同。$V$ 画成下一步矩阵乘的后板，行从后往前排，所以每一段和它在 $V$ 里的行是前后反过来对齐的。`,
    },
    {
      show: O,
      label: '$PV$',
      title: '加权求和',
      text: String.raw`
        $O=PV$，每一行是这个 query 能看到的 value 的加权平均。$V$ 里被 mask 挡住的行权重为 0，不影响结果。`,
    },
    {
      show: logits,
      label: '$O_{\\mathrm{last}}$',
      title: '只取每个请求的最后一行',
      text: String.raw`
        下一个 token 只由每个请求最后一个位置的输出决定，所以引擎在 lm_head 之前按 <code>logits_indices</code> 取出这些行，即 ${lastRows}，共 ${R} 行。只有这 ${R} 行会乘 lm_head、算 logits 和采样。

        $$O_{\mathrm{last}}=O[\mathrm{idx}]=${pmatrix('O_last', R, d)}$$

        其余行的输出在完整的模型里继续送进后面的层，用来算这些位置在后面各层的 k、v，写进 KV cache，但不产生 logits。`,
    },
  );

  const at = (x: number, level: number, z: number): Vec3 => [x, level * LEVEL, z];
  const scaled = { scale: SCALE };
  const flow = { ...scaled, level: LEVEL / SCALE, back: 0 };
  const boardZ = -4;
  const row = BATCH_ROW_SPACING * SCALE;
  const front = boardZ + (BOARD_FRONT + (N - 1) * BATCH_ROW_SPACING) * SCALE;
  const vRow = (r: number) => boardZ + (BOARD_FRONT - BOARD_BACK_OFFSET - (total - 1 - r) * BATCH_ROW_SPACING) * SCALE;
  const keyPieces = (column: number, level: number, cache: Map<string, Node>, fresh: Node[], z: (r: PackedRequest, offset: number, n: number) => number) =>
    requests.flatMap((r, i) => [
      ...(cache.has(r.id) ? [stack(cache.get(r.id)!, { ...scaled, at: at(column - PIECE_SHIFT, level, z(r, 0, r.cached)) })] : []),
      stack(fresh[i], { ...scaled, at: at(column + PIECE_SHIFT, level, z(r, r.cached, r.n)) }),
    ]);
  const layout: LayoutItem[] = [
    board(x, Wq.T, q, { ...scaled, at: at(Q_X, 0, boardZ) }),
    board(x, Wk.T, k, { ...scaled, at: at(K_X, 0, boardZ) }),
    board(x, Wv.T, v, { ...scaled, at: at(V_X, 0, boardZ) }),
    ...keyPieces(K_X, 1.5, kCache, kNew, (r, offset) => front - (r.keyStart + offset) * row),
    stack(K, { ...scaled, at: at(K_X, 3, front) }),
    board(q, K.T, S, { ...scaled, at: at(Q_X, 4, boardZ) }),
    stack(S, Ss, M, Sm, P, flow),
    ...keyPieces(V_X, 7.5, vCache, vNew, (r, offset, n) => vRow(r.keyStart + offset + n - 1)),
    board(P, V, O, { ...scaled, at: at(V_X, 9, boardZ), gap: BATCH_ROW_SPACING }),
    stack(O, logits, { ...flow, at: [0, 0, front], offsets: { [logits.name]: [LAST_SHIFT, 0, 0] } }),
  ];

  const steps: StepsSpec = {
    overview: {
      label: '总览',
      title: '一次打包前向',
      text: String.raw`
        这一页画推理引擎（vLLM、SGLang）的一步：${R} 个请求的新 token 沿 token 维拼成一块，做一次单 head 注意力的前向。打包时不补齐，每个请求的 query 只读自己的 key 和 value，包括 KV cache 里已有的和这一步新算的。

        推理只有前向，没有损失，也不算梯度，所以这一页没有反向步骤。模型维度 $d=${d}$，新 token 共 $N=${N}$ 个，连同缓存共 $N_{\mathrm{all}}=${total}$ 个 key。`,
    },
    forward,
    backward: 'auto',
  };

  const requestTable = `<table class="vf-packed"><tr><th>请求</th><th>缓存</th><th>新增</th><th>打包后的行</th><th>类型</th></tr>${requests
    .map((r) => `<tr><td>${r.id}</td><td>${r.cached}</td><td>${r.n}</td><td>${r.start + 1 === r.last + 1 ? r.start + 1 : `${r.start + 1} 到 ${r.last + 1}`}</td><td>${kindOf(r)}</td></tr>`)
    .join('')}</table>`;
  const panels: PanelItem[] = [
    { note: requestTable, title: '这一步的请求' },
    {
      code: `# 一次前向，N 个新 token 打包，不补齐
q, k, v = x @ Wq.T, x @ Wk.T, x @ Wv.T
cache.write(k, v, slot_mapping)
o = varlen_attn(q, cache, query_start_loc,
                seq_lens, block_table)
h = o[logits_indices]  # 每个请求最后一行
logits = lm_head(h)`,
      title: '对应的引擎代码（简化）',
      note: '<code>cache.write</code> 把新 k、v 写进各请求的槽位；<code>varlen_attn</code> 按 <code>query_start_loc</code> 找到每个请求的 query 行，按 block table 读它的 k、v，效果等于这一页的分块 mask。',
    },
  ];

  return {
    graph: g,
    steps,
    layout,
    panels,
    requests,
    nodes: { x, q, k, v, K, V, S, M, P, O, logits, kCache, vCache },
  };
}
