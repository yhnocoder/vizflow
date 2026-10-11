const vec = (name, n, digits = 2) => `\\begin{pmatrix}${[...Array(n).keys()].map((i) => `{{${name}[0, ${i}]:${digits}}}`).join('&')}\\end{pmatrix}`;

export function stepTexts(kind, t, info, c) {
  const { ratio, emits, win, widx, K } = info;
  const slot = t % win;
  const m = ratio;
  const holds = widx.map((s) => (s < 0 ? null : t - ((slot - s + win) % win)));
  const csa = ratio === 4;
  const nEntries = ratio ? Math.floor((t + 1) / ratio) : 0;
  const entryRow = ratio ? win + Math.floor(t / ratio) : 0;
  const realM = csa ? 4 : 128;
  const S = {};

  S.inputs = {
    label: '输入与 cache',
    title: `位置 t = ${t} 的输入与这一层的 cache`,
    text: String.raw`$\mathbf{x}_t$ 是这一步新 token 在这一层的输入，形状 $(1, ${c.dim})$。$\mathcal{C}$ 是这一层唯一的主 cache，官方代码在 <code>Attention.__init__</code> 里一次分配好：$(\text{max\_batch}, \text{window\_size} + \text{max\_seq\_len}/m, \text{head\_dim})$，这里是 $(${info.nCache}, ${c.head_dim})$。前 ${win} 行是滑动窗口的环形缓冲区，位置 $p$ 写在第 $p \bmod ${win}$ 行${ratio ? String.raw`；后 ${info.nCache - win} 行存压缩条目，第 $i$ 个条目在第 $${win}+i$ 行` : '；这一层不压缩，cache 只有窗口'}。

K 与 V 是同一个张量（shared K=V，单个 kv head），所以 cache 只存一份。真实的 V4-Flash 是 window 128、head\_dim 512${ratio ? String.raw`、$m=${realM}$` : ''}。toy 尺寸见页面上方的说明。`,
  };
  S.qa = {
    label: String.raw`$\mathbf{q}^{a}$`,
    title: 'query 的低秩投影与 q_norm',
    text: String.raw`<code>qr = q = self.q_norm(self.wq_a(x))</code>。$\mathbf{q}^{a}=\mathbf{x}_tW^{qa\top}$ 降到秩 ${c.q_lora_rank}（真实 1024），再做带权重的 RMSNorm。归一化后的 $\overline{\mathbf{q}^{a}}$ 还会交给 indexer 当它的 query 输入${csa ? '' : '（这一层没有 indexer）'}。`,
  };
  S.qb = {
    label: String.raw`$\mathbf{q}^{b}$`,
    title: '升维并分成 head',
    text: String.raw`<code>q = self.wq_b(q).unflatten(-1, (n_heads, head_dim))</code>，得到 ${c.n_heads} 个 head（真实 64 个），每个 ${c.head_dim} 维。`,
  };
  S.qnorm = {
    label: '逐 head 归一化',
    title: '每个 head 再做一次不带权重的 RMSNorm',
    text: String.raw`<code>q *= torch.rsqrt(q.square().mean(-1, keepdim=True) + self.eps)</code>。这一步没有可学的权重，每个 head 单独归一化。`,
  };
  S.qrope = {
    label: 'RoPE(q)',
    title: 'query 的最后几维加 RoPE',
    text: String.raw`只有最后 rope\_head\_dim 维旋转（真实 64，这里 ${c.rope_head_dim}），其余维度不动。官方代码用 <code>view_as_complex</code> 把相邻两维当一个复数相乘，这里只有一对，相邻配对与前后两半配对的写法结果相同，真实尺寸下两者不同，这一页按官方的相邻配对理解。${kind === 'swa' ? '滑动窗口层用 $\\theta=10000$、不加 YaRN。' : 'CSA 与 HCA 层用 compress\\_rope\\_theta $=160000$ 并加 YaRN（factor 16），与它们的压缩器共用同一张表。'}`,
  };
  S.kv = {
    label: String.raw`$\mathbf{kv}$`,
    title: '新 token 的 K=V',
    text: String.raw`<code>kv = self.kv_norm(self.wkv(x))</code>。一个投影同时给出 key 和 value，只有一个 kv head，所有 query head 共用。$$\overline{\mathbf{kv}}=${vec('norm(kv)', c.head_dim)}$$`,
  };
  S.kvrope = {
    label: 'RoPE(kv)',
    title: 'K=V 的最后几维加 RoPE',
    text: String.raw`和 query 一样只旋转最后几维。因为 V 就是 K，V 的这几维也带上了位置旋转，最后输出时要反向转回来（倒数第二步）。

真实代码在这里还把前 448 维（nope 部分）做一次 FP8 量化再反量化（<code>act_quant(kv[..., :-rd], 64, ...)</code>，每 64 维一个 ue8m0 缩放），这样 cache 里存的就是 FP8 能表示的值；rope 的 64 维保持 BF16。这一页全程用 float64，没有模拟这次舍入。`,
  };
  S.widx = {
    label: String.raw`$I^{\mathrm{win}}$`,
    title: '窗口要读哪几行',
    text: String.raw`<code>get_window_topk_idxs(win, bsz, seqlen, start_pos)</code>：$t=${t}\ge \text{win}-1$ 时，按从旧到新的顺序列出环形缓冲区的 ${win} 个行号：$(${widx.join(', ')})$，分别存着位置 $(${holds.join(', ')})$。行号只取决于 $t$，与 cache 内容无关。后面「写入窗口」一步会在 attention 之前把新 token 写进第 ${slot} 行，所以位置 ${t} 也在窗口里，所以窗口一共看 ${win} 个位置，包括自己。

$t<\text{win}-1$ 时还没有绕回来，空位用 $-1$ 填，kernel 遇到 $-1$ 时取全零行并把分数设为 $-\infty$。这一页从 $t=${c.prefill ?? 5}$ 开始，用不到 $-1$。`,
  };
  const comp = (who) => {
    const isI = who === 'I';
    const d = isI ? c.index_head_dim : c.head_dim;
    const row = (csa ? m : 0) + (t % m);
    return {
      proj: {
        label: isI ? 'indexer 压缩器投影' : '压缩器投影',
        title: isI ? 'indexer 的压缩器：投影与位置偏置' : '主压缩器：投影与位置偏置',
        text: String.raw`<code>kv = self.wkv(x); score = self.wgate(x); score += self.ape[start_pos % ratio]</code>。每个 token 给出要被压缩的内容 $\mathbf{c}$ 和打分 $\mathbf{z}$，打分加上它在窗口内位置（$t \bmod m = ${t % m}$）的可学偏置 $B$。${csa ? String.raw`压缩比 4 的层用重叠窗口，所以宽度是 $2d$：前一半 $a$ 留给下一个窗口，后一半 $b$ 给当前窗口。` : '压缩比 128 的层不重叠，宽度就是 $d$。'}压缩器在 float32 下计算。`,
      },
      state: {
        label: '写入压缩器状态',
        title: isI ? 'indexer 压缩器的状态缓冲' : '压缩器的状态缓冲',
        text: String.raw`还没凑满一个窗口的 token 不进 cache，先存在压缩器自己的状态缓冲 <code>kv_state</code>、<code>score_state</code> 里，形状 $(${(csa ? 2 : 1) * m}, ${(csa ? 2 : 1) * d})$。本步写在第 ${row} 行${csa ? String.raw`（$m + t \bmod m$）。前 $m$ 行是上一个窗口留下的，后 $m$ 行是当前窗口` : String.raw`（$t \bmod m$）`}。还没写过的行在初始化时是 <code>score_state = -inf</code>，参与 softmax 时权重为 0，但只要窗口凑满，这些行都已经被覆盖。

${emits ? `$(t+1) \\bmod m = 0$，这个窗口凑满了，下面几步压缩出一个条目。` : `$(t+1) \\bmod m = ${(t + 1) % m} \\ne 0$，窗口还没满，这一步${isI ? 'indexer 的压缩器' : '压缩器'}到此为止，cache 的压缩区不变。`}${isI ? '' : String.raw` 真实 vLLM 把这份状态也放在分页显存里，按滑动窗口管理（压缩比 4 留 8 个 token，128 留 128 个）。`}`,
      },
      softmax: {
        label: '窗口内 softmax',
        title: '对窗口里的 token 做门控 softmax',
        text: csa
          ? String.raw`<code>torch.cat([kv_state[:, :ratio, :d], kv_state[:, ratio:, d:]], dim=1)</code>：取上一个窗口 ${m} 个 token 的 $a$ 半和当前窗口 ${m} 个 token 的 $b$ 半，拼成 $${2 * m}$ 行，相邻两个压缩条目各覆盖 8 个 token、重叠 4 个。打分在这 ${2 * m} 行上沿 token 方向做 softmax，每一维单独归一化。`
          : String.raw`<code>self.score_state.softmax(dim=1)</code>：当前窗口的 ${m} 个 token 的打分沿 token 方向做 softmax，每一维单独归一化。`,
      },
      pool: {
        label: '加权求和与归一化',
        title: '压成一个条目',
        text: String.raw`$$\mathbf{e}=\sum_{j} \mathrm{softmax}_j(\mathbf{z}_j + B_j)\odot \mathbf{c}_j$$ 然后做一次带权重的 RMSNorm（<code>self.norm</code>）。`,
      },
      rope: {
        label: '条目的 RoPE',
        title: '条目按窗口起点的位置旋转',
        text: String.raw`<code>freqs_cis[start_pos + 1 - ratio]</code>：条目用窗口第一个 token 的位置 $${t + 1 - m}$ 加 RoPE。${isI ? String.raw`indexer 的条目随后乘 Hadamard 矩阵 $H$（<code>rotate_activation</code>），再做 FP4 量化。$H$ 是正交矩阵，query 也乘同一个 $H$，所以点积不变；它的作用是把离群值摊开，减小 FP4 的误差。这一页没有模拟 FP4 舍入。` : String.raw`真实代码随后对前 448 维做 FP8 量化（与窗口里的 K=V 一样），这一页没有模拟。`}`,
      },
      write: {
        label: isI ? '写入 indexer cache' : '写入压缩区',
        title: isI ? '写入 indexer 的 key cache' : '写入主 cache 的压缩区',
        text: String.raw`<code>self.kv_cache[:bsz, start_pos // ratio] = kv</code>：第 $\lfloor ${t}/${m}\rfloor = ${Math.floor(t / m)}$ 个条目${isI ? '写进 indexer 自己的 cache' : String.raw`写进主 cache 的第 $${win}+${Math.floor(t / m)}=${entryRow}$ 行`}。之后状态缓冲${csa ? '把后一半整体搬到前一半（\<code>kv\_state[:, :ratio] = kv\_state[:, ratio:]</code>），作为下一个窗口的 $a$ 半' : '不需要搬动，下一个窗口直接覆盖'}。这个条目从位置 $${t + 1 - m}$ 到 $${t}$ 的 token 压出来，只有到 $t=${t}$ 这一步才出现，所以 $t<${t}$ 的 query 看不到它。`,
      },
    };
  };
  if (ratio) {
    const main = comp('C');
    for (const k of Object.keys(main)) S[`c.compressor.${k}`] = main[k];
  }
  if (csa) {
    const ind = comp('I');
    for (const k of Object.keys(ind)) S[`i.indexer.compressor.${k}`] = ind[k];
    S.iq = {
      label: 'indexer query',
      title: 'Lightning Indexer 的 query',
      text: String.raw`indexer 决定这个 query 从压缩条目里读哪 $k$ 个（真实 $k=512$，这里 ${c.index_topk}）。它的 query 从 $\overline{\mathbf{q}^{a}}$ 投影出 ${c.index_n_heads} 个 head（真实 64），每个 ${c.index_head_dim} 维（真实 128），加 RoPE，再乘 Hadamard 矩阵并做 FP4 量化（未模拟）。`,
    };
    S.icache = {
      label: 'indexer cache',
      title: 'indexer 有自己的一份压缩 key cache',
      text: String.raw`<code>Indexer.__init__</code> 分配 $(\text{max\_seq\_len}/4, \text{index\_head\_dim})$ 的 cache。它和主 cache 用同样的窗口切分，但只存 ${c.index_head_dim} 维的 key，是另一套权重压出来的。每个 CSA 层都有一份，这是 V4 cache 里除主 KV 之外的第二种全局 cache。`,
    };
    S.iscore = {
      label: 'indexer 打分',
      title: '给每个已有条目打分',
      text: String.raw`$$s_j=\sum_{h} w_h\,\mathrm{ReLU}\big(\mathbf{q}^{I}_h H\cdot \mathbf{k}^{I}_j\big)$$ 只看 $\lfloor (t+1)/4 \rfloor = ${nEntries}$ 个已经生成的条目（<code>self.kv_cache[:bsz, :end_pos // ratio]</code>）。$w_h$ 由 $\mathbf{x}_t$ 投影得到，乘 $1/\sqrt{d_I H_I}$。decode 时不需要因果掩码，因为未来的条目还不存在。`,
    };
    S.itopk = {
      label: 'top-k',
      title: `选出 ${Math.min(c.index_topk, nEntries)} 个条目`,
      text: String.raw`<code>index_score.topk(min(self.index_topk, end_pos // ratio))</code>，再加上偏移 win $=${win}$，变成主 cache 里的行号：$$I^{\mathrm{cmp}}=${vec('compressed idxs', Math.min(c.index_topk, nEntries), 0)}$$ 被选中的顺序按分数从高到低，不按位置排序。`,
    };
  }
  if (ratio && !csa) {
    S.hidx = {
      label: String.raw`$I^{\mathrm{cmp}}$`,
      title: 'HCA 读全部压缩条目',
      text: String.raw`<code>get_compress_topk_idxs(ratio, ..., offset)</code>：HCA 层没有 indexer，读所有已经生成的 $\lfloor (t+1)/m \rfloor = ${nEntries}$ 个条目，行号是 $${win}$ 起的连续 ${nEntries} 个。真实压缩比 128，1M 上下文也只有 8192 个条目，所以全读。${nEntries === 0 ? '' : ''}`,
    };
  }
  S.idx = {
    label: '$I$',
    title: '拼出这一步要读的全部行号',
    text: String.raw`窗口的 ${win} 个行号接上压缩区的行号，一共 ${K} 个。注意本步若刚好凑满窗口，新条目还没写进 cache，但它的行号已经可以出现在这里（HCA 必然包含，CSA 由 indexer 决定），因为真正读取发生在写入之后。`,
  };
  S.wwrite = {
    label: '写入窗口',
    title: `把新 token 写进环形缓冲区第 ${slot} 行`,
    text: String.raw`<code>self.kv_cache[:bsz, start_pos % win] = kv.squeeze(1)</code>。第 ${slot} 行原来存的是位置 $${t - win}$，它已经滑出窗口，被直接覆盖。窗口部分的显存与上下文长度无关，每层固定 ${win} 行（真实 128 行）。`,
  };
  S.gather = {
    label: String.raw`$\mathcal{C}[I]$`,
    title: '按行号从 cache 取出 K=V',
    text: String.raw`<code>sparse_attn(q, self.kv_cache[:bsz], self.attn_sink, topk_idxs, self.softmax_scale)</code>：kernel 按 $I$ 从整个 cache（窗口加压缩区）取出 ${K} 行，同一份既当 K 也当 V。连线从 cache 的行连到取出的行，悬停可以看每一行来自哪个槽。`,
  };
  S.scores = {
    label: String.raw`$Q\mathcal{C}[I]^{\top}$`,
    title: '打分',
    text: String.raw`每个 head 的 query 与取出的 ${K} 行做点积，乘 $1/\sqrt{\text{head\_dim}}$。`,
  };
  S.softmax = {
    label: 'softmax 加 sink',
    title: '带 attention sink 的 softmax',
    text: String.raw`kernel 在分母上加一项 $e^{b^{\mathrm{sink}}_h}$（<code>sum_exp += exp(attn_sink - scores_max)</code>），sink 不乘缩放，也没有对应的 value：$$P_{hj}=\frac{e^{s_{hj}}}{\sum_{j'} e^{s_{hj'}}+e^{b^{\mathrm{sink}}_h}}$$ 这里写成在分数后面接一列 sink、做 softmax、再去掉这一列，结果相同。每行的权重加起来小于 1。`,
  };
  S.o = { label: '$PV$', title: '加权求和', text: String.raw`$O = P\,\mathcal{C}[I]$，value 就是取出的同一批行。` };
  S.orope = {
    label: '反向 RoPE',
    title: '把输出的 rope 维转回去',
    text: String.raw`<code>apply_rotary_emb(o[..., -rd:], freqs_cis, True)</code>：用共轭（角度取负）在 query 的位置 $t$ 上旋转。因为 V=K 带着各自位置 $s$ 的旋转，转回 $-t$ 之后，每一项贡献只依赖相对距离 $s-t$。`,
  };
  S.oa = {
    label: String.raw`$\mathbf{u}$`,
    title: '分组低秩输出投影',
    text: String.raw`<code>einsum("bsgd,grd->bsgr", o, wo_a)</code>：head 分成 ${c.o_groups} 组（真实 8 组），每组单独投影到 ${c.o_lora_rank} 维（真实 1024）。`,
  };
  S.ob = {
    label: String.raw`$\mathbf{y}$`,
    title: '输出',
    text: String.raw`<code>self.wo_b(o.flatten(2))</code>。这个结果已与官方 <code>inference/model.py</code> 在同一组权重下逐值对照（误差小于 $2\times 10^{-5}$，来自压缩器的 float32）。$$\mathbf{y}=${vec('y', c.dim)}$$`,
  };
  return S;
}
