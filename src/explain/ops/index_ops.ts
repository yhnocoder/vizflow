import type { OpCtx, OpExplain } from '../../autograd';
import { type PadAttrs, type SelectAttrs, type SliceScatterAttrs, selectScatterTarget, sliceScatterSource, type PermuteAttrs, type RepeatKvAttrs, type ScatterAttrs, type SortAttrs, convTaps, cummaxSources, indexAddTarget, indexPutRows, indexSource, padSource, permuteSource, repeatKvAsInterleave, repeatSource, repeatSourceIndex, scatterPairs, sortSources } from '../../autograd/ops/index_ops';
import { broadcastIndex, formatNumber, indexOf, normDim, offsetOf } from '../../tensor';

export const gatherExplain: OpExplain = {
  pattern: 'select',
  local: () => 1,
  tex: {
    forward: (s) => `\\operatorname{gather}(${s.in[0]},${s.in[1]})`,
    local: () => '1',
  },
  selected: (ctx, out) => {
    const dim = normDim((ctx.attrs as { dim: number }).dim, out.length);
    const index = [...out];
    index[dim] = Math.round(ctx.inputs[1].get(out));
    return [{ arg: 0, index }];
  },
};

export const embeddingExplain: OpExplain = {
  pattern: 'elementwise',
  local: () => 1,
  tex: {
    forward: (s) => `${s.in[1]}[${s.in[0]}]`,
    local: () => '1',
  },
  selected: (ctx, out) => {
    const row = Math.round(ctx.inputs[0].get(out.slice(0, -1)));
    return [{ arg: 1, index: [row, out[out.length - 1]] }];
  },
};

export const narrowExplain: OpExplain = {
  pattern: 'elementwise',
  local: () => 1,
  tex: {
    forward: (s) => {
      const { dim, start, length, step = 1 } = s.attrs as { dim: number; start: number; length: number; step?: number };
      return `\\operatorname{narrow}(${s.in[0]},${dim},${start},${length}${step === 1 ? '' : `,${step}`})`;
    },
    local: () => '1',
  },
  selected: (ctx, out) => {
    const attrs = ctx.attrs as { dim: number; start: number; step?: number };
    const dim = normDim(attrs.dim, out.length);
    const index = [...out];
    index[dim] = (attrs.start < 0 ? attrs.start + ctx.inputs[0].shape[dim] : attrs.start) + index[dim] * (attrs.step ?? 1);
    return [{ arg: 0, index }];
  },
};

export const catExplain: OpExplain = {
  pattern: 'elementwise',
  local: () => 1,
  tex: {
    forward: (s) => `\\operatorname{cat}(${s.in.join(',')})`,
    local: () => '1',
  },
  selected: (ctx, out) => {
    const dim = normDim((ctx.attrs as { dim: number }).dim, out.length);
    const index = [...out];
    let arg = 0;
    while (index[dim] >= ctx.inputs[arg].shape[dim]) {
      index[dim] -= ctx.inputs[arg].shape[dim];
      arg++;
    }
    return [{ arg, index }];
  },
};

export const viewExplain: OpExplain = {
  pattern: 'elementwise',
  local: () => 1,
  tex: {
    forward: (s) => `\\operatorname{view}(${s.in[0]})`,
    local: () => '1',
  },
  selected: (ctx, out) => [{ arg: 0, index: indexOf(ctx.inputs[0].shape, offsetOf(ctx.out.shape, out)) }],
};

export const selectExplain: OpExplain = {
  pattern: 'elementwise',
  local: () => 1,
  tex: {
    forward: (s) => {
      const { dim, index } = s.attrs as { dim: number; index: number };
      return `\\operatorname{select}(${s.in[0]},${dim},${index})`;
    },
    local: () => '1',
  },
  selected: (ctx, out) => {
    const attrs = ctx.attrs as { dim: number; index: number };
    const dim = normDim(attrs.dim, ctx.inputs[0].shape.length);
    const index = attrs.index < 0 ? attrs.index + ctx.inputs[0].shape[dim] : attrs.index;
    return [{ arg: 0, index: [...out.slice(0, dim), index, ...out.slice(dim)] }];
  },
};

export const repeatInterleaveExplain: OpExplain = {
  pattern: 'elementwise',
  local: () => 1,
  tex: {
    forward: (s) => {
      const { repeats, dim } = s.attrs as { repeats: number; dim: number };
      return `\\operatorname{repeat\\_interleave}(${s.in[0]},${repeats},${dim})`;
    },
    local: () => '1',
  },
  selected: (ctx, out) => {
    const attrs = ctx.attrs as { repeats: number; dim: number };
    const dim = normDim(attrs.dim, out.length);
    const index = [...out];
    index[dim] = Math.floor(out[dim] / attrs.repeats);
    return [{ arg: 0, index }];
  },
};

export const repeatKvExplain: OpExplain = {
  pattern: 'elementwise',
  local: () => 1,
  tex: {
    forward: (s) => `\\operatorname{repeat\\_kv}(${s.in[0]},${(s.attrs as RepeatKvAttrs).n_rep})`,
    local: () => '1',
  },
  selected: (ctx, out) => [{ arg: 0, index: repeatSource(ctx.inputs[0].shape, repeatKvAsInterleave(ctx.inputs[0].shape, ctx.attrs as RepeatKvAttrs), out) }],
};

export const contiguousExplain: OpExplain = {
  pattern: 'elementwise',
  local: () => 1,
  tex: {
    forward: (s) => `\\operatorname{contiguous}(${s.in[0]})`,
    local: () => '1',
  },
  selected: (_ctx, out) => [{ arg: 0, index: [...out] }],
};

const dilationOf = (ctx: OpCtx) => (ctx.attrs as { dilation: number }).dilation;

function convTap(ctx: OpCtx, out: number[], arg: number, input: number[]): { s: number; j: number } {
  const K = ctx.inputs[1].shape[1];
  const d = dilationOf(ctx);
  const t = out[0];
  return arg === 0 ? { s: input[0], j: K - 1 - (t - input[0]) / d } : { s: t - (K - 1 - input[1]) * d, j: input[1] };
}

export const causalConv1dExplain: OpExplain = {
  pattern: 'elementwise',
  local: (ctx, out, arg, input) => {
    const { s, j } = convTap(ctx, out, arg, input);
    const c = out[1];
    return arg === 0 ? ctx.inputs[1].get([c, j]) : ctx.inputs[0].get([s, c]);
  },
  tex: {
    forward: (s) => `\\operatorname{conv1d}(${s.in[0]},${s.in[1]})`,
    local: (s) => {
      const { s: row, j } = convTap(s.ctx, s.outIndex, s.arg, s.inIndex);
      const c = s.outIndex[1];
      return s.arg === 0 ? s.elem(1, [c, j]) : s.elem(0, [row, c]);
    },
  },
  selected: (ctx, out) => {
    const [T] = ctx.inputs[0].shape;
    const K = ctx.inputs[1].shape[1];
    const c = out[1];
    return convTaps(T, K, out[0], dilationOf(ctx)).flatMap(({ s, j }) => [
      { arg: 0, index: [s, c] },
      { arg: 1, index: [c, j] },
    ]);
  },
};

export const topkMaskExplain: OpExplain = {
  pattern: 'dense',
  local: () => 0,
  tex: {
    forward: (s) => `\\operatorname{topk\\_mask}(${s.in[0]},${(s.attrs as { k: number }).k})`,
    local: () => '0',
  },
};

export const topkIndicesExplain: OpExplain = {
  pattern: 'dense',
  local: () => 0,
  tex: {
    forward: (s) => `\\operatorname{topk\\_indices}(${s.in[0]},${(s.attrs as { k: number }).k})`,
    local: () => '0',
  },
};

export const oneHotExplain: OpExplain = {
  pattern: 'select',
  local: () => 0,
  tex: {
    forward: (s) => `\\operatorname{one\\_hot}(${s.in[0]},${(s.attrs as { num_classes: number }).num_classes})`,
    local: () => '0',
  },
  selected: () => [],
};

export const permuteExplain: OpExplain = {
  pattern: 'elementwise',
  local: () => 1,
  tex: {
    forward: (s) => `\\operatorname{permute}(${s.in[0]},${(s.attrs as { dims: number[] }).dims.join(',')})`,
    local: () => '1',
  },
  selected: (ctx, out) => [{ arg: 0, index: permuteSource(ctx.inputs[0].shape, ctx.attrs as PermuteAttrs, out) }],
};

export const nonzeroStaticExplain: OpExplain = {
  pattern: 'select',
  local: () => 0,
  tex: {
    forward: (s) => `\\operatorname{nonzero\\_static}(${s.in[0]},${(s.attrs as { size: number }).size})`,
    local: () => '0',
  },
  selected: () => [],
};

export const indexExplain: OpExplain = {
  pattern: 'select',
  local: () => 1,
  tex: {
    forward: (s) => `${s.in[0]}[${s.in.slice(1).join(',')}]`,
    local: () => '1',
  },
  selected: (ctx, out) => {
    const [input, ...indices] = ctx.inputs;
    const lead = indices[0].ndim;
    const row = offsetOf(indices[0].shape, out.slice(0, lead));
    return [{ arg: 0, index: [...indexSource(input.shape, indices, row), ...out.slice(lead)] }];
  },
};

export const indexAddExplain: OpExplain = {
  pattern: 'select',
  local: () => 1,
  tex: {
    forward: (s) => `\\operatorname{index\\_add}(${s.in[0]},${(s.attrs as { dim: number }).dim},${s.in[1]},${s.in[2]})`,
    local: () => '1',
  },
  selected: (ctx, out) => {
    const [input, index] = ctx.inputs;
    const dim = normDim((ctx.attrs as { dim: number }).dim, input.ndim);
    const picks = [{ arg: 0, index: [...out] }];
    for (let i = 0; i < index.size; i++) {
      if (indexAddTarget(input.shape, index, dim, i) !== out[dim]) continue;
      const source = [...out];
      source[dim] = i;
      picks.push({ arg: 2, index: source });
    }
    return picks;
  },
};

export const unsqueezeExplain: OpExplain = {
  ...viewExplain,
  tex: {
    forward: (s) => `\\operatorname{unsqueeze}(${s.in[0]},${(s.attrs as { dim: number }).dim})`,
    local: () => '1',
  },
};

export const squeezeExplain: OpExplain = {
  ...viewExplain,
  tex: {
    forward: (s) => `\\operatorname{squeeze}(${s.in[0]},${(s.attrs as { dim: number }).dim})`,
    local: () => '1',
  },
};

export const padExplain: OpExplain = {
  pattern: 'elementwise',
  local: () => 1,
  tex: {
    forward: (s) => `\\operatorname{pad}(${s.in[0]})`,
    local: () => '1',
  },
  selected: (ctx, out) => {
    const source = padSource(ctx.inputs[0].shape, ctx.attrs as PadAttrs, out);
    return source === undefined ? [] : [{ arg: 0, index: source }];
  },
};

export const repeatExplain: OpExplain = {
  pattern: 'elementwise',
  local: () => 1,
  tex: {
    forward: (s) => `\\operatorname{repeat}(${s.in[0]},${(s.attrs as { sizes: number[] }).sizes.join(',')})`,
    local: () => '1',
  },
  selected: (ctx, out) => [{ arg: 0, index: repeatSourceIndex(ctx.inputs[0].shape, out) }],
};

function cummaxSource(ctx: OpCtx, out: number[]): number[] {
  const input = ctx.inputs[0];
  return indexOf(input.shape, cummaxSources(input, ctx.attrs as { dim: number })[offsetOf(input.shape, out)]);
}

export const cummaxExplain: OpExplain = {
  pattern: 'elementwise',
  local: (ctx, out, _arg, input) => (offsetOf(ctx.inputs[0].shape, cummaxSource(ctx, out)) === offsetOf(ctx.inputs[0].shape, input) ? 1 : 0),
  tex: {
    forward: (s) => `\\operatorname{cummax}(${s.in[0]})`,
    local: () => '1',
  },
  selected: (ctx, out) => [{ arg: 0, index: cummaxSource(ctx, out) }],
};

function sortSource(ctx: OpCtx, out: number[]): number[] {
  const input = ctx.inputs[0];
  return indexOf(input.shape, sortSources(input, ctx.attrs as SortAttrs)[offsetOf(input.shape, out)]);
}

export const sortExplain: OpExplain = {
  pattern: 'elementwise',
  local: (ctx, out, _arg, input) => (offsetOf(ctx.inputs[0].shape, sortSource(ctx, out)) === offsetOf(ctx.inputs[0].shape, input) ? 1 : 0),
  tex: {
    forward: (s) => `\\operatorname{sort}(${s.in[0]})`,
    local: () => '1',
  },
  selected: (ctx, out) => [{ arg: 0, index: sortSource(ctx, out) }],
};

function scatterSource(ctx: OpCtx, out: number[]): { arg: number; index: number[] } | undefined {
  const [input, index, src] = ctx.inputs;
  const target = offsetOf(input.shape, out);
  const hit = scatterPairs(input, index, src, ctx.attrs as ScatterAttrs).filter(([to]) => to === target).pop();
  if (hit === undefined) return { arg: 0, index: [...out] };
  return src === undefined ? undefined : { arg: 2, index: indexOf(src.shape, hit[1]) };
}

export const scatterExplain: OpExplain = {
  pattern: 'elementwise',
  local: (ctx, out, arg) => (scatterSource(ctx, out)?.arg === arg ? 1 : 0),
  tex: {
    forward: (s) => {
      const { dim, src } = s.attrs as ScatterAttrs;
      return `\\operatorname{scatter}(${s.in[0]},${dim},${s.in[1]},${src === undefined ? s.in[2] : formatNumber(src, { tex: true })})`;
    },
    local: (s) => (scatterSource(s.ctx, s.outIndex)?.arg === s.arg ? '1' : '0'),
  },
  selected: (ctx, out) => {
    const source = scatterSource(ctx, out);
    return source === undefined ? [] : [source];
  },
};

function indexPutSource(ctx: OpCtx, out: number[]): { arg: number; index: number[] } {
  const [, mask, values] = ctx.inputs;
  const lead = mask.ndim;
  const row = offsetOf(mask.shape, out.slice(0, lead));
  const k = indexPutRows(mask).indexOf(row);
  if (k < 0) return { arg: 0, index: [...out] };
  return { arg: 2, index: broadcastIndex([k, ...out.slice(lead)], values.shape) };
}

export const indexPutExplain: OpExplain = {
  pattern: 'select',
  local: (ctx, out, arg) => (indexPutSource(ctx, out).arg === arg ? 1 : 0),
  tex: {
    forward: (s) => `${s.in[0]}[${s.in[1]}]\\leftarrow ${s.in[2]}`,
    local: () => '1',
  },
  selected: (ctx, out) => [indexPutSource(ctx, out)],
};

export const expandExplain: OpExplain = {
  ...repeatExplain,
  tex: {
    forward: (s) => `\\operatorname{expand}(${s.in[0]},${(s.attrs as { sizes: number[] }).sizes.join(',')})`,
    local: () => '1',
  },
};

export const reshapeExplain: OpExplain = {
  ...viewExplain,
  tex: {
    forward: (s) => `\\operatorname{reshape}(${s.in[0]})`,
    local: () => '1',
  },
};

function selectScatterSource(ctx: OpCtx, out: number[]): { arg: number; index: number[] } {
  const [input, src] = ctx.inputs;
  const { dim, index } = selectScatterTarget(input.shape, src.shape, ctx.attrs as SelectAttrs);
  if (out[dim] !== index) return { arg: 0, index: [...out] };
  return { arg: 1, index: out.filter((_, d) => d !== dim) };
}

export const selectScatterExplain: OpExplain = {
  pattern: 'select',
  local: (ctx, out, arg) => (selectScatterSource(ctx, out).arg === arg ? 1 : 0),
  tex: {
    forward: (s) => {
      const { dim, index } = s.attrs as SelectAttrs;
      return `\\operatorname{select\\_scatter}(${s.in[0]},${s.in[1]},${dim},${index})`;
    },
    local: () => '1',
  },
  selected: (ctx, out) => [selectScatterSource(ctx, out)],
};

export const sliceScatterExplain: OpExplain = {
  pattern: 'select',
  local: (ctx, out, arg) => (sliceScatterSource(ctx.inputs[0].shape, ctx.inputs[1].shape, ctx.attrs as SliceScatterAttrs, out).arg === arg ? 1 : 0),
  tex: {
    forward: (s) => {
      const { dim, start, end, step } = s.attrs as SliceScatterAttrs;
      const range = `${start ?? ''}:${end ?? ''}${step === 1 ? '' : `:${step}`}`;
      return `\\operatorname{slice\\_scatter}(${s.in[0]},${s.in[1]},${dim},${range})`;
    },
    local: () => '1',
  },
  selected: (ctx, out) => [sliceScatterSource(ctx.inputs[0].shape, ctx.inputs[1].shape, ctx.attrs as SliceScatterAttrs, out)],
};
