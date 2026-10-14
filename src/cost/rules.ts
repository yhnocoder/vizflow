export interface FlopRule {
  ops: string[];
  forward: string;
  backward: string;
}

export const FLOP_RULES: readonly FlopRule[] = [
  { ops: ['matmul', 'select_matmul', 'grouped_linear'], forward: '2mkn', backward: '2mkn' },
  { ops: ['solve_triangular'], forward: 'n|y|', backward: 'n|y|' },
  { ops: ['causal_conv1d'], forward: '2TCK', backward: '2TCK' },
  { ops: ['add'], forward: '|y|', backward: '0\\text{，被广播时 }|y|' },
  { ops: ['sub', 'rsub'], forward: '|y|', backward: '\\text{被减数 }0\\text{（被广播时 }|y|\\text{），减数 }|y|' },
  { ops: ['mul'], forward: '|y|', backward: '|y|' },
  { ops: ['div'], forward: '|y|', backward: '\\text{被除数 }|y|\\text{，除数 }3|y|' },
  { ops: ['neg', 'rotate_half', 'abs', 'clamp_min', 'remainder', 'cumsum', 'cummax'], forward: '|y|', backward: '|y|' },
  { ops: ['copysign'], forward: '|y|', backward: '\\text{第一个输入 }|y|\\text{，第二个输入 }0' },
  { ops: ['sign', 'eq', 'ge', 'greater', 'lt', 'logical_not', 'floor_divide', 'bitwise_xor', 'bitwise_and', 'bitwise_or'], forward: '|y|', backward: '0' },
  { ops: ['exp', 'log', 'cos', 'sin', 'pow', 'rsqrt', 'sqrt', 'reciprocal', 'erf', 'sigmoid', 'silu', 'softplus', 'relu', 'gelu', 'clamp'], forward: '|y|', backward: '2|y|' },
  { ops: ['sum', 'max'], forward: '|x|', backward: '0' },
  { ops: ['rrms'], forward: '2|x|+3|y|', backward: '|x|+4|y|' },
  { ops: ['mean'], forward: '|x|', backward: '|x|' },
  { ops: ['logsumexp'], forward: '4|x|', backward: '3|x|' },
  { ops: ['softmax', 'log_softmax'], forward: '5|x|', backward: '4|x|' },
  { ops: ['nll_loss'], forward: '|t|', backward: '|t|' },
  { ops: ['gather', 'embedding', 'repeat_interleave', 'repeat_kv', 'repeat', 'expand', 'index', 'sort', 'index_put'], forward: '0', backward: '|y|' },
  { ops: ['index_add'], forward: '|s|', backward: '0' },
  { ops: ['view', 'reshape', 'transpose', 'permute', 'unsqueeze', 'contiguous', 'narrow', 'select', 'identity', 'alias', 'cat', 'pad', 'where', 'masked_fill', 'triu', 'scatter', 'topk_mask', 'topk_indices', 'one_hot', 'nonzero_static', 'arange', 'causal_mask'], forward: '0', backward: '0' },
];
