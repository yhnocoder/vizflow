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
  { ops: ['sub'], forward: '|y|', backward: '\\text{被减数 }0\\text{（被广播时 }|y|\\text{），减数 }|y|' },
  { ops: ['mul'], forward: '|y|', backward: '|y|' },
  { ops: ['div'], forward: '|y|', backward: '\\text{被除数 }|y|\\text{，除数 }3|y|' },
  { ops: ['neg', 'rotate_half', 'abs', 'clamp_min', 'remainder', 'cumsum', 'cummax'], forward: '|y|', backward: '|y|' },
  { ops: ['sign', 'eq', 'ge', 'greater', 'logical_not', 'floor_divide', 'bitwise_xor', 'bitwise_and'], forward: '|y|', backward: '0' },
  { ops: ['exp', 'log', 'pow', 'rsqrt', 'sqrt', 'reciprocal', 'erf', 'sigmoid', 'silu', 'softplus', 'relu', 'gelu', 'clamp'], forward: '|y|', backward: '2|y|' },
  { ops: ['sum', 'max'], forward: '|x|', backward: '0' },
  { ops: ['mean'], forward: '|x|', backward: '|x|' },
  { ops: ['logsumexp'], forward: '4|x|', backward: '3|x|' },
  { ops: ['softmax', 'log_softmax'], forward: '5|x|', backward: '4|x|' },
  { ops: ['nll_loss'], forward: '|t|', backward: '|t|' },
  { ops: ['gather', 'embedding', 'repeat_interleave', 'repeat', 'expand', 'index'], forward: '0', backward: '|y|' },
  { ops: ['index_add'], forward: '|s|', backward: '0' },
  { ops: ['view', 'reshape', 'transpose', 'permute', 'unsqueeze', 'narrow', 'select', 'identity', 'cat', 'pad', 'where', 'masked_fill', 'triu', 'scatter', 'topk_mask', 'topk_indices', 'one_hot', 'nonzero_static'], forward: '0', backward: '0' },
];
