export interface FlopRule {
  ops: string[];
  forward: string;
  backward: string;
}

export const FLOP_RULES: readonly FlopRule[] = [
  { ops: ['matmul', 'select_matmul'], forward: '2mkn', backward: '2mkn' },
  { ops: ['causal_conv1d'], forward: '2TCK', backward: '2TCK' },
  { ops: ['add'], forward: '|y|', backward: '0\\text{，被广播时 }|y|' },
  { ops: ['sub'], forward: '|y|', backward: '\\text{被减数 }0\\text{（被广播时 }|y|\\text{），减数 }|y|' },
  { ops: ['mul'], forward: '|y|', backward: '|y|' },
  { ops: ['div'], forward: '|y|', backward: '\\text{被除数 }|y|\\text{，除数 }3|y|' },
  { ops: ['neg', 'rotate_half'], forward: '|y|', backward: '|y|' },
  { ops: ['exp', 'log', 'pow', 'rsqrt', 'sqrt', 'sigmoid', 'silu', 'softplus', 'relu', 'gelu'], forward: '|y|', backward: '2|y|' },
  { ops: ['sum', 'max'], forward: '|x|', backward: '0' },
  { ops: ['mean'], forward: '|x|', backward: '|x|' },
  { ops: ['logsumexp'], forward: '4|x|', backward: '3|x|' },
  { ops: ['softmax', 'log_softmax'], forward: '5|x|', backward: '4|x|' },
  { ops: ['nll_loss'], forward: '|t|', backward: '|t|' },
  { ops: ['gather', 'embedding', 'repeat_interleave'], forward: '0', backward: '|y|' },
  { ops: ['view', 'transpose', 'narrow', 'select', 'identity', 'cat', 'where', 'topk_mask'], forward: '0', backward: '0' },
];
