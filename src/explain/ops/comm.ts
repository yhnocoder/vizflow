import type { OpExplain } from '../../autograd';

export const allReduceExplain: OpExplain = {
  pattern: 'elementwise',
  local: () => 1,
  tex: {
    forward: (s) => `\\operatorname{all\\_reduce}(${s.in[0]})`,
    local: () => '1',
  },
};
