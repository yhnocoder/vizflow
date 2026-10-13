import type { OpExplain } from '../../autograd';
import type { ArangeAttrs, CausalMaskAttrs } from '../../autograd/ops/creation';

function creation(forward: (attrs: object) => string): OpExplain {
  return {
    pattern: 'select',
    local: () => 0,
    tex: { forward: (s) => forward(s.attrs), local: () => '0' },
    selected: () => [],
  };
}

export const arangeExplain = creation((attrs) => {
  const { start, end, step } = attrs as ArangeAttrs;
  const bounds = end === null || end === undefined ? [start] : step === 1 || step === undefined ? [start, end] : [start, end, step];
  return `\\operatorname{arange}(${bounds.join(',')})`;
});

export const causalMaskExplain = creation((attrs) => `\\operatorname{causal\\_mask}(${(attrs as CausalMaskAttrs).size})`);
