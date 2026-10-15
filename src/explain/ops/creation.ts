import type { OpExplain } from '../../autograd';
import type { ArangeAttrs, CausalMaskAttrs, EyeAttrs, FillAttrs, FullAttrs } from '../../autograd/ops/creation';
import { formatNumber } from '../../tensor';

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

export const zerosExplain = creation((attrs) => `\\mathbf{0}_{${[(attrs as FillAttrs).size].flat().join('\\times ')}}`);

export const onesExplain = creation((attrs) => `\\mathbf{1}_{${[(attrs as FillAttrs).size].flat().join('\\times ')}}`);

export const fullExplain = creation((attrs) => `${formatNumber((attrs as FullAttrs).fill_value, { tex: true })}\\cdot\\mathbf{1}_{${[(attrs as FullAttrs).size].flat().join('\\times ')}}`);

export const eyeExplain = creation((attrs) => `I_{${(attrs as EyeAttrs).n}}`);

export const zerosLikeExplain: OpExplain = {
  pattern: 'select',
  local: () => 0,
  tex: { forward: (s) => `\\operatorname{zeros\\_like}(${s.in[0]})`, local: () => '0' },
  selected: () => [],
};

export const gradOfExplain: OpExplain = {
  pattern: 'select',
  local: () => 0,
  tex: { forward: (s) => `\\frac{\\partial\\ell}{\\partial ${s.in[0]}}`, local: () => '0' },
  selected: () => [],
};
