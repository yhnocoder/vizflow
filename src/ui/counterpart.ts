import type { StepInfo } from '../steps';

export function counterpart(list: readonly StepInfo[], current: number): number {
  const info = list[current];
  if (info.dir === 'backward') {
    const forward = list.findIndex((step) => step.dir === 'forward' && step.id === info.forwardId);
    return forward >= 0 ? forward : list.findIndex((step) => step.dir === 'forward');
  }
  const backwardOf = (position: number) => list.findIndex((step) => step.dir === 'backward' && step.forwardId === list[position].id);
  const forwardEnd = list.findIndex((step) => step.dir !== 'forward');
  const end = forwardEnd < 0 ? list.length : forwardEnd;
  for (let position = current; position < end; position++) {
    const backward = backwardOf(position);
    if (backward >= 0) return backward;
  }
  for (let position = current - 1; position >= 0; position--) {
    const backward = backwardOf(position);
    if (backward >= 0) return backward;
  }
  return list.findIndex((step) => step.dir === 'backward');
}
