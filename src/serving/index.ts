import '../theme';
import servingCss from './serving.css?inline';

const styleId = 'vf-serving-style';

if (!document.getElementById(styleId)) {
  const style = document.createElement('style');
  style.id = styleId;
  style.textContent = servingCss;
  document.head.append(style);
}

export { mountServing, tokenState, cellSize, type ServingViewOptions, type ServingHandle, type TokenState } from './view';
export { describeEvent, narrate, graphText } from './text';
export { simulateVllm } from './vllm';
export { simulateSglang } from './sglang';
export { BLOCK_SIZE } from './tokens';
export type * from './types';
export { SERVING_PRESETS, type ServingPreset } from './presets';
