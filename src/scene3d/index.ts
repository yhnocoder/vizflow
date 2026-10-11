import '../theme';
import sceneCss from './scene.css?inline';

const styleId = 'vf-scene3d-style';

if (!document.getElementById(styleId)) {
  const style = document.createElement('style');
  style.id = styleId;
  style.textContent = sceneCss;
  document.head.appendChild(style);
}

export { mountScene, type SceneOptions, type SceneHandle, type CameraOptions } from './scene';
export type { SceneModule } from './modules';
export { MODE_TOKENS, MODE_TOKEN_TEXT, RING_NOTE, byteSize, gradReduces, stepMode, type ParallelMode, type SceneParallel } from './modes';
export { autoLayout, type LayoutTree, type LayoutEntry } from './auto';
export { board, stack, node, type LayoutItem, type StackOptions, type TemplateOptions, type Scales } from './layout';
