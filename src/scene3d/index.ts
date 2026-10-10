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
export { board, stack, node, type LayoutItem, type StackOptions, type TemplateOptions, type Scales } from './layout';
