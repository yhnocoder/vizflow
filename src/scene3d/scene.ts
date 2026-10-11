import { WebGLRenderer } from 'three';
import type { Graph } from '../autograd';
import type { Steps } from '../steps';
import { SceneController } from './controller';
import { editableLayout } from './editconfig';
import type { LayoutItem, Vec3 } from './layout';
import type { LevelPlan } from './levels';
import type { SceneModule } from './modules';
import type { SceneParallel } from './modes';

export interface CameraOptions {
  direction?: Vec3;
  azimuth?: [number, number];
  polar?: [number, number];
  zoom?: [number, number];
  follow?: boolean;
}

export interface SceneOptions {
  steps?: Steps;
  layout: LayoutItem[];
  edit?: boolean;
  camera?: CameraOptions;
  hint?: string;
  labels?: Record<string, [number, number]>;
  modules?: SceneModule[];
  levels?: LevelPlan;
  sequence?: { get(): number[]; set(ids: number[]): void };
  parallel?: SceneParallel;
}

export interface SceneHandle {
  resetView(): void;
  dispose(): void;
}

export const NO_WEBGL_TEXT = '这个浏览器启用不了 WebGL，立体图显示不出来。下方的分步讲解和右侧数值仍然可用。';

export function mountScene(el: string | HTMLElement, g: Graph, options: SceneOptions): SceneHandle {
  const container = typeof el === 'string' ? document.querySelector<HTMLElement>(el) : el;
  if (container === null) throw new Error(`mountScene: container '${el}' not found`);
  const root = document.createElement('div');
  root.className = 'vf-scene';
  container.appendChild(root);
  let renderer: WebGLRenderer | undefined;
  try {
    renderer = new WebGLRenderer({ antialias: true });
  } catch {
    renderer = undefined;
  }
  if (renderer === undefined) {
    const notice = document.createElement('div');
    notice.className = 'vf-nogl';
    notice.textContent = NO_WEBGL_TEXT;
    root.appendChild(notice);
    return { resetView() {}, dispose: () => root.remove() };
  }
  const sceneOptions = options.edit === true ? { ...options, layout: editableLayout(options.layout), camera: { ...options.camera }, labels: { ...options.labels } } : options;
  const controller = new SceneController(root, g, sceneOptions, renderer);
  return { resetView: () => controller.resetView(), dispose: () => controller.dispose() };
}
