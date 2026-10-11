import {
  AmbientLight,
  Color,
  CubicBezierCurve3,
  DirectionalLight,
  type InstancedMesh,
  Matrix4,
  Mesh,
  MeshBasicMaterial,
  type Object3D,
  OrthographicCamera,
  Raycaster,
  SRGBColorSpace,
  Scene,
  TubeGeometry,
  Vector2,
  Vector3,
  WebGLRenderer,
} from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import type { Graph, Node } from '../autograd';
import { type Explainer, type FlowEdge, explain } from '../explain';
import { render, renderText } from '../formula';
import { type StepInfo, type StepState, type Steps, allVisible } from '../steps';
import { formatNumber, offsetOf } from '../tensor';
import { type LinearRgb, type ThemeColors, onThemeChange, readTheme } from '../theme';
import { BarMesh, NodeMesh, PlateMesh, barBottom, barTop } from './blocks';
import { type ValueGrid, edgeCardElement, nodeCardElement, placePop, placeTip } from './cards';
import { barColor, nodeColor, outlineOpacity } from './colors';
import { type EditHost, mountEditor } from './edit';
import { editableLayout } from './editconfig';
import { LABEL_SAFE_MARGIN, type Label, LabelLayer, anchorOffset } from './labels';
import {
  type Block,
  type BoardGeometry,
  DISC_RADIUS,
  LINE_HEIGHT,
  type LayoutItem,
  type Vec3,
  classAlignment,
  elementAt,
  resolveLayout,
  rowsCols,
  targetColumn,
  tokenText,
} from './layout';
import { LineBatch } from './lines';
import { type Brace, type SceneModule, BRACE_OPACITY, blockSpan, braceOffset, bracePoints, braceTip, checkModules, layoutBraces } from './modules';
import { TICK_COUNT, barHeight, capFor, niceTicks, scaleFor } from './scales';
import { type EdgeStyle, edgeStyle } from './style';
import { Transition } from './transition';

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
}

export interface SceneHandle {
  resetView(): void;
  dispose(): void;
}

interface FoldingSteps extends Steps {
  folded?(id: string): boolean;
  setFolded?(id: string, folded: boolean): void;
  setAllFolded?(folded: boolean): void;
}

type PageStepInfo = StepInfo & { kind?: string; sub?: string };

export const DEFAULT_DIRECTION: Vec3 = [0.42, 0.55, 1];
export const DEFAULT_AZIMUTH: [number, number] = [-0.8, 0.7];
export const DEFAULT_POLAR: [number, number] = [0.3, 1.5];
export const DEFAULT_ZOOM: [number, number] = [0.1, 4];
export const DAMPING = 0.1;
export const PIXEL_RATIO_CAP = 2;
export const WHEEL_ZOOM_RATE = 0.0025;
export const WHEEL_SETTLE_MS = 120;
export const FOLLOW_PIXELS_PER_UNIT = 44;
export const FOLLOW_MIN_HALF_WIDTH = 17.6;
export const FOLLOW_RATE = 9;
export const FOLLOW_TOP_ROOM = 1.5;
export const FOLLOW_OVERVIEW_MARGIN = 6;
export const FOLLOW_STEP_MARGIN = 10;
export const FOLLOW_STEP_LIFT = 0.9;
export const FOLLOW_EDGE_FILL = 0.82;
export const FOLLOW_BOTTOM_SLACK = 2;
export const FOLLOW_TOP_SLACK = 1;
export const FAR_LEVELS = 2;
export const VALUE_MIN_ZOOM = 0.55;
export const VALUE_ALL_ROWS_ZOOM = 1.45;
export const EDGE_HOVER_DISTANCE = 6;
export const LONG_EDGE = 4.2;
export const EDGE_BOW = 0.9;
export const NO_WEBGL_TEXT = '这个浏览器启用不了 WebGL，立体图显示不出来。下方的分步讲解和右侧数值仍然可用。';
export const KEY_FRAME_TEXT = '墨线框 = 被取用的行或标签';
export const JUMP_PLACEHOLDER = '跳到模块…';

const FRAME_SEARCH_STEPS = 30;
const CAMERA_DISTANCE = 150;
const FOLLOW_SETTLE_Y = 0.01;
const FOLLOW_SETTLE_ZOOM = 0.002;
const MAX_FRAME_SECONDS = 0.05;
const AMBIENT_INTENSITY = 0.72;
const DIRECTIONAL_INTENSITY = 0.42;
const LEGACY_LIGHT_FACTOR = Math.PI;
const LIGHT_POSITION: Vec3 = [-4, 12, 8];
const CURVE_SEGMENTS = 14;
const CURVE_TANGENT = 0.45;
const BOW_TANGENT = 0.3;
const CLICK_SLOP = 5;
const VALUE_LABEL_GAP = 0.1;
const NODE_TITLE_OFFSET: Vec3 = [0.05, 0.28, -0.2];
const NODE_VALUE_OFFSET = 0.32;
const HIGHLIGHT_RADIUS = 0.03;
const HIGHLIGHT_SEGMENTS = 30;
const HIGHLIGHT_SIDES = 6;
const TIP_DIGITS = 3;
const LEVEL_PRECISION = 1000;

interface Renderable {
  block: Block;
  bars?: BarMesh;
  nodes?: NodeMesh;
  plate?: PlateMesh;
  transition: Transition;
  clipped: boolean[];
  cap: number;
  far: boolean;
  scale: { forward: number; backward: number };
  ticks: { forward: number[]; backward: number[] };
  title: Label | undefined;
  values: Label[];
  classes: Label[];
  rows: Label[];
  tickLabels: Label[];
  pickables: Object3D[];
}

interface DrawnEdge {
  edge: FlowEdge;
  fromNode: Node;
  toNode: Node;
  contract: boolean;
  endpoints: () => [Vec3, Vec3];
  flat: boolean;
  points: number[];
  style: EdgeStyle;
  screen: number[];
}

interface FollowGoal {
  y: number;
  zoom: number;
}

const tmpVector = new Vector3();

interface FrameItem {
  x: number;
  y: number;
  left: number;
  right: number;
  up: number;
  down: number;
}

interface Frustum {
  left: number;
  right: number;
  top: number;
  bottom: number;
}

function boundsOf(points: Vec3[]): { min: number[]; max: number[] } {
  const min = [Infinity, Infinity, Infinity];
  const max = [-Infinity, -Infinity, -Infinity];
  for (const p of points) for (let axis = 0; axis < 3; axis++) {
    min[axis] = Math.min(min[axis], p[axis]);
    max[axis] = Math.max(max[axis], p[axis]);
  }
  return { min, max };
}

function span(items: FrameItem[], k: number, at: (i: FrameItem) => number, before: (i: FrameItem) => number, after: (i: FrameItem) => number): { lo: number; hi: number; size: number } {
  let lo = Infinity;
  let hi = -Infinity;
  for (const item of items) {
    lo = Math.min(lo, at(item) * k - before(item));
    hi = Math.max(hi, at(item) * k + after(item));
  }
  return { lo, hi, size: hi - lo };
}

export function controlPoints(a: Vec3, b: Vec3, flat: boolean): [Vec3, Vec3, Vec3, Vec3] {
  if (flat) return [a, a.map((v, i) => v + (b[i] - v) / 3) as Vec3, a.map((v, i) => v + ((b[i] - v) * 2) / 3) as Vec3, b];
  const d = b[1] - a[1];
  if (d > LONG_EDGE) return [a, [a[0], a[1] + d * BOW_TANGENT, a[2] + EDGE_BOW], [b[0], b[1] - d * BOW_TANGENT, b[2] + EDGE_BOW], b];
  return [a, [a[0], a[1] + d * CURVE_TANGENT, a[2]], [b[0], b[1] - d * CURVE_TANGENT, b[2]], b];
}

function sampleCurve(a: Vec3, b: Vec3, flat: boolean): number[] {
  const c = controlPoints(a, b, flat);
  const points: number[] = [];
  for (let s = 0; s <= CURVE_SEGMENTS; s++) {
    const t = s / CURVE_SEGMENTS;
    const r = 1 - t;
    const w = [r * r * r, 3 * r * r * t, 3 * r * t * t, t * t * t];
    for (let axis = 0; axis < 3; axis++) points.push(w[0] * c[0][axis] + w[1] * c[1][axis] + w[2] * c[2][axis] + w[3] * c[3][axis]);
  }
  return points;
}

function segmentDistance(px: number, py: number, ax: number, ay: number, bx: number, by: number): number {
  const dx = bx - ax;
  const dy = by - ay;
  const len = dx * dx + dy * dy;
  const t = len === 0 ? 0 : Math.max(0, Math.min(1, ((px - ax) * dx + (py - ay) * dy) / len));
  return Math.hypot(px - (ax + t * dx), py - (ay + t * dy));
}

function texOf(label: Label, tex: string): void {
  label.setContent(tex, () => render(tex));
}

function chip(text: string): HTMLButtonElement {
  const button = document.createElement('button');
  button.className = 'vf-chip';
  button.type = 'button';
  button.textContent = text;
  return button;
}

function levelKey(y: number): number {
  return Math.round(y * LEVEL_PRECISION);
}

class SceneController implements EditHost {
  private readonly renderer: WebGLRenderer;
  private readonly scene = new Scene();
  readonly camera = new OrthographicCamera(-1, 1, 1, -1, 0.1, CAMERA_DISTANCE * 5);
  readonly controls: OrbitControls;
  private readonly direction: Vector3;
  private readonly explainer: Explainer;
  private readonly aligned: Map<Node, Node>;
  readonly labels: LabelLayer;
  private readonly tip: HTMLDivElement;
  private readonly pop: HTMLDivElement;
  private readonly lines: LineBatch;
  private readonly braceInk: LineBatch;
  private readonly moduleLabels: Label[];
  private readonly follow: boolean;
  private readonly zoomRange: [number, number];
  private braces: Brace[] = [];
  private stepKey: string | undefined;
  private goal: FollowGoal | undefined;
  private lastFrame: number | undefined;
  private wheelTimer = 0;
  private farOn = true;
  private lod = false;
  private stepNodes = new Set<Node>();
  private readonly highlightMaterial: MeshBasicMaterial;
  private readonly raycaster = new Raycaster();
  private readonly pointer = new Vector2();
  private readonly reduceMotion = matchMedia('(prefers-reduced-motion: reduce)');
  private readonly resizeObserver: ResizeObserver;
  private readonly unsubscribe: (() => void)[] = [];
  private renderables: Renderable[] = [];
  private boards: BoardGeometry[] = [];
  private edges: DrawnEdge[] = [];
  private theme: ThemeColors;
  private state: StepState;
  private width = 1;
  private height = 1;
  private frameHandle = 0;
  private geometryDirty = true;
  private labelsDirty = true;
  private screenVersion = 0;
  private screenCached = -1;
  private hoverEdge: DrawnEdge | undefined;
  private highlight: Mesh | undefined;
  private pinned: { node: Node; el: HTMLElement } | undefined;
  private downAt: [number, number] | undefined;
  private ovnav: { prev: HTMLButtonElement; next: HTMLButtonElement; text: HTMLSpanElement } | undefined;
  private disposed = false;

  constructor(
    readonly root: HTMLDivElement,
    private readonly g: Graph,
    readonly options: SceneOptions,
    renderer: WebGLRenderer,
  ) {
    this.renderer = renderer;
    this.theme = readTheme();
    this.explainer = explain(g);
    this.aligned = classAlignment(g.nodes);
    this.follow = options.camera?.follow === true;
    root.classList.toggle('vf-follow', this.follow);
    this.zoomRange = options.camera?.zoom ?? DEFAULT_ZOOM;
    this.state = this.currentState();
    renderer.setPixelRatio(Math.min(devicePixelRatio, PIXEL_RATIO_CAP));
    renderer.outputColorSpace = SRGBColorSpace;
    root.appendChild(renderer.domElement);
    this.labels = new LabelLayer(root, () => {
      this.reframe();
      this.labelsDirty = true;
      this.requestFrame();
    });
    this.buildChrome();
    this.tip = document.createElement('div');
    this.tip.className = 'vf-tip';
    this.tip.hidden = true;
    this.pop = document.createElement('div');
    this.pop.className = 'vf-pop';
    this.pop.hidden = true;
    root.append(this.tip, this.pop);

    const cameraOptions = options.camera ?? {};
    this.direction = new Vector3(...(cameraOptions.direction ?? DEFAULT_DIRECTION)).normalize();
    this.controls = new OrbitControls(this.camera, renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = DAMPING;
    this.controls.enablePan = true;
    this.controls.enableZoom = false;
    this.controls.screenSpacePanning = true;
    this.limitAzimuth();
    const [polarLo, polarHi] = cameraOptions.polar ?? DEFAULT_POLAR;
    this.controls.minPolarAngle = polarLo;
    this.controls.maxPolarAngle = polarHi;
    this.controls.addEventListener('change', () => {
      this.labelsDirty = true;
      this.requestFrame();
    });

    this.scene.add(new AmbientLight(0xffffff, AMBIENT_INTENSITY * LEGACY_LIGHT_FACTOR));
    const sun = new DirectionalLight(0xffffff, DIRECTIONAL_INTENSITY * LEGACY_LIGHT_FACTOR);
    sun.position.set(...LIGHT_POSITION);
    this.scene.add(sun);
    this.scene.background = new Color(...this.theme.surface).convertSRGBToLinear();
    this.lines = new LineBatch(this.theme.ink, this.theme.surface);
    this.scene.add(this.lines.object);
    this.highlightMaterial = new MeshBasicMaterial({ color: new Color(...this.theme.ink) });
    this.braceInk = new LineBatch(this.theme.ink);
    this.scene.add(this.braceInk.object);

    this.buildBlocks();
    const modules = options.modules ?? [];
    checkModules(modules, this.renderables.map((r) => r.block.node));
    this.moduleLabels = modules.map((module) => this.moduleLabel(module));
    this.buildTools();
    this.refresh();
    this.resize();
    this.placeCamera();
    this.controls.saveState();

    this.unsubscribe.push(g.subscribe(() => this.refresh()));
    if (options.steps !== undefined) this.unsubscribe.push(options.steps.subscribe(() => this.refresh()));
    this.unsubscribe.push(onThemeChange(() => this.applyTheme()));
    this.resizeObserver = new ResizeObserver(() => {
      this.resize();
      this.requestFrame();
    });
    this.resizeObserver.observe(root);
    this.bindPointer();
    if (options.edit === true) this.unsubscribe.push(mountEditor(this));
    this.requestFrame();
  }

  private limitAzimuth(): void {
    const azimuth = Math.atan2(this.direction.x, this.direction.z);
    const [azLo, azHi] = this.options.camera?.azimuth ?? DEFAULT_AZIMUTH;
    this.controls.minAzimuthAngle = azimuth + azLo;
    this.controls.maxAzimuthAngle = azimuth + azHi;
  }

  get canvas(): HTMLCanvasElement {
    return this.renderer.domElement;
  }

  get visible(): readonly Node[] {
    return this.options.steps?.visible ?? this.g.nodes;
  }

  private get steps(): FoldingSteps | undefined {
    return this.options.steps as FoldingSteps | undefined;
  }

  setDirection(direction: Vec3): void {
    this.direction.set(...direction).normalize();
    this.limitAzimuth();
    this.refresh();
    this.resetView();
  }

  relayout(reframe: boolean): void {
    this.resolve();
    for (const renderable of this.renderables) {
      if (renderable.plate === undefined) continue;
      this.scene.remove(renderable.plate.group);
      renderable.plate.dispose();
      renderable.plate = new PlateMesh(renderable.block, this.theme);
      this.scene.add(renderable.plate.group);
    }
    this.refresh();
    if (reframe) this.resetView();
  }

  blockAt(clientX: number, clientY: number): { node: Node; y: number } | undefined {
    const rect = this.renderer.domElement.getBoundingClientRect();
    this.pointer.set(((clientX - rect.left) / rect.width) * 2 - 1, -((clientY - rect.top) / rect.height) * 2 + 1);
    this.raycaster.setFromCamera(this.pointer, this.camera);
    const owners = new Map<Object3D, Renderable>();
    for (const renderable of this.renderables) {
      for (const object of renderable.pickables) owners.set(object, renderable);
      if (renderable.plate !== undefined) owners.set(renderable.plate.mesh, renderable);
    }
    const hits = this.raycaster.intersectObjects([...owners.keys()], false);
    if (hits.length === 0) return undefined;
    const block = owners.get(hits[0].object)!.block;
    return { node: block.node, y: block.elements[0].y };
  }

  private currentState(): StepState {
    return this.options.steps?.state() ?? allVisible(this.g);
  }

  private currentInfo(): PageStepInfo | undefined {
    const steps = this.options.steps;
    return steps === undefined ? undefined : (steps.list[steps.current] as PageStepInfo);
  }

  private buildChrome(): void {
    const key = document.createElement('div');
    key.className = 'vf-key';
    key.innerHTML = `<span><i class="vf-sw" style="background:var(--pos)"></i>正</span><span><i class="vf-sw" style="background:var(--neg)"></i>负</span><span>${KEY_FRAME_TEXT}</span>`;
    this.root.appendChild(key);
    if (this.options.hint !== undefined) {
      const hint = document.createElement('div');
      hint.className = 'vf-hint';
      hint.textContent = this.options.hint;
      this.root.appendChild(hint);
    }
  }

  private buildTools(): void {
    const tools = document.createElement('div');
    tools.className = 'vf-stage-tools';
    const modules = this.options.modules ?? [];
    if (this.follow && modules.length > 0) {
      const jump = document.createElement('select');
      jump.className = 'vf-chip';
      jump.setAttribute('aria-label', '跳到模块');
      const placeholder = document.createElement('option');
      placeholder.value = '';
      placeholder.textContent = JUMP_PLACEHOLDER;
      jump.appendChild(placeholder);
      modules.forEach((module, i) => {
        const option = document.createElement('option');
        option.value = String(i);
        option.textContent = module.title;
        jump.appendChild(option);
      });
      jump.addEventListener('change', () => {
        const brace = this.braces[Number(jump.value)];
        jump.value = '';
        if (brace === undefined) return;
        this.goal = { y: this.clampFollowY((brace.y0 + brace.y1) / 2, this.halfHeight()), zoom: 1 };
        this.refresh();
      });
      tools.appendChild(jump);
    }
    if (this.follow) {
      const far = chip('远处淡出');
      far.setAttribute('aria-pressed', 'true');
      far.addEventListener('click', () => {
        this.farOn = !this.farOn;
        far.setAttribute('aria-pressed', String(this.farOn));
        this.refresh();
      });
      tools.appendChild(far);
    }
    if (this.follow && modules.length > 0 && this.options.steps !== undefined) {
      const foldAll = chip('全部折叠');
      foldAll.addEventListener('click', () => this.steps?.setAllFolded?.(true));
      const unfoldAll = chip('全部展开');
      unfoldAll.addEventListener('click', () => this.steps?.setAllFolded?.(false));
      tools.append(foldAll, unfoldAll);
    }
    const reset = chip('复位视角');
    reset.addEventListener('click', () => this.resetView());
    tools.appendChild(reset);
    this.root.appendChild(tools);
    if (this.follow && this.options.steps !== undefined) {
      const steps = this.options.steps;
      const nav = document.createElement('div');
      nav.className = 'vf-ovnav';
      const button = (text: string, label: string, go: () => void) => {
        const b = document.createElement('button');
        b.className = 'vf-nav';
        b.type = 'button';
        b.textContent = text;
        b.setAttribute('aria-label', label);
        b.addEventListener('click', go);
        return b;
      };
      const prev = button('←', '上一步', () => steps.prev());
      const next = button('→', '下一步', () => steps.next());
      const text = document.createElement('span');
      text.className = 'vf-ovnav-text';
      nav.append(prev, next, text);
      this.root.appendChild(nav);
      this.ovnav = { prev, next, text };
    }
  }

  private updateNav(): void {
    const steps = this.options.steps;
    const info = this.currentInfo();
    if (this.ovnav === undefined || steps === undefined || info === undefined) return;
    this.ovnav.prev.disabled = steps.current === 0;
    this.ovnav.next.disabled = steps.current === steps.list.length - 1;
    const key = `${steps.current}:${info.label}:${info.sub ?? ''}`;
    if (this.ovnav.text.dataset.key === key) return;
    this.ovnav.text.dataset.key = key;
    const small = document.createElement('small');
    small.textContent = `${info.dir === 'forward' ? '前向' : '反向'}${info.sub ? ` · ${info.sub}` : ''}`;
    const label = document.createElement('span');
    renderText(info.label, label);
    this.ovnav.text.replaceChildren(small, ...label.childNodes);
  }

  private resolve(): void {
    const resolved = resolveLayout(this.options.layout, {
      visible: this.visible,
      aligned: this.aligned,
      targetOf: targetColumn,
      focus: this.state.focus,
    });
    this.boards = resolved.boards;
    if (this.renderables.length === 0) {
      this.renderables = resolved.blocks.map((block) => this.createRenderable(block));
    } else {
      for (const block of resolved.blocks) {
        const renderable = this.renderables.find((r) => r.block.node === block.node);
        if (renderable !== undefined) renderable.block = block;
      }
    }
  }

  private buildBlocks(): void {
    this.resolve();
    for (const renderable of this.renderables) {
      const group = renderable.bars?.group ?? renderable.nodes?.group;
      if (group !== undefined) this.scene.add(group);
      if (renderable.plate !== undefined) this.scene.add(renderable.plate.group);
    }
  }

  private createRenderable(block: Block): Renderable {
    const node = block.node;
    const grad = node.grad;
    const cap = capFor(block.shape, node.shape.length);
    const scale = {
      forward: block.scales.forward ?? scaleFor(node.value.data, cap),
      backward: block.scales.backward ?? scaleFor(grad === null ? [] : grad.data, cap),
    };
    const renderable: Renderable = {
      block,
      transition: new Transition(block.elements.length, () => this.reduceMotion.matches),
      clipped: block.elements.map(() => false),
      cap,
      far: false,
      scale,
      ticks: {
        forward: block.axis === undefined ? [] : niceTicks(node.value.data, scale.forward, TICK_COUNT, cap),
        backward: block.axis === undefined || grad === null ? [] : niceTicks(grad.data, scale.backward, TICK_COUNT, cap),
      },
      title: undefined,
      values: [],
      classes: [],
      rows: [],
      tickLabels: [],
      pickables: [],
    };
    if (block.shape === 'bar' || block.shape === 'tile') {
      renderable.bars = new BarMesh(block, this.theme);
      renderable.pickables.push(renderable.bars.mesh);
      if (block.shape === 'bar') renderable.values = block.elements.map(() => this.labels.add('val'));
      renderable.title = this.labels.add('ttl');
    } else {
      renderable.nodes = new NodeMesh(block, this.theme);
      renderable.pickables.push(...renderable.nodes.meshes);
      renderable.values = block.elements.map(() => this.labels.add('nv'));
      renderable.title = this.labels.add('ntl');
    }
    this.bindCard(renderable.title, node);
    if (block.plate !== undefined) renderable.plate = new PlateMesh(block, this.theme);
    renderable.classes = block.classLabels.map((text) => {
      const label = this.labels.add(block.role === 'board-b' ? 'colh' : 'cls');
      if (block.role === 'planar') label.under = renderable.values.filter((_, i) => block.elements[i].col === text.col);
      return label;
    });
    renderable.rows = block.rowLabels.map(() => this.labels.add('kk'));
    if (block.axis !== undefined) renderable.tickLabels = Array.from({ length: Math.max(renderable.ticks.forward.length, renderable.ticks.backward.length) }, () => this.labels.add('tick'));
    return renderable;
  }

  private foldable(module: SceneModule): string | undefined {
    const steps = this.steps;
    if (module.id === undefined || module.fixed === true || steps?.setFolded === undefined || steps.folded === undefined) return undefined;
    return module.id;
  }

  private moduleLabel(module: SceneModule): Label {
    const side = module.side ?? 'left';
    const id = this.foldable(module);
    const label = this.labels.add('mod', `vf-mod-${side}${id === undefined ? '' : ' vf-foldable'}`);
    label.anchor = side === 'left' ? 'right-middle' : 'left-middle';
    label.el.dataset.module = module.title;
    if (id !== undefined) label.el.addEventListener('click', () => this.steps?.setFolded?.(id, this.steps.folded?.(id) !== true));
    return label;
  }

  private moduleContent(label: Label, module: SceneModule): void {
    const id = this.foldable(module);
    const folded = id === undefined ? undefined : this.steps?.folded?.(id);
    const action = folded === undefined ? '' : folded ? '▸ 展开' : '▾ 折叠';
    const brief = this.lod;
    const subtitle = brief ? undefined : module.subtitle;
    const act = brief ? '' : action;
    label.setContent(`${module.title}\n${subtitle ?? ''}\n${act}`, () => {
      const fragment = document.createDocumentFragment();
      const title = document.createElement('span');
      title.className = 'vf-mod-title';
      title.textContent = module.title;
      fragment.appendChild(title);
      if (subtitle !== undefined) {
        const sub = document.createElement('span');
        sub.className = 'vf-mod-sub';
        sub.textContent = subtitle;
        fragment.appendChild(sub);
      }
      if (act !== '') {
        const el = document.createElement('span');
        el.className = 'vf-mod-act';
        el.textContent = act;
        fragment.appendChild(el);
      }
      return fragment;
    });
  }

  private updateBraces(): void {
    const modules = this.options.modules ?? [];
    if (modules.length === 0) return;
    const { min, max } = boundsOf(this.framingPoints(this.renderables));
    this.braces = layoutBraces(
      modules,
      this.renderables.map((r) => blockSpan(r.block)),
      { xMin: min[0], xMax: max[0] },
    );
    this.braces.forEach((brace, i) => {
      const label = this.moduleLabels[i];
      this.moduleContent(label, brace.module);
      label.visible = true;
      if (label.pos === undefined) label.pos = braceTip(brace);
    });
  }

  private placeBraces(): void {
    if (this.braces.length === 0) return;
    const view = this.viewDirection();
    const direction: Vec3 = [view.x, view.y, view.z];
    const polylines = this.braces.map((brace, i) => {
      const offset = braceOffset(brace, direction);
      this.moduleLabels[i].pos = braceTip(brace, offset);
      return { points: bracePoints(brace, offset), opacity: BRACE_OPACITY, dashed: false };
    });
    this.braceInk.setPolylines(polylines);
  }

  private bindCard(label: Label, node: Node): void {
    const show = () => this.showNodeCard(node, label.el);
    label.el.addEventListener('pointerenter', show);
    label.el.addEventListener('click', show);
    label.el.addEventListener('pointerleave', () => this.hideNodeCard());
  }

  private renderableOf(node: Node): Renderable | undefined {
    return this.renderables.find((r) => r.block.node === node);
  }

  private targetOf(block: Block, row: number): number | undefined {
    const label = this.aligned.get(block.node);
    return label === undefined ? undefined : targetColumn(label, row);
  }

  private valuesOf(node: Node): ArrayLike<number> {
    if (this.state.dir === 'forward') return node.value.data;
    const grad = node.grad;
    return grad === null ? new Float64Array(node.value.size) : grad.data;
  }

  private zoomNow(): number {
    return this.goal?.zoom ?? this.camera.zoom;
  }

  private markFar(): void {
    const levels = [...new Set(this.renderables.map((r) => levelKey(r.block.elements[0].y)))].sort((a, b) => a - b);
    const levelOf = (r: Renderable) => levels.indexOf(levelKey(r.block.elements[0].y));
    const stepLevels = this.renderables.filter((r) => r.block.node.kind === 'op' && this.state.active(r.block.node)).map(levelOf);
    const enabled = this.follow && this.farOn && !this.lod && stepLevels.length > 0;
    for (const r of this.renderables) {
      const level = levelOf(r);
      r.far = enabled && !this.state.active(r.block.node) && Math.min(...stepLevels.map((s) => Math.abs(s - level))) > FAR_LEVELS;
    }
  }

  private refresh(): void {
    if (this.disposed) return;
    this.state = this.currentState();
    this.resolve();
    const now = performance.now();
    const dir = this.state.dir;
    const backward = dir === 'backward';
    const info = this.currentInfo();
    const overview = dir === 'forward' && this.state.index === 0;
    this.lod = this.follow && (overview || info?.kind === 'inputs');
    this.stepNodes = new Set(info?.nodes ?? this.visible);
    this.markFar();
    const stepKey = `${dir}:${this.state.index}`;
    if (this.follow && this.stepKey !== undefined && stepKey !== this.stepKey) {
      this.goal = this.followGoal();
      if (this.reduceMotion.matches) this.applyGoal(1);
    }
    this.stepKey = stepKey;
    const zoom = this.zoomNow();
    for (const renderable of this.renderables) {
      const { block } = renderable;
      const node = block.node;
      const revealed = this.state.revealed(node);
      const active = this.state.active(node);
      const far = renderable.far;
      const values = this.valuesOf(node);
      const scale = renderable.scale[dir];
      const heights = new Float64Array(block.elements.length);
      for (let i = 0; i < heights.length; i++) {
        const { height, clipped } = barHeight(values[i], scale, revealed, renderable.cap);
        heights[i] = height;
        renderable.clipped[i] = revealed && clipped;
      }
      renderable.transition.retarget(heights, now);
      const focusedRow = (row: number) => !block.batched || row === this.state.focus;
      const front = Math.max(...block.elements.map((e) => e.z));
      const showValues = revealed && this.stepNodes.has(node) && !this.lod && (!this.follow || zoom >= VALUE_MIN_ZOOM);
      const named = revealed && !this.lod && !far;
      const title = renderable.title!;
      title.el.dataset.node = node.name;
      title.offset = this.options.labels?.[node.name];
      title.visible = named;
      texOf(title, backward ? `\\partial\\ell/\\partial ${node.tex}` : node.tex);
      if (renderable.bars !== undefined) {
        title.pos = [block.title.x, block.title.y, block.title.z];
        title.anchor = block.title.anchor === 'left' ? 'right-middle' : block.title.anchor === 'below' ? 'bottom-center' : 'top-center';
        renderable.values.forEach((label, i) => {
          const element = block.elements[i];
          label.setText(formatNumber(values[i]));
          label.el.dataset.node = node.name;
          label.el.dataset.index = element.index.join(',');
          const rowShown = !this.follow || element.z === front || zoom >= VALUE_ALL_ROWS_ZOOM;
          label.visible = showValues && focusedRow(element.row) && rowShown;
        });
      }
      if (renderable.nodes !== undefined) {
        const mesh = renderable.nodes;
        const first = block.elements[0];
        const center = first.y + mesh.height / 2;
        title.pos = [first.x + NODE_TITLE_OFFSET[0], center + NODE_TITLE_OFFSET[1], first.z + NODE_TITLE_OFFSET[2]];
        title.anchor = 'above-right';
        renderable.values.forEach((label, i) => {
          const element = block.elements[i];
          label.setText(formatNumber(values[i]));
          label.el.dataset.node = node.name;
          label.el.dataset.index = element.index.join(',');
          label.pos = [element.x + NODE_VALUE_OFFSET, element.y + mesh.height / 2, element.z];
          label.visible = showValues && focusedRow(element.row);
        });
        const colors = block.elements.map((element, i) => nodeColor(values[i], this.theme, { dir, revealed, active, far, focused: focusedRow(element.row) }));
        mesh.update(block.elements, colors, block.elements.map(() => revealed));
      }
      renderable.classes.forEach((label, i) => {
        const text = block.classLabels[i];
        const target = this.targetOf(block, block.rows - 1);
        const isTarget = this.aligned.has(node) && !block.batched && target === text.col;
        label.setText(block.role === 'planar' && isTarget ? `${text.text} · 目标` : text.text);
        label.setFlag('vf-on', isTarget);
        label.pos = [text.x, text.y, text.z];
        label.anchor = text.anchor === 'above' ? 'top-center' : text.anchor === 'left' ? 'right-middle' : 'bottom-center';
        label.visible = named;
      });
      renderable.rows.forEach((label, i) => {
        const text = block.rowLabels[i];
        label.setText(text.text);
        label.pos = [text.x, text.y, text.z];
        label.visible = named;
      });
      const ticks = renderable.ticks[dir];
      renderable.tickLabels.forEach((label, i) => {
        const axis = block.axis!;
        if (i < ticks.length) {
          label.setText(String(ticks[i]).replace('-', '−'));
          label.pos = [axis.x - 0.18, block.elements[0].y + ticks[i] * scale, axis.z];
          label.visible = named;
        } else label.visible = false;
      });
      if (renderable.plate !== undefined) {
        const frameXs = block.frameRows.map((_, r) => {
          const target = this.targetOf(block, r);
          const column = block.elements.find((e) => e.col === (target ?? 0));
          return column?.x ?? block.elements[0].x;
        });
        renderable.plate.update({ revealed, far }, frameXs, ticks, scale);
      }
    }
    this.updateBraces();
    this.buildEdges();
    this.updateNav();
    this.geometryDirty = true;
    this.labelsDirty = true;
    if (this.pinned !== undefined) this.showNodeCard(this.pinned.node, this.pinned.el);
    if (this.hoverEdge !== undefined) {
      const still = this.edges.find((e) => e.edge.id === this.hoverEdge!.edge.id);
      this.setHoverEdge(still !== undefined && this.state.dir === 'backward' && this.state.revealed(still.fromNode) ? still : undefined);
      if (this.hoverEdge === undefined && this.pinned === undefined) this.pop.hidden = true;
    }
    this.requestFrame();
  }

  private anchorOut(node: Node, index: number[]): Vec3 | undefined {
    const renderable = this.renderableOf(node);
    if (renderable === undefined) return undefined;
    const element = elementAt(renderable.block, index);
    if (renderable.nodes !== undefined) return renderable.nodes.anchorOut(element);
    return barTop(element, renderable.transition.current[offsetOf(node.shape, index)]);
  }

  private anchorIn(node: Node, index: number[]): Vec3 | undefined {
    const renderable = this.renderableOf(node);
    if (renderable === undefined) return undefined;
    const element = elementAt(renderable.block, index);
    if (renderable.nodes !== undefined) return renderable.nodes.anchorIn(element);
    return barBottom(element, renderable.transition.current[offsetOf(node.shape, index)]);
  }

  private buildEdges(): void {
    const rowOf = (node: Node, index: number[]) => {
      const renderable = this.renderableOf(node);
      return renderable !== undefined && renderable.block.batched ? elementAt(renderable.block, index).row : undefined;
    };
    const drawn: DrawnEdge[] = [];
    this.edges = drawn;
    if (this.lod) return;
    const backward = this.state.dir === 'backward';
    const overview = this.state.dir === 'forward' && this.state.index === 0;
    for (const edge of this.explainer.edges()) {
      let entry: Omit<DrawnEdge, 'points' | 'style' | 'screen'> | undefined;
      let rows: (number | undefined)[] = [];
      if ('cell' in edge.from || 'cell' in edge.to) {
        const board = this.boards.find((b) => b.op === edge.op);
        if (board === undefined || edge.arg === 1) continue;
        if (!this.renderableOf(board.a.node) || !this.renderableOf(board.b.node) || !this.renderableOf(board.out.node)) continue;
        const [r, k, i] = 'cell' in edge.to ? edge.to.cell.index : (edge.from as { cell: { index: [number, number, number] } }).cell.index;
        const y = board.y + LINE_HEIGHT;
        const tile = (kk: number, ii: number) => board.b.elements.find((e) => e.row === kk && e.col === ii)!;
        const [tileW, tileD] = board.b.footprint;
        const outElement = board.out.elements.find((e) => e.row === r && e.col === i)!;
        rows = [board.out.batched ? r : undefined];
        if (edge.arg === 0) {
          const aElement = board.a.elements.find((e) => e.row === r && e.col === k)!;
          const to: Vec3 = [tile(k, i).x - tileW / 2 - 0.01, y, tile(k, i).z];
          if (i === 0 && board.sharedA) {
            entry = { edge, fromNode: board.a.node, toNode: edge.op, contract: true, endpoints: () => [this.anchorOut(board.a.node, aElement.index)!, to], flat: false };
          } else {
            const from: Vec3 = i === 0 ? [aElement.x, y, aElement.z - board.a.footprint[1] / 2 - 0.02] : [tile(k, i - 1).x + tileW / 2 + 0.01, y, tile(k, i - 1).z];
            entry = { edge, fromNode: board.a.node, toNode: edge.op, contract: true, endpoints: () => [from, to], flat: true };
          }
        } else {
          const last = k === board.b.rows - 1;
          const from: Vec3 = [tile(k, i).x, y, tile(k, i).z + tileD / 2 + 0.01];
          const to: Vec3 = last ? [outElement.x, y, outElement.z - board.out.footprint[1] / 2 - 0.02] : [tile(k + 1, i).x, y, tile(k + 1, i).z - tileD / 2 - 0.01];
          entry = { edge, fromNode: board.b.node, toNode: edge.op, contract: true, endpoints: () => [from, to], flat: true };
        }
      } else {
        const from = edge.from;
        const to = edge.to;
        if (!this.renderableOf(from.node) || !this.renderableOf(to.node)) continue;
        rows = [rowOf(from.node, from.index), rowOf(to.node, to.index)];
        entry = {
          edge,
          fromNode: from.node,
          toNode: to.node,
          contract: false,
          endpoints: () => [this.anchorOut(from.node, from.index)!, this.anchorIn(to.node, to.index)!],
          flat: false,
        };
      }
      const style = edgeStyle({
        dir: this.state.dir,
        overview,
        contract: entry.contract,
        zeroGrad: this.explainer.zeroGrad(edge),
        focused: rows.every((row) => row === undefined || row === this.state.focus),
        far: this.renderableOf(entry.fromNode)?.far === true && this.renderableOf(entry.toNode)?.far === true,
        current: this.stepNodes.has(backward ? entry.fromNode : entry.toNode),
        revealedFrom: this.state.revealed(entry.fromNode),
        revealedTo: this.state.revealed(entry.toNode),
      });
      if (style === null) continue;
      drawn.push({ ...entry, points: [], style, screen: [] });
    }
  }

  private updateGeometry(): void {
    for (const renderable of this.renderables) {
      if (renderable.bars === undefined) continue;
      const { block } = renderable;
      const node = block.node;
      const revealed = this.state.revealed(node);
      const active = this.state.active(node);
      const values = this.valuesOf(node);
      const heights = renderable.transition.current;
      const colors: LinearRgb[] = [];
      const outline: number[] = [];
      const far = renderable.far;
      block.elements.forEach((element, i) => {
        const focused = !block.batched || element.row === this.state.focus;
        colors.push(barColor(values[i], this.theme, { active: revealed && active, far, focused }));
        outline.push(outlineOpacity({ active: active && revealed, far, tile: block.shape === 'tile' }));
        const label = renderable.values[i];
        if (label !== undefined) {
          const h = heights[i];
          label.anchor = h < 0 ? 'bottom-center' : 'top-center';
          label.pos = [element.x, element.y + (h >= 0 ? h + VALUE_LABEL_GAP : h - VALUE_LABEL_GAP), element.z];
        }
      });
      renderable.bars.update(block.elements, { heights, clipped: renderable.clipped, colors, outlineOpacity: outline });
    }
    for (const d of this.edges) {
      const [a, b] = d.endpoints();
      d.points = sampleCurve(a, b, d.flat);
    }
    this.lines.setPolylines(this.edges.map((d) => ({ points: d.points, opacity: d.style.opacity, dashed: d.style.dashed })));
    this.updateHighlight();
  }

  private updateHighlight(): void {
    if (this.highlight !== undefined) {
      this.scene.remove(this.highlight);
      this.highlight.geometry.dispose();
      this.highlight = undefined;
    }
    const d = this.hoverEdge;
    if (d === undefined) return;
    const [a, b] = d.endpoints();
    const [p0, p1, p2, p3] = controlPoints(a, b, d.flat).map((p) => new Vector3(...p));
    const curve = new CubicBezierCurve3(p0, p1, p2, p3);
    this.highlight = new Mesh(new TubeGeometry(curve, HIGHLIGHT_SEGMENTS, HIGHLIGHT_RADIUS / Math.sqrt(this.camera.zoom), HIGHLIGHT_SIDES, false), this.highlightMaterial);
    this.scene.add(this.highlight);
  }

  private requestFrame(): void {
    if (this.frameHandle !== 0 || this.disposed) return;
    this.frameHandle = requestAnimationFrame((now) => {
      this.frameHandle = 0;
      this.frame(now);
    });
  }

  private applyGoal(k: number): boolean {
    const goal = this.goal;
    if (goal === undefined) return false;
    const dy = (goal.y - this.controls.target.y) * k;
    this.controls.target.y += dy;
    this.camera.position.y += dy;
    this.camera.zoom += (goal.zoom - this.camera.zoom) * k;
    this.camera.updateProjectionMatrix();
    if (Math.abs(goal.y - this.controls.target.y) < FOLLOW_SETTLE_Y && Math.abs(goal.zoom - this.camera.zoom) < FOLLOW_SETTLE_ZOOM) this.goal = undefined;
    this.labelsDirty = true;
    return true;
  }

  private frame(now: number): void {
    if (this.disposed) return;
    const dt = this.lastFrame === undefined ? 1 / 60 : Math.min((now - this.lastFrame) / 1000, MAX_FRAME_SECONDS);
    this.lastFrame = now;
    let stepped = false;
    for (const renderable of this.renderables) {
      if (renderable.transition.step(now)) stepped = true;
    }
    const k = this.reduceMotion.matches ? 1 : 1 - Math.exp(-dt * FOLLOW_RATE);
    const following = this.applyGoal(k);
    const running = this.renderables.some((r) => r.transition.active) || this.goal !== undefined;
    if (this.geometryDirty || stepped) {
      this.updateGeometry();
      this.geometryDirty = false;
      this.labelsDirty = true;
    }
    const moved = this.controls.update();
    if (this.labelsDirty || moved || following) this.placeBraces();
    this.renderer.render(this.scene, this.camera);
    if (this.labelsDirty || moved) {
      this.labels.project(this.camera, this.width, this.height, !this.follow);
      this.labelsDirty = false;
      this.screenVersion++;
    }
    if (running || moved) this.requestFrame();
    else this.lastFrame = undefined;
  }

  private halfHeight(): number {
    return (this.camera.top - this.camera.bottom) / 2;
  }

  private resize(): void {
    this.width = Math.max(1, this.root.clientWidth);
    this.height = Math.max(1, this.root.clientHeight);
    this.renderer.setSize(this.width, this.height, false);
    this.reframe();
    this.labelsDirty = true;
  }

  private reframe(): void {
    if (this.follow) this.followFrustum();
    else this.frameAll();
  }

  private followFrustum(): void {
    const aspect = this.width / this.height;
    const half = Math.max(this.height / (2 * FOLLOW_PIXELS_PER_UNIT), FOLLOW_MIN_HALF_WIDTH / aspect);
    this.camera.left = -half * aspect;
    this.camera.right = half * aspect;
    this.camera.top = half;
    this.camera.bottom = -half;
    this.camera.updateProjectionMatrix();
  }

  private levelRange(): { lo: number; top: number } {
    const ys = this.renderables.map((r) => r.block.elements[0].y);
    return { lo: Math.min(...ys), top: Math.max(...ys) + FOLLOW_TOP_ROOM };
  }

  private clampFollowY(y: number, half: number): number {
    const { lo, top } = this.levelRange();
    return Math.min(Math.max(y, lo + half * FOLLOW_EDGE_FILL - FOLLOW_BOTTOM_SLACK), top - half * FOLLOW_EDGE_FILL + FOLLOW_TOP_SLACK);
  }

  private followGoal(): FollowGoal {
    const hh = this.halfHeight();
    const { lo, top } = this.levelRange();
    const keys = this.lod ? [] : this.renderables.filter((r) => this.state.active(r.block.node));
    if (keys.length === 0) return { y: (lo + top - FOLLOW_TOP_ROOM) / 2, zoom: Math.min(1, (2 * hh) / (top - lo + FOLLOW_OVERVIEW_MARGIN)) };
    const ys = keys.map((r) => r.block.elements[0].y);
    const kLo = Math.min(...ys);
    const kHi = Math.max(...ys);
    const zoom = Math.min(1, (2 * hh) / (kHi - kLo + FOLLOW_STEP_MARGIN));
    return { y: this.clampFollowY((kLo + kHi) / 2 + FOLLOW_STEP_LIFT, hh / zoom), zoom };
  }

  private framingPoints(renderables: Renderable[]): Vec3[] {
    const points: Vec3[] = [];
    for (const renderable of renderables) {
      const { block } = renderable;
      const [w, d] = block.footprint;
      const values = block.node.value.data;
      const grad = block.node.grad;
      let high = 0;
      let low = 0;
      for (let i = 0; i < values.length; i++) {
        for (const h of [values[i] * renderable.scale.forward, grad === null ? 0 : grad.data[i] * renderable.scale.backward]) {
          high = Math.max(high, Math.min(h, renderable.cap));
          low = Math.min(low, Math.max(h, -renderable.cap));
        }
      }
      const top = renderable.nodes === undefined ? high : renderable.nodes.height;
      const bottom = renderable.nodes === undefined ? low : 0;
      const radius = block.shape === 'disc' ? DISC_RADIUS : w / 2;
      const depth = block.shape === 'disc' ? DISC_RADIUS : d / 2;
      for (const e of block.elements) {
        for (const dx of [-radius, radius]) for (const dz of [-depth, depth]) points.push([e.x + dx, e.y + bottom, e.z + dz], [e.x + dx, e.y + top, e.z + dz]);
      }
      if (block.plate !== undefined) {
        const p = block.plate;
        for (const dx of [-p.w / 2, p.w / 2]) for (const dz of [-p.d / 2, p.d / 2]) points.push([p.x + dx, p.y, p.z + dz]);
      }
      points.push([block.title.x, block.title.y, block.title.z]);
      for (const text of [...block.classLabels, ...block.rowLabels]) points.push([text.x, text.y, text.z]);
      if (block.axis !== undefined) points.push([block.axis.x - 0.5, block.elements[0].y, block.axis.z]);
      points.push(...block.extent);
    }
    return points;
  }

  private allPoints(): Vec3[] {
    const points = this.framingPoints(this.renderables);
    for (const brace of this.braces) points.push([brace.x, brace.y0, brace.z], [brace.x, brace.y1, brace.z], braceTip(brace));
    return points;
  }

  private center(points: Vec3[]): Vector3 {
    const { min, max } = boundsOf(points);
    return new Vector3((min[0] + max[0]) / 2, (min[1] + max[1]) / 2, (min[2] + max[2]) / 2);
  }

  private placeCamera(): void {
    const center = this.center(this.allPoints());
    let zoom = 1;
    if (this.follow) {
      const goal = this.followGoal();
      center.y = goal.y;
      zoom = goal.zoom;
    }
    this.goal = undefined;
    this.camera.position.copy(center).addScaledVector(this.direction, CAMERA_DISTANCE);
    this.camera.lookAt(center);
    this.controls.target.copy(center);
    this.camera.zoom = zoom;
    this.camera.updateProjectionMatrix();
    this.controls.update();
  }

  private computeFrame(points: Vec3[], labels: readonly Label[], center: Vector3, direction: Vector3): Frustum {
    const eye = center.clone().addScaledVector(direction, CAMERA_DISTANCE);
    const view = new Matrix4().lookAt(eye, center, this.camera.up).setPosition(eye).invert();
    const marginX = LABEL_SAFE_MARGIN;
    const marginY = LABEL_SAFE_MARGIN;
    const items: FrameItem[] = points.map((p) => {
      tmpVector.set(...p).applyMatrix4(view);
      return { x: tmpVector.x, y: tmpVector.y, left: marginX, right: marginX, up: marginY, down: marginY };
    });
    const geometry = { x: span(items, 1, (i) => i.x, () => 0, () => 0), y: span(items, 1, (i) => i.y, () => 0, () => 0) };
    for (const label of labels) {
      if (label.pos === undefined) continue;
      const [w, h] = label.frameSize();
      if (w === 0 && h === 0) continue;
      const [ax, ay] = anchorOffset(label.anchor);
      tmpVector.set(...label.pos).applyMatrix4(view);
      const [dx, dy] = label.offset ?? [0, 0];
      items.push({
        x: tmpVector.x,
        y: tmpVector.y,
        left: LABEL_SAFE_MARGIN - ax * w - dx,
        right: LABEL_SAFE_MARGIN + (ax + 1) * w + dx,
        up: LABEL_SAFE_MARGIN - ay * h - dy,
        down: (ay + 1) * h + dy + LABEL_SAFE_MARGIN,
      });
    }
    const reserve = this.chromeBottom();
    const availableH = Math.max(this.height - reserve, this.height / 2);
    const availableW = Math.max(this.width - this.chromeRight(), this.width / 2);
    const spanX = (k: number) => span(items, k, (i) => i.x, (i) => i.left, (i) => i.right);
    const spanY = (k: number) => span(items, k, (i) => -i.y, (i) => i.up, (i) => i.down);
    const fits = (k: number) => spanX(k).size <= availableW && spanY(k).size <= availableH;
    let k = Math.min((availableW - 2 * marginX) / geometry.x.size, (availableH - 2 * marginY) / geometry.y.size);
    if (!fits(k) && fits(0)) {
      let lo = 0;
      let hi = k;
      for (let i = 0; i < FRAME_SEARCH_STEPS; i++) {
        const mid = (lo + hi) / 2;
        if (fits(mid)) lo = mid;
        else hi = mid;
      }
      k = lo;
    }
    const unit = 1 / k;
    const sx = spanX(k);
    const sy = spanY(k);
    const cx = ((sx.lo + sx.hi) / 2) * unit;
    const cy = -((sy.lo + sy.hi) / 2) * unit;
    const left = cx - (availableW * unit) / 2;
    const top = cy + (reserve + availableH / 2) * unit;
    return { left, right: left + this.width * unit, top, bottom: top - this.height * unit };
  }

  private frameAll(): void {
    const points = this.allPoints();
    const frame = this.computeFrame(points, this.labels.labels, this.center(points), this.direction);
    this.camera.left = frame.left;
    this.camera.right = frame.right;
    this.camera.top = frame.top;
    this.camera.bottom = frame.bottom;
    this.camera.updateProjectionMatrix();
  }

  private viewDirection(): Vector3 {
    return this.camera.position.clone().sub(this.controls.target).normalize();
  }

  private chromeRight(): number {
    const panel = this.root.querySelector<HTMLElement>('.vf-edit[open]');
    return panel === null ? 0 : this.width - panel.offsetLeft;
  }

  private chromeBottom(): number {
    let bottom = 0;
    for (const el of this.root.querySelectorAll<HTMLElement>('.vf-key, .vf-hint')) {
      if (el.offsetHeight > 0) bottom = Math.max(bottom, el.offsetTop + el.offsetHeight);
    }
    return bottom;
  }

  resetView(): void {
    this.controls.enableDamping = false;
    this.controls.update();
    this.controls.enableDamping = true;
    this.controls.reset();
    this.reframe();
    this.placeCamera();
    this.labelsDirty = true;
    this.requestFrame();
  }

  private applyTheme(): void {
    this.theme = readTheme();
    this.scene.background = new Color(...this.theme.surface).convertSRGBToLinear();
    for (const renderable of this.renderables) {
      renderable.bars?.setTheme(this.theme);
      renderable.nodes?.setTheme(this.theme);
      renderable.plate?.setTheme(this.theme);
    }
    this.lines.setColor(this.theme.ink, this.theme.surface);
    this.braceInk.setColor(this.theme.ink);
    this.highlightMaterial.color.setRGB(...this.theme.ink);
    this.refresh();
  }

  private hit(ev: PointerEvent): { renderable: Renderable; index: number } | undefined {
    const rect = this.renderer.domElement.getBoundingClientRect();
    this.pointer.set(((ev.clientX - rect.left) / rect.width) * 2 - 1, -((ev.clientY - rect.top) / rect.height) * 2 + 1);
    this.raycaster.setFromCamera(this.pointer, this.camera);
    const pickables = this.renderables.filter((r) => this.state.revealed(r.block.node)).flatMap((r) => r.pickables);
    const hits = this.raycaster.intersectObjects(pickables, false);
    if (hits.length === 0) return undefined;
    const object = hits[0].object;
    const block = object.userData.block as Block;
    const renderable = this.renderableOf(block.node)!;
    const index = (object as InstancedMesh).isInstancedMesh ? (hits[0].instanceId ?? 0) : (object.userData.element as number);
    return { renderable, index };
  }

  private edgeAt(x: number, y: number): DrawnEdge | undefined {
    if (this.state.dir !== 'backward') return undefined;
    if (this.screenCached !== this.screenVersion) {
      for (const d of this.edges) {
        d.screen = [];
        for (let i = 0; i < d.points.length; i += 3) {
          tmpVector.set(d.points[i], d.points[i + 1], d.points[i + 2]).project(this.camera);
          d.screen.push(((tmpVector.x + 1) / 2) * this.width, ((1 - tmpVector.y) / 2) * this.height);
        }
      }
      this.screenCached = this.screenVersion;
    }
    let best: DrawnEdge | undefined;
    let bestDistance = EDGE_HOVER_DISTANCE;
    for (const d of this.edges) {
      if (!this.state.revealed(d.fromNode)) continue;
      for (let i = 0; i + 3 < d.screen.length; i += 2) {
        const distance = segmentDistance(x, y, d.screen[i], d.screen[i + 1], d.screen[i + 2], d.screen[i + 3]);
        if (distance < bestDistance) {
          bestDistance = distance;
          best = d;
        }
      }
    }
    return best;
  }

  private setHoverEdge(d: DrawnEdge | undefined): void {
    if (this.hoverEdge === d) return;
    this.hoverEdge = d;
    this.geometryDirty = true;
    this.requestFrame();
  }

  private tipContent(renderable: Renderable, index: number): DocumentFragment {
    const { block } = renderable;
    const node = block.node;
    const element = block.elements[index];
    const backward = this.state.dir === 'backward';
    const elemTex = this.explainer.elemTex({ node, index: element.index });
    const value = this.valuesOf(node)[index];
    const label = this.aligned.get(node);
    const token = label?.tokens?.[element.col];
    const fragment = document.createDocumentFragment();
    const head = document.createElement('div');
    head.className = 'vf-f';
    head.appendChild(render(backward ? `\\partial\\ell/\\partial ${elemTex}` : elemTex));
    if (token !== undefined) head.append(`，${tokenText(token)}`);
    fragment.appendChild(head);
    const line = document.createElement('div');
    line.className = 'vf-m';
    line.textContent = `= ${value === -Infinity ? '−∞' : formatNumber(value, { digits: TIP_DIGITS })}`;
    fragment.appendChild(line);
    const name = document.createElement('div');
    name.className = 'vf-s';
    name.textContent = node.name;
    fragment.appendChild(name);
    const target = this.targetOf(block, element.row);
    if (label !== undefined && target !== element.col && !backward) {
      const note = document.createElement('div');
      note.className = 'vf-s';
      note.textContent = label.tokens === undefined ? '点击设为目标类' : '点击设为目标 token';
      fragment.appendChild(note);
    }
    return fragment;
  }

  private showTip(renderable: Renderable, index: number, ev: PointerEvent): void {
    this.tip.replaceChildren(this.tipContent(renderable, index));
    this.tip.hidden = false;
    const rect = this.root.getBoundingClientRect();
    placeTip(this.root, this.tip, ev.clientX - rect.left, ev.clientY - rect.top);
  }

  private gridOf(node: Node): ValueGrid {
    const [, cols] = rowsCols(node.shape);
    return { label: this.state.dir === 'backward' ? '梯度' : '当前值', values: this.valuesOf(node), cols };
  }

  private showNodeCard(node: Node, el: HTMLElement): void {
    this.pinned = { node, el };
    this.pop.replaceChildren(nodeCardElement(this.explainer.nodeCard(node, this.state.dir), this.gridOf(node)));
    this.pop.hidden = false;
    this.tip.hidden = true;
    const rect = this.root.getBoundingClientRect();
    const anchor = el.getBoundingClientRect();
    placePop(this.root, this.pop, anchor.right - 8 - rect.left, anchor.top - 22 - rect.top, anchor.width);
  }

  private hideNodeCard(): void {
    this.pinned = undefined;
    this.pop.hidden = true;
  }

  private showEdgeCard(d: DrawnEdge, ev: PointerEvent): void {
    this.pop.replaceChildren(edgeCardElement(this.explainer.edgeCard(d.edge), d.edge.id));
    this.pop.hidden = false;
    const rect = this.root.getBoundingClientRect();
    placePop(this.root, this.pop, ev.clientX - rect.left, ev.clientY - rect.top);
  }

  private setTarget(renderable: Renderable, index: number): void {
    const { block } = renderable;
    const label = this.aligned.get(block.node);
    if (label === undefined) return;
    const element = block.elements[index];
    const data = Float64Array.from(label.value.data);
    data[Math.min(element.row, data.length - 1)] = element.col;
    label.set(label.shape.length === 0 ? data[0] : Array.from(data));
    if (block.batched) this.options.steps?.setFocus(element.row);
  }

  private onWheel(ev: WheelEvent): void {
    if (!(ev.ctrlKey || ev.metaKey)) return;
    ev.preventDefault();
    const [lo, hi] = this.zoomRange;
    this.camera.zoom = Math.min(hi, Math.max(lo, this.camera.zoom * Math.exp(-ev.deltaY * WHEEL_ZOOM_RATE)));
    this.camera.updateProjectionMatrix();
    this.goal = undefined;
    this.labelsDirty = true;
    this.requestFrame();
    clearTimeout(this.wheelTimer);
    this.wheelTimer = window.setTimeout(() => this.refresh(), WHEEL_SETTLE_MS);
  }

  private bindPointer(): void {
    const canvas = this.renderer.domElement;
    const onMove = (ev: PointerEvent) => {
      const hit = this.hit(ev);
      const rect = this.root.getBoundingClientRect();
      if (hit === undefined) {
        this.tip.hidden = true;
        const edge = this.edgeAt(ev.clientX - rect.left, ev.clientY - rect.top);
        this.setHoverEdge(edge);
        canvas.classList.toggle('vf-cursor-help', edge !== undefined);
        canvas.classList.remove('vf-cursor-pointer');
        if (edge !== undefined) this.showEdgeCard(edge, ev);
        else if (this.pinned === undefined) this.pop.hidden = true;
        return;
      }
      this.setHoverEdge(undefined);
      if (this.pinned === undefined) this.pop.hidden = true;
      const clickable = this.aligned.has(hit.renderable.block.node) && this.state.dir === 'forward';
      canvas.classList.toggle('vf-cursor-pointer', clickable);
      canvas.classList.remove('vf-cursor-help');
      this.showTip(hit.renderable, hit.index, ev);
    };
    const onLeave = () => {
      this.tip.hidden = true;
      this.setHoverEdge(undefined);
      if (this.pinned === undefined) this.pop.hidden = true;
    };
    const onDown = (ev: PointerEvent) => {
      this.downAt = [ev.clientX, ev.clientY];
    };
    const onUp = (ev: PointerEvent) => {
      if (this.downAt === undefined) return;
      const moved = Math.hypot(ev.clientX - this.downAt[0], ev.clientY - this.downAt[1]);
      this.downAt = undefined;
      if (moved > CLICK_SLOP) return;
      const hit = this.hit(ev);
      if (hit === undefined || !this.aligned.has(hit.renderable.block.node) || this.state.dir !== 'forward') return;
      this.setTarget(hit.renderable, hit.index);
      this.showTip(hit.renderable, hit.index, ev);
    };
    const onWheel = (ev: WheelEvent) => this.onWheel(ev);
    canvas.addEventListener('pointermove', onMove);
    canvas.addEventListener('pointerleave', onLeave);
    canvas.addEventListener('pointerdown', onDown);
    canvas.addEventListener('pointerup', onUp);
    this.root.addEventListener('wheel', onWheel, { passive: false });
    this.unsubscribe.push(() => {
      canvas.removeEventListener('pointermove', onMove);
      canvas.removeEventListener('pointerleave', onLeave);
      canvas.removeEventListener('pointerdown', onDown);
      canvas.removeEventListener('pointerup', onUp);
      this.root.removeEventListener('wheel', onWheel);
    });
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    if (this.frameHandle !== 0) cancelAnimationFrame(this.frameHandle);
    clearTimeout(this.wheelTimer);
    this.resizeObserver.disconnect();
    this.unsubscribe.forEach((fn) => fn());
    this.controls.dispose();
    for (const renderable of this.renderables) {
      renderable.bars?.dispose();
      renderable.nodes?.dispose();
      renderable.plate?.dispose();
    }
    this.lines.dispose();
    this.braceInk.dispose();
    this.highlight?.geometry.dispose();
    this.highlightMaterial.dispose();
    this.labels.dispose();
    this.renderer.dispose();
    this.renderer.forceContextLoss();
    this.root.remove();
  }
}

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
