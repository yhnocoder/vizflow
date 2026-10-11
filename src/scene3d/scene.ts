import {
  AmbientLight,
  BoxGeometry,
  BufferGeometry,
  Color,
  ConeGeometry,
  CubicBezierCurve3,
  DirectionalLight,
  EdgesGeometry,
  Group,
  type InstancedMesh,
  Line,
  LineBasicMaterial,
  LineSegments,
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
import { type Graph, type Node, moduleJacobian } from '../autograd';
import { type Explainer, type FlowEdge, explain, matmulOperands } from '../explain';
import { render, renderText } from '../formula';
import { type StepInfo, type StepModule, type StepState, type Steps, allVisible, isLeafLike } from '../steps';
import { formatNumber, formatShape, offsetOf } from '../tensor';
import { type LinearRgb, type ThemeColors, onThemeChange, readTheme } from '../theme';
import { BarMesh, NodeMesh, PlateMesh, type SeeThrough, barBottom, barTop } from './blocks';
import { type ValueGrid, edgeCardElement, nodeCardElement, placePop, placeTip, valueGridElement } from './cards';
import { barColor, nodeColor, outlineOpacity } from './colors';
import { type EditHost, mountEditor } from './edit';
import { editableLayout } from './editconfig';
import { LABEL_SAFE_MARGIN, type Label, LabelLayer, anchorOffset } from './labels';
import {
  type Block,
  type Element,
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
import { type FoldBox, type LevelLayout, type LevelPlan, BRACE_X, CHIP_LIFT } from './levels';
import { type Brace, type SceneModule, BRACE_OPACITY, BRACE_THICK_OFFSET, blockSpan, braceOffset, bracePoints, braceTip, checkModules, layoutBraces, levelBraces } from './modules';
import { TICK_COUNT, barHeight, capFor, niceTicks, scaleFor } from './scales';
import { type EdgeStyle, edgeStyle } from './style';
import { Transition } from './transition';
import { WAVE_INVERSE, checkFigure, figureTag } from '../ui/figure-spec';
import { renderFigures } from '../ui/figures';

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
}

export interface SceneHandle {
  resetView(): void;
  dispose(): void;
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
export const SEE_THROUGH_DEFAULT: SeeThrough = 'fade';
export const SEE_THROUGH_ORDER: SeeThrough[] = ['off', 'fade', 'wire'];
export const SEE_THROUGH_TEXT: Record<SeeThrough, string> = { off: '遮挡：关', fade: '遮挡：半透明', wire: '遮挡：线框' };
export const SEE_THROUGH_MIN_OVERLAP = 6;
export const SEE_THROUGH_DEPTH_GAP = 0.3;
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
const BOX_MIN_HEIGHT = 0.01;
const BOX_LABEL_OFFSET: [number, number] = [0.2, 0.15];
const BOX_OPACITY = 0.3;
const BOX_EDGE_OPACITY = 0.6;
export const BOX_FOLDED_TEXT = '已折叠';
export const BOX_FIXED_TEXT = '占位';
export const BOX_FIXED_NOTE = '数值按恒等';
export const BOX_CARD_NOTE = '点这个标签可以展开。';
export const BOX_FIXED_CARD_NOTE = '这个盒子只用来示意，数值按恒等映射处理，不能展开。';
export const ARROW_TEXT = '位置顺序';
const ARROW_SIDE = 2.0;
const ARROW_BACK = 0.35;
const ARROW_EXTRA = 0.7;
const ARROW_LIFT = 0.06;
const ARROW_TEXT_OFFSET: [number, number] = [0.2, -0.25];
const ARROW_OPACITY = 0.6;
const ARROW_HEAD: [number, number] = [0.06, 0.2];
const ARROW_HEAD_SEGMENTS = 14;
const ARROW_HEAD_INSET = 0.1;
const SAME_LEVEL = 0.05;
export const BOARD_BOW = -2.6;
export const RESIDUAL_BOW = 1.8;
export const SPLIT_JACOBIAN_BOW = 1.4;
const JACOBIAN_EPSILON = 1e-12;
const LINK_REVEALED = 0.32;
const LINK_HIDDEN = 0.06;
const ROPE_TOLERANCE = 1e-6;
export const JACOBIAN_CARD_NOTE = '折叠后的整体 Jacobian 元素。';
export const LINK_CARD_NOTE = '同一个张量画了两次，这条虚线连接原处与它在下一个模块入口的复制。';

interface Renderable {
  block: Block;
  bars?: BarMesh;
  nodes?: NodeMesh;
  plate?: PlateMesh;
  transition: Transition;
  clipped: boolean[];
  cap: number;
  far: boolean;
  seeThrough: SeeThrough;
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
  bow?: number;
  jac?: { module: StepModule; value: number };
  link?: boolean;
}

interface SceneBox {
  box: FoldBox;
  mesh: Mesh;
  material: MeshBasicMaterial;
  edgeMaterial: LineBasicMaterial;
  label: Label;
}

interface SceneArrow {
  group: Group;
  label: Label;
  setLength(length: number): void;
}

function mixRgb(a: LinearRgb, b: LinearRgb, k: number): LinearRgb {
  return [a[0] + (b[0] - a[0]) * k, a[1] + (b[1] - a[1]) * k, a[2] + (b[2] - a[2]) * k];
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

export function controlPoints(a: Vec3, b: Vec3, flat: boolean, bow?: number): [Vec3, Vec3, Vec3, Vec3] {
  if (flat) return [a, a.map((v, i) => v + (b[i] - v) / 3) as Vec3, a.map((v, i) => v + ((b[i] - v) * 2) / 3) as Vec3, b];
  const d = b[1] - a[1];
  const z = bow ?? (d > LONG_EDGE ? EDGE_BOW : 0);
  if (z !== 0) return [a, [a[0], a[1] + d * BOW_TANGENT, a[2] + z], [b[0], b[1] - d * BOW_TANGENT, b[2] + z], b];
  return [a, [a[0], a[1] + d * CURVE_TANGENT, a[2]], [b[0], b[1] - d * CURVE_TANGENT, b[2]], b];
}

function sampleCurve(a: Vec3, b: Vec3, flat: boolean, bow?: number): number[] {
  const c = controlPoints(a, b, flat, bow);
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

function valueOf(label: Label, v: number): void {
  if (Number.isFinite(v)) label.setText(formatNumber(v));
  else texOf(label, formatNumber(v, { tex: true }));
}

function chip(text: string): HTMLButtonElement {
  const button = document.createElement('button');
  button.className = 'vf-chip';
  button.type = 'button';
  button.textContent = text;
  return button;
}

function ropeKind(node: Node): string | undefined {
  const shape = node.shape;
  if (shape.length < 2 || shape.slice(1, -1).some((size) => size !== 1)) return undefined;
  const T = shape[0];
  const dh = shape[shape.length - 1];
  const at = (t: number, c: number) => node.value.get([t, ...shape.slice(1, -1).map(() => 0), c]);
  return Object.keys(WAVE_INVERSE).find((kind) => {
    if (checkFigure('rope_wave', shape, [kind]) !== undefined) return false;
    const f = kind === 'cos' ? Math.cos : Math.sin;
    const omegas = Array.from({ length: dh }, (_, c) => WAVE_INVERSE[kind](at(1, c)));
    for (let t = 0; t < T; t++) for (let c = 0; c < dh; c++) if (Math.abs(at(t, c) - f(t * omegas[c])) > ROPE_TOLERANCE) return false;
    return true;
  });
}

function texTitle(title: string): string {
  const parts = title.split('$');
  return parts.map((part, i) => (i % 2 === 1 ? part : part.trim() === '' ? '' : `\\text{${part}}`)).join('');
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
  private seeThroughMode: SeeThrough = SEE_THROUGH_DEFAULT;
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
  private levelLayout: LevelLayout | undefined;
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
  private readonly boxes = new Map<string, SceneBox>();
  private readonly boxGeometry = new BoxGeometry(1, 1, 1);
  private readonly boxEdges = new EdgesGeometry(this.boxGeometry);
  private readonly chipLabels = new Map<string, Label[]>();
  private arrow: SceneArrow | undefined;
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

  private get steps(): Steps | undefined {
    return this.options.steps as Steps | undefined;
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
    const state = this.options.steps?.state() ?? allVisible(this.g);
    const views = [...(this.levelLayout?.sourceOf ?? [])];
    if (views.length === 0) return state;
    return {
      dir: state.dir,
      index: state.index,
      kind: state.kind,
      focus: state.focus,
      revealed: (node) => state.revealed(node) || views.some(([view, source]) => source === node && state.revealed(view)),
      active: (node) => state.active(node) || views.some(([view, source]) => source === node && state.active(view)),
    };
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
      const jump = document.createElement('div');
      jump.className = 'vf-jump';
      const toggle = chip(JUMP_PLACEHOLDER);
      toggle.setAttribute('aria-label', '跳到模块');
      toggle.setAttribute('aria-haspopup', 'menu');
      toggle.setAttribute('aria-expanded', 'false');
      const menu = document.createElement('div');
      menu.className = 'vf-jump-menu';
      menu.setAttribute('role', 'menu');
      menu.hidden = true;
      const close = () => {
        menu.hidden = true;
        toggle.setAttribute('aria-expanded', 'false');
      };
      modules.forEach((module) => {
        const item = document.createElement('button');
        item.type = 'button';
        item.className = 'vf-jump-item';
        item.setAttribute('role', 'menuitem');
        renderText(module.title, item);
        item.addEventListener('click', () => {
          close();
          const brace = this.braces.find((b) => b.module === module);
          if (brace === undefined) return;
          this.goal = { y: this.clampFollowY((brace.y0 + brace.y1) / 2, this.halfHeight()), zoom: 1 };
          this.refresh();
        });
        menu.appendChild(item);
      });
      toggle.addEventListener('click', () => {
        menu.hidden = !menu.hidden;
        toggle.setAttribute('aria-expanded', String(!menu.hidden));
      });
      jump.addEventListener('focusout', (ev) => {
        if (!jump.contains(ev.relatedTarget as globalThis.Node | null)) close();
      });
      jump.append(toggle, menu);
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
    if (this.options.steps !== undefined) {
      const see = chip(SEE_THROUGH_TEXT[this.seeThroughMode]);
      see.setAttribute('aria-pressed', String(this.seeThroughMode !== 'off'));
      see.addEventListener('click', () => {
        this.seeThroughMode = SEE_THROUGH_ORDER[(SEE_THROUGH_ORDER.indexOf(this.seeThroughMode) + 1) % SEE_THROUGH_ORDER.length];
        see.textContent = SEE_THROUGH_TEXT[this.seeThroughMode];
        see.setAttribute('aria-pressed', String(this.seeThroughMode !== 'off'));
        this.markSeeThrough();
        this.refresh();
      });
      tools.appendChild(see);
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
    const where = steps.modules.find((m) => m.id === info.module)?.title;
    const key = `${steps.current}:${info.label}:${where ?? ''}`;
    if (this.ovnav.text.dataset.key === key) return;
    this.ovnav.text.dataset.key = key;
    const small = document.createElement('small');
    renderText(`${info.dir === 'forward' ? '前向' : '反向'}${where ? ` · ${where}` : ''}`, small);
    const label = document.createElement('span');
    renderText(info.label, label);
    this.ovnav.text.replaceChildren(small, ...label.childNodes);
  }

  private resolve(): void {
    const steps = this.steps;
    const resolved = resolveLayout(this.options.layout, {
      visible: this.visible,
      aligned: this.aligned,
      targetOf: targetColumn,
      focus: this.state.focus,
      folded: steps === undefined ? undefined : (id) => steps.folded(id),
      plan: this.options.levels,
    });
    this.boards = resolved.boards;
    this.levelLayout = resolved.levels;
    if (this.renderables.length === 0) {
      this.renderables = resolved.blocks.map((block) => this.createRenderable(block));
    } else {
      for (const block of resolved.blocks) {
        const renderable = this.renderables.find((r) => (r.block.key ?? r.block.node) === (block.key ?? block.node));
        if (renderable === undefined) continue;
        const moved = renderable.block.plate?.y !== block.plate?.y;
        renderable.block = block;
        if (moved && renderable.plate !== undefined) {
          this.scene.remove(renderable.plate.group);
          renderable.plate.dispose();
          renderable.plate = new PlateMesh(block, this.theme);
          this.scene.add(renderable.plate.group);
        }
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
      seeThrough: 'off',
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
      const label = this.labels.add(block.role === 'board-b' || text.anchor === 'above' ? 'colh' : 'cls');
      if (block.role === 'planar' && text.anchor === 'below') label.under = renderable.values.filter((_, i) => block.elements[i].col === text.col);
      return label;
    });
    renderable.rows = block.rowLabels.map(() => this.labels.add('kk'));
    if (block.axis !== undefined) renderable.tickLabels = Array.from({ length: Math.max(renderable.ticks.forward.length, renderable.ticks.backward.length) }, () => this.labels.add('tick'));
    return renderable;
  }

  private foldable(module: SceneModule): string | undefined {
    const steps = this.steps;
    if (module.id === undefined || module.fixed === true || steps?.setFolded === undefined || steps.folded === undefined) return undefined;
    if (!steps.modules.some((m) => m.id === module.id)) return undefined;
    return module.id;
  }

  private moduleLabel(module: SceneModule): Label {
    const side = module.side ?? 'left';
    const id = this.foldable(module);
    const label = this.labels.add('mod', `vf-mod-${side === 'outer' ? 'right vf-mod-thick' : side}${id === undefined ? '' : ' vf-foldable'}`);
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
      renderText(module.title, title);
      fragment.appendChild(title);
      if (subtitle !== undefined) {
        const sub = document.createElement('span');
        sub.className = 'vf-mod-sub';
        renderText(subtitle, sub);
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
    const levels = this.levelLayout;
    if (levels !== undefined) {
      const xs = this.renderables.flatMap((r) => r.block.elements.map((e) => e.x));
      const shown = modules.filter((m) => !(this.lod && m.inner === true));
      this.braces = levelBraces(shown, levels.ranges, { xMin: Math.min(...xs), xMax: Math.max(...xs), ...levels.braces }, BRACE_X);
    } else {
      const { min, max } = boundsOf(this.framingPoints(this.renderables));
      this.braces = layoutBraces(
        modules,
        this.renderables.map((r) => blockSpan(r.block)),
        { xMin: min[0], xMax: max[0] },
      );
    }
    modules.forEach((module, i) => {
      const label = this.moduleLabels[i];
      const brace = this.braces.find((b) => b.module === module);
      label.visible = brace !== undefined;
      if (brace === undefined) return;
      this.moduleContent(label, brace.module);
      if (label.pos === undefined) label.pos = braceTip(brace);
    });
  }

  private placeBraces(): void {
    if (this.braces.length === 0) return;
    const view = this.viewDirection();
    const direction: Vec3 = [view.x, view.y, view.z];
    const modules = this.options.modules ?? [];
    const polylines = this.braces.flatMap((brace) => {
      const offset = braceOffset(brace, direction);
      this.moduleLabels[modules.indexOf(brace.module)].pos = braceTip(brace, offset);
      const points = bracePoints(brace, offset);
      const line = { points, opacity: BRACE_OPACITY, dashed: false };
      if (brace.thick !== true) return [line];
      return [line, { ...line, points: points.map((v, i) => (i % 3 === 0 ? v + BRACE_THICK_OFFSET : v)) }];
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
    const source = this.levelLayout?.sourceOf.get(node) ?? node;
    return this.renderables.find((r) => r.block.node === source && r.block.copyOf === undefined);
  }

  private elementOf(block: Block, node: Node, index: readonly number[]): Element {
    return block.node === node ? elementAt(block, index) : block.elements[offsetOf(node.shape, index as number[])];
  }

  private renderableFor(block: Block): Renderable | undefined {
    return this.renderables.find((r) => (r.block.key ?? r.block.node) === (block.key ?? block.node));
  }

  private shown(block: Block | undefined): boolean {
    return block !== undefined && block.hidden !== true;
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
    const stepLevels = this.renderables.filter((r) => r.block.copyOf === undefined && !isLeafLike(r.block.node) && this.state.active(r.block.node)).map(levelOf);
    const enabled = this.follow && this.farOn && !this.lod && stepLevels.length > 0;
    for (const r of this.renderables) {
      const level = levelOf(r);
      r.far = enabled && !this.state.active(r.block.node) && Math.min(...stepLevels.map((s) => Math.abs(s - level))) > FAR_LEVELS;
    }
  }

  private screenBox(renderable: Renderable, direction: Vector3): { x0: number; x1: number; y0: number; y1: number; depth: number } | undefined {
    const { block } = renderable;
    if (block.hidden === true || block.elements.length === 0) return undefined;
    const [w, d] = block.footprint;
    const lo = [Infinity, Infinity, Infinity];
    const hi = [-Infinity, -Infinity, -Infinity];
    const grow = (x: number, y: number, z: number) => {
      lo[0] = Math.min(lo[0], x);
      lo[1] = Math.min(lo[1], y);
      lo[2] = Math.min(lo[2], z);
      hi[0] = Math.max(hi[0], x);
      hi[1] = Math.max(hi[1], y);
      hi[2] = Math.max(hi[2], z);
    };
    const lift = renderable.nodes?.height;
    block.elements.forEach((e, i) => {
      const h = lift ?? renderable.transition.current[i] ?? 0;
      grow(e.x - w / 2, e.y + Math.min(h, 0), e.z - d / 2);
      grow(e.x + w / 2, e.y + Math.max(h, 0), e.z + d / 2);
    });
    const plate = block.plate;
    if (plate !== undefined) {
      grow(plate.x - plate.w / 2, plate.y, plate.z - plate.d / 2);
      grow(plate.x + plate.w / 2, plate.y, plate.z + plate.d / 2);
    }
    const box = { x0: Infinity, x1: -Infinity, y0: Infinity, y1: -Infinity, depth: 0 };
    const corner = new Vector3();
    for (let k = 0; k < 8; k++) {
      corner.set(k & 1 ? hi[0] : lo[0], k & 2 ? hi[1] : lo[1], k & 4 ? hi[2] : lo[2]).project(this.camera);
      const px = ((corner.x + 1) / 2) * this.width;
      const py = ((1 - corner.y) / 2) * this.height;
      box.x0 = Math.min(box.x0, px);
      box.x1 = Math.max(box.x1, px);
      box.y0 = Math.min(box.y0, py);
      box.y1 = Math.max(box.y1, py);
    }
    corner.set((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2).sub(this.camera.position);
    box.depth = corner.dot(direction);
    return box;
  }

  private markSeeThrough(): boolean {
    const mode = this.seeThroughMode;
    const enabled = mode !== 'off' && !this.lod && this.options.steps !== undefined && !(this.state.dir === 'forward' && this.state.index === 0);
    const direction = this.camera.getWorldDirection(new Vector3());
    const boxes = new Map<Renderable, NonNullable<ReturnType<SceneController['screenBox']>>>();
    if (enabled) {
      for (const r of this.renderables) {
        const box = this.screenBox(r, direction);
        if (box !== undefined) boxes.set(r, box);
      }
    }
    const backward = this.state.dir === 'backward';
    const involved = new Set(this.stepNodes);
    for (const d of this.edges) {
      if (!this.stepNodes.has(backward ? d.fromNode : d.toNode)) continue;
      involved.add(d.fromNode);
      involved.add(d.toNode);
    }
    const inStep = (node: Node) => this.state.active(node) || involved.has(node);
    const targets = [...boxes].filter(([r]) => inStep(r.block.node) && this.state.revealed(r.block.node)).map(([, box]) => box);
    let changed = false;
    for (const r of this.renderables) {
      const box = boxes.get(r);
      const covers =
        box !== undefined &&
        !inStep(r.block.node) &&
        targets.some(
          (t) =>
            box.depth < t.depth - SEE_THROUGH_DEPTH_GAP &&
            Math.min(box.x1, t.x1) - Math.max(box.x0, t.x0) > SEE_THROUGH_MIN_OVERLAP &&
            Math.min(box.y1, t.y1) - Math.max(box.y0, t.y0) > SEE_THROUGH_MIN_OVERLAP,
        );
      const next: SeeThrough = covers ? mode : 'off';
      if (next !== r.seeThrough) changed = true;
      r.seeThrough = next;
    }
    return changed;
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
    this.markSeeThrough();
    const stepKey = `${dir}:${this.state.index}:${info?.id ?? ''}`;
    if (this.follow && this.stepKey !== undefined && stepKey !== this.stepKey) {
      this.goal = this.followGoal();
      if (this.reduceMotion.matches) this.applyGoal(1);
    }
    this.stepKey = stepKey;
    const zoom = this.zoomNow();
    for (const renderable of this.renderables) {
      const { block } = renderable;
      const node = block.node;
      const hidden = block.hidden === true;
      const revealed = this.state.revealed(node) && !hidden;
      const active = this.state.active(node);
      const far = renderable.far;
      const clear = renderable.seeThrough === 'off';
      const values = this.valuesOf(node);
      const scale = renderable.scale[dir];
      for (const group of [renderable.bars?.group, renderable.nodes?.group, renderable.plate?.group]) if (group !== undefined) group.visible = !hidden;
      const heights = new Float64Array(block.elements.length);
      for (let i = 0; i < heights.length; i++) {
        const { height, clipped } = barHeight(values[i], scale, revealed, renderable.cap);
        heights[i] = height;
        renderable.clipped[i] = revealed && clipped;
      }
      renderable.transition.retarget(heights, now);
      const focusedRow = (row: number) => !block.batched || row === this.state.focus;
      const front = Math.max(...block.elements.map((e) => e.z));
      const stepped = block.copyOf === undefined || backward ? this.stepNodes.has(node) : active && !this.stepNodes.has(node);
      const showValues = revealed && clear && stepped && block.quiet !== true && !this.lod && (!this.follow || zoom >= VALUE_MIN_ZOOM);
      const named = revealed && clear && !this.lod && !far;
      const title = renderable.title!;
      title.el.dataset.node = node.name;
      title.offset = this.options.labels?.[node.name];
      title.visible = named && block.untitled !== true;
      const view = block.copyOf === undefined ? [...(this.levelLayout?.sourceOf ?? [])].find(([, source]) => source === node)?.[0] : undefined;
      const continued = this.levelLayout?.boards.some((b) => b.op === node && b.previous !== undefined) === true;
      const tex = view === undefined ? node.tex : continued ? view.tex : `${view.tex}=${node.tex}`;
      texOf(title, backward ? `\\partial\\ell/\\partial ${(view ?? node).tex}` : tex);
      if (renderable.bars !== undefined) {
        title.pos = [block.title.x, block.title.y, block.title.z];
        title.anchor = block.title.anchor === 'left' ? 'right-middle' : block.title.anchor === 'right' ? 'left-middle' : block.title.anchor === 'below' ? 'bottom-center' : 'top-center';
        renderable.values.forEach((label, i) => {
          const element = block.elements[i];
          valueOf(label, values[i]);
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
          valueOf(label, values[i]);
          label.el.dataset.node = node.name;
          label.el.dataset.index = element.index.join(',');
          label.pos = [element.x + NODE_VALUE_OFFSET, element.y + mesh.height / 2, element.z];
          label.visible = showValues && focusedRow(element.row);
        });
        const colors = block.elements.map((element, i) => nodeColor(values[i], this.theme, { dir, revealed, active, far, focused: focusedRow(element.row) }));
        mesh.update(block.elements, colors, block.elements.map(() => revealed), renderable.seeThrough);
      }
      renderable.classes.forEach((label, i) => {
        const text = block.classLabels[i];
        const target = this.targetOf(block, block.rows - 1);
        const isTarget = this.aligned.has(node) && !block.batched && block.rows === 1 && target === text.col;
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
        const table = block.table;
        const frameXs = block.frameRows.map((_, r) => {
          if (table !== undefined) return block.plate!.x;
          const target = this.targetOf(block, r);
          const column = block.elements.find((e) => e.col === (target ?? 0));
          return column?.x ?? block.elements[0].x;
        });
        const frameZs = table === undefined ? undefined : Array.from(table.label.value.data, (id) => block.elements.find((e) => e.row === Math.round(id))?.z ?? block.elements[0].z);
        renderable.plate.update({ revealed, far, seeThrough: renderable.seeThrough }, frameXs, ticks, scale, frameZs);
      }
    }
    this.updateBraces();
    this.updateBoxes();
    this.updateChips();
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

  private anchorOut(node: Node, index: number[], renderable = this.renderableOf(node)): Vec3 | undefined {
    if (renderable === undefined) return undefined;
    const element = this.elementOf(renderable.block, node, index);
    if (renderable.nodes !== undefined) return renderable.nodes.anchorOut(element);
    return barTop(element, renderable.transition.current[offsetOf(node.shape, index)]);
  }

  private anchorIn(node: Node, index: number[], renderable = this.renderableOf(node)): Vec3 | undefined {
    if (renderable === undefined) return undefined;
    const element = this.elementOf(renderable.block, node, index);
    if (renderable.nodes !== undefined) return renderable.nodes.anchorIn(element);
    return barBottom(element, renderable.transition.current[offsetOf(node.shape, index)]);
  }

  private updateBoxes(): void {
    const shown = new Set<string>();
    for (const box of this.levelLayout?.boxes ?? []) {
      const id = box.module.id;
      shown.add(id);
      const entry = this.boxes.get(id) ?? this.createBox(box);
      entry.box = box;
      const [x0, x1] = box.x;
      const [y0, y1] = box.y;
      const [z0, z1] = box.z;
      entry.mesh.scale.set(x1 - x0, Math.max(BOX_MIN_HEIGHT, y1 - y0), z1 - z0);
      entry.mesh.position.set((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2);
      entry.mesh.visible = true;
      entry.material.color.setRGB(...this.theme.bg);
      entry.edgeMaterial.color.setRGB(...this.theme.ink);
      entry.label.pos = [x0 + BOX_LABEL_OFFSET[0], y0 + BOX_LABEL_OFFSET[1], z1];
      entry.label.visible = true;
    }
    for (const [id, entry] of this.boxes) {
      if (shown.has(id)) continue;
      entry.mesh.visible = false;
      entry.label.visible = false;
    }
  }

  private createBox(box: FoldBox): SceneBox {
    const material = new MeshBasicMaterial({ transparent: true, opacity: BOX_OPACITY, depthWrite: false });
    const edgeMaterial = new LineBasicMaterial({ transparent: true, opacity: BOX_EDGE_OPACITY });
    const mesh = new Mesh(this.boxGeometry, material);
    mesh.add(new LineSegments(this.boxEdges, edgeMaterial));
    this.scene.add(mesh);
    const module = box.module;
    const label = this.labels.add('boxlb', module.fixed ? 'vf-fixed' : '');
    label.anchor = 'above-right';
    label.setContent(module.title, () => {
      const fragment = document.createDocumentFragment();
      const title = document.createElement('span');
      renderText(`${module.title} · ${module.fixed ? BOX_FIXED_TEXT : BOX_FOLDED_TEXT}`, title);
      fragment.appendChild(title);
      if (module.fixed) {
        const small = document.createElement('small');
        small.textContent = BOX_FIXED_NOTE;
        fragment.appendChild(small);
      }
      return fragment;
    });
    const entry: SceneBox = { box, mesh, material, edgeMaterial, label };
    label.el.addEventListener('pointerenter', () => this.showBoxCard(entry));
    label.el.addEventListener('pointerleave', () => this.hideNodeCard());
    if (!module.fixed) label.el.addEventListener('click', () => this.steps?.setFolded(module.id, false));
    this.boxes.set(module.id, entry);
    return entry;
  }

  private showBoxCard(entry: SceneBox): void {
    const module = entry.box.module;
    const io = this.levelLayout?.io.find((p) => p.module.id === module.id);
    const fragment = document.createDocumentFragment();
    const title = document.createElement('h4');
    title.className = 'vf-pop-title';
    renderText(`${module.title}（${module.fixed ? BOX_FIXED_TEXT : BOX_FOLDED_TEXT}）`, title);
    fragment.appendChild(title);
    if (io !== undefined) {
      const J = moduleJacobian(this.g, module, io.input, io.output);
      const lab = document.createElement('div');
      lab.className = 'vf-lab';
      lab.append(render(`\\partial ${io.output.tex}/\\partial ${io.input.tex}`), '，', render(`${J.length}\\times ${J[0]?.length ?? 0}`));
      fragment.append(lab, valueGridElement({ label: '', values: J.flat(), cols: J[0]?.length ?? 1 }));
      const dl = document.createElement('dl');
      dl.className = 'vf-rows';
      const row = (name: string, build: (dd: HTMLElement) => void) => {
        const dt = document.createElement('dt');
        dt.textContent = name;
        const dd = document.createElement('dd');
        build(dd);
        dl.append(dt, dd);
      };
      row('输入 → 输出', (dd) => dd.append(render(`${io.input.tex}\\;${formatShape(io.input.shape).replace(/×/g, '\\times ')}\\to ${io.output.tex}`)));
      const params = module.params.filter((n) => n.kind === 'param');
      row('参数', (dd) => {
        if (params.length === 0) dd.textContent = '无';
        else dd.append(render(params.map((n) => n.tex).join(',\\ ')));
      });
      fragment.appendChild(dl);
    }
    const note = document.createElement('p');
    note.className = 'vf-nt';
    note.textContent = module.fixed ? BOX_FIXED_CARD_NOTE : BOX_CARD_NOTE;
    fragment.appendChild(note);
    this.pinned = undefined;
    this.pop.replaceChildren(fragment);
    this.pop.hidden = false;
    this.tip.hidden = true;
    const rect = this.root.getBoundingClientRect();
    const anchor = entry.label.el.getBoundingClientRect();
    placePop(this.root, this.pop, anchor.right - 8 - rect.left, anchor.top - 22 - rect.top, anchor.width);
  }

  private updateChips(): void {
    const chips = this.levelLayout?.chips ?? [];
    let arrowShown = false;
    for (const chip of chips) {
      const renderable = this.renderableFor(chip.block);
      if (renderable === undefined) continue;
      const block = chip.block;
      const rows = block.rows;
      let labels = this.chipLabels.get(block.key ?? block.node.name + chip.label.name);
      if (labels === undefined) {
        labels = Array.from({ length: rows }, () => this.labels.add('tok'));
        this.chipLabels.set(block.key ?? block.node.name + chip.label.name, labels);
      }
      const y = block.elements[0].y;
      const zs = Array.from({ length: rows }, (_, r) => block.elements.find((e) => e.row === r)!.z);
      const shown = !this.lod && !renderable.far && block.hidden !== true && (chip.arrow || this.state.revealed(block.node));
      labels.forEach((label, t) => {
        const id = Math.round(chip.label.value.data[Math.min(t, chip.label.value.size - 1)]);
        const token = chip.label.tokens?.[id] ?? String(id);
        const elem = this.explainer.elemTex({ node: chip.label, index: [t] });
        label.setContent(`${elem}|${token}`, () => {
          const fragment = document.createDocumentFragment();
          const small = document.createElement('small');
          small.appendChild(render(elem));
          fragment.append(small, tokenText(token));
          return fragment;
        });
        label.pos = [chip.x, y + CHIP_LIFT, zs[t]];
        label.visible = shown;
      });
      if (chip.arrow) {
        const arrow = this.arrow ?? this.createArrow();
        const z0 = zs[0] - ARROW_BACK;
        const length = zs[rows - 1] - zs[0] + ARROW_EXTRA;
        arrow.group.position.set(chip.x + ARROW_SIDE, y + ARROW_LIFT, z0);
        arrow.setLength(length);
        const pointing = block.hidden !== true;
        arrow.group.visible = pointing;
        arrow.label.pos = [chip.x + ARROW_SIDE + ARROW_TEXT_OFFSET[0], y + CHIP_LIFT, z0 + ARROW_TEXT_OFFSET[1]];
        arrow.label.visible = pointing;
        arrowShown = true;
      }
    }
    if (!arrowShown && this.arrow !== undefined) {
      this.arrow.group.visible = false;
      this.arrow.label.visible = false;
    }
  }

  private createArrow(): SceneArrow {
    const group = new Group();
    const shaftMaterial = new LineBasicMaterial({ color: new Color(...this.theme.ink), transparent: true, opacity: ARROW_OPACITY });
    const shaft = new Line(new BufferGeometry().setFromPoints([new Vector3(0, 0, 0), new Vector3(0, 0, 1)]), shaftMaterial);
    const headMaterial = new MeshBasicMaterial({ color: new Color(...mixRgb(this.theme.surface, this.theme.ink, ARROW_OPACITY)) });
    const head = new Mesh(new ConeGeometry(ARROW_HEAD[0], ARROW_HEAD[1], ARROW_HEAD_SEGMENTS), headMaterial);
    head.rotation.x = Math.PI / 2;
    group.add(shaft, head);
    this.scene.add(group);
    const label = this.labels.add('kk', 'vf-arrow-tag');
    label.anchor = 'top-center';
    label.setText(ARROW_TEXT);
    const arrow: SceneArrow = {
      group,
      label,
      setLength: (length) => {
        shaft.geometry.setFromPoints([new Vector3(0, 0, 0), new Vector3(0, 0, length - ARROW_HEAD[1] + ARROW_HEAD_INSET)]);
        head.position.z = length - ARROW_HEAD_INSET;
      },
    };
    this.arrow = arrow;
    return arrow;
  }

  private buildEdges(): void {
    const drawn: DrawnEdge[] = [];
    this.edges = drawn;
    if (this.lod) return;
    const levels = this.levelLayout;
    const backward = this.state.dir === 'backward';
    const overview = this.state.dir === 'forward' && this.state.index === 0;
    const rowOf = (renderable: Renderable, index: number[]) => (renderable.block.batched ? elementAt(renderable.block, index).row : undefined);
    const push = (entry: Omit<DrawnEdge, 'points' | 'style' | 'screen'>, from: Renderable, to: Renderable, rows: (number | undefined)[], zeroGrad: boolean) => {
      const style = edgeStyle({
        dir: this.state.dir,
        overview,
        contract: entry.contract,
        zeroGrad,
        focused: rows.every((row) => row === undefined || row === this.state.focus),
        far: from.far && to.far,
        current: this.stepNodes.has(backward ? entry.fromNode : entry.toNode),
        revealedFrom: this.state.revealed(entry.fromNode),
        revealedTo: this.state.revealed(entry.toNode),
      });
      if (style !== null) drawn.push({ ...entry, points: [], style, screen: [] });
    };
    for (const edge of this.explainer.edges()) {
      if ('cell' in edge.from || 'cell' in edge.to) {
        const board = this.boards.find((b) => b.op === edge.op);
        if (board === undefined || edge.arg === 1) continue;
        const aR = this.renderableFor(board.a);
        const bR = this.renderableFor(board.b);
        const oR = this.renderableFor(board.out);
        if (aR === undefined || bR === undefined || oR === undefined || !this.shown(board.out) || !this.shown(board.b) || !this.shown(board.a)) continue;
        const index = 'cell' in edge.to ? edge.to.cell.index : (edge.from as { cell: { index: [number, number, number] } }).cell.index;
        const [r, k, i] = index;
        const operands = matmulOperands(edge.op, index);
        const y = board.y + LINE_HEIGHT;
        const tile = (block: Block, kk: number, ii: number) => block.elements.find((e) => e.row === kk && e.col === ii)!;
        const [tileW, tileD] = board.b.footprint;
        const outElement = elementAt(board.out, operands.out);
        const rows = [board.out.batched ? r : undefined];
        let entry: Omit<DrawnEdge, 'points' | 'style' | 'screen'>;
        if (edge.arg === 0) {
          const aElement = this.elementOf(board.a, edge.op.inputs[0], operands.a);
          const to: Vec3 = [tile(board.b, k, i).x - tileW / 2 - 0.01, y, tile(board.b, k, i).z];
          const previous = board.previous;
          if (i === 0 && previous !== undefined) {
            const last = tile(previous, k, previous.cols - 1);
            const from: Vec3 = [last.x + previous.footprint[0] / 2 + 0.01, y, last.z];
            entry = { edge, fromNode: board.a.node, toNode: edge.op, contract: true, endpoints: () => [from, to], flat: true };
          } else if (i === 0 && board.sharedA) {
            entry = { edge, fromNode: board.a.node, toNode: edge.op, contract: true, endpoints: () => [this.anchorOut(board.a.node, aElement.index, aR)!, to], flat: false };
          } else {
            const from: Vec3 = i === 0 ? [aElement.x, y, aElement.z - board.a.footprint[1] / 2 - 0.02] : [tile(board.b, k, i - 1).x + tileW / 2 + 0.01, y, tile(board.b, k, i - 1).z];
            entry = { edge, fromNode: board.a.node, toNode: edge.op, contract: true, endpoints: () => [from, to], flat: true };
          }
        } else {
          const last = k === board.b.rows - 1;
          const from: Vec3 = [tile(board.b, k, i).x, y, tile(board.b, k, i).z + tileD / 2 + 0.01];
          const to: Vec3 = last ? [outElement.x, y, outElement.z - board.out.footprint[1] / 2 - 0.02] : [tile(board.b, k + 1, i).x, y, tile(board.b, k + 1, i).z - tileD / 2 - 0.01];
          entry = { edge, fromNode: board.b.node, toNode: edge.op, contract: true, endpoints: () => [from, to], flat: true };
        }
        push(entry, aR, oR, rows, this.explainer.zeroGrad(edge));
        continue;
      }
      const from = edge.from;
      const to = edge.to;
      const copy = levels?.copyFor(from.node, to.node);
      const fromR = copy === undefined ? this.renderableOf(from.node) : this.renderableFor(copy);
      const toR = this.renderableOf(to.node);
      if (fromR === undefined || toR === undefined || fromR === toR || !this.shown(fromR.block) || !this.shown(toR.block)) continue;
      const rows = [rowOf(fromR, from.index), rowOf(toR, to.index)];
      const fromElement = this.elementOf(fromR.block, from.node, from.index);
      const toElement = this.elementOf(toR.block, to.node, to.index);
      let entry: Omit<DrawnEdge, 'points' | 'style' | 'screen'>;
      if (fromR.block.table !== undefined && Math.abs(fromElement.y - toElement.y) < SAME_LEVEL) {
        const y = toElement.y + LINE_HEIGHT;
        const a: Vec3 = [fromElement.x, y, fromElement.z + fromR.block.footprint[1] / 2];
        const b: Vec3 = [toElement.x, y, toElement.z - toR.block.footprint[1] / 2];
        entry = { edge, fromNode: from.node, toNode: to.node, contract: false, endpoints: () => [a, b], flat: true };
      } else {
        const long = toElement.y - fromElement.y > LONG_EDGE;
        const bow = toR.block.role === 'board-b' && long ? BOARD_BOW : to.node.kind === 'op' && to.node.op === 'add' && long ? RESIDUAL_BOW : undefined;
        entry = {
          edge,
          fromNode: from.node,
          toNode: to.node,
          contract: false,
          endpoints: () => [this.anchorOut(from.node, from.index, fromR)!, this.anchorIn(to.node, to.index, toR)!],
          flat: false,
          bow,
        };
      }
      push(entry, fromR, toR, rows, this.explainer.zeroGrad(edge));
    }
    if (levels === undefined) return;
    for (const pair of levels.io) {
      const inR = this.renderableOf(pair.input);
      const outR = this.renderableFor(pair.outputBlock);
      if (inR === undefined || outR === undefined || !this.shown(inR.block) || !this.shown(outR.block)) continue;
      const J = moduleJacobian(this.g, pair.module, pair.input, pair.output);
      const grad = pair.output.grad;
      J.forEach((row, oi) => {
        row.forEach((value, ii) => {
          if (Math.abs(value) <= JACOBIAN_EPSILON) return;
          const fromElement = inR.block.elements[ii];
          const toElement = outR.block.elements[oi];
          const edge: FlowEdge = { id: `jac:${pair.module.id}:${pair.input.id}:${pair.output.id}:${oi}:${ii}`, op: pair.output, arg: 0, from: { node: pair.input, index: fromElement.index }, to: { node: pair.output, index: toElement.index } };
          const entry = {
            edge,
            fromNode: pair.input,
            toNode: pair.output,
            contract: true,
            endpoints: (): [Vec3, Vec3] => [this.anchorOut(pair.input, fromElement.index, inR)!, this.anchorIn(pair.output, toElement.index, outR)!],
            flat: false,
            bow: pair.split ? SPLIT_JACOBIAN_BOW : 0,
            jac: { module: pair.module, value },
          };
          push(entry, inR, outR, [rowOf(inR, fromElement.index), rowOf(outR, toElement.index)], (grad === null ? 0 : grad.data[oi]) === 0);
        });
      });
    }
    for (const link of levels.links) {
      const origR = this.renderableFor(link.original);
      const copyR = this.renderableFor(link.copy);
      if (origR === undefined || copyR === undefined || !this.shown(link.original) || !this.shown(link.copy) || (origR.far && copyR.far)) continue;
      const node = link.original.node;
      const opacity = this.state.revealed(node) ? LINK_REVEALED : LINK_HIDDEN;
      link.copy.elements.forEach((element) => {
        const edge: FlowEdge = { id: `link:${link.copy.key}:${element.index.join(',')}`, op: node, arg: 0, from: { node, index: element.index }, to: { node, index: element.index } };
        drawn.push({
          edge,
          fromNode: node,
          toNode: node,
          contract: false,
          endpoints: () => [this.anchorOut(node, element.index, origR)!, this.anchorIn(node, element.index, copyR)!],
          flat: false,
          link: true,
          points: [],
          style: { opacity, dashed: true },
          screen: [],
        });
      });
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
      renderable.bars.update(block.elements, { heights, clipped: renderable.clipped, colors, outlineOpacity: outline, seeThrough: renderable.seeThrough });
    }
    for (const d of this.edges) {
      const [a, b] = d.endpoints();
      d.points = sampleCurve(a, b, d.flat, d.bow);
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
    const [p0, p1, p2, p3] = controlPoints(a, b, d.flat, d.bow).map((p) => new Vector3(...p));
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
    if ((moved || stepped || following || this.labelsDirty) && this.markSeeThrough()) this.refresh();
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
    if (this.levelLayout !== undefined) return { lo: this.levelLayout.lo, top: this.levelLayout.top };
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
    const keys = this.lod ? [] : this.renderables.filter((r) => r.block.copyOf === undefined && this.state.active(r.block.node));
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
    const pickables = this.renderables.filter((r) => this.state.revealed(r.block.node) && this.shown(r.block)).flatMap((r) => r.pickables);
    const hits = this.raycaster.intersectObjects(pickables, false);
    if (hits.length === 0) return undefined;
    const object = hits[0].object;
    const block = object.userData.block as Block;
    const renderable = this.renderableFor(block)!;
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
    line.appendChild(render(`= ${value === -Infinity ? '-\\infty' : formatNumber(value, { digits: TIP_DIGITS, tex: true })}`));
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
    return { label: this.state.dir === 'backward' ? '梯度' : '当前值', values: this.valuesOf(node), cols, heads: node.shape.length === 3 ? node.shape[1] : undefined };
  }

  private showNodeCard(node: Node, el: HTMLElement): void {
    this.pinned = { node, el };
    const card = nodeCardElement(this.explainer.nodeCard(node, this.state.dir), this.gridOf(node));
    const wave = this.state.dir === 'forward' && node.kind === 'buffer' ? ropeKind(node) : undefined;
    if (wave !== undefined) {
      const figure = document.createElement('div');
      figure.className = 'vf-pop-figure';
      figure.innerHTML = figureTag('rope_wave', node.name, [wave]);
      renderFigures(figure, this.g);
      card.querySelector('.vf-eqs')?.after(figure);
    }
    this.pop.replaceChildren(card);
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
    this.pop.replaceChildren(d.link === true ? this.linkCard(d) : d.jac !== undefined ? this.jacobianCard(d, d.jac) : edgeCardElement(this.explainer.edgeCard(d.edge), d.edge.id));
    this.pop.hidden = false;
    const rect = this.root.getBoundingClientRect();
    placePop(this.root, this.pop, ev.clientX - rect.left, ev.clientY - rect.top);
  }

  private jacobianCard(d: DrawnEdge, jac: { module: StepModule; value: number }): DocumentFragment {
    const { from, to } = d.edge;
    if ('cell' in from || 'cell' in to) return document.createDocumentFragment();
    const outTex = this.explainer.elemTex({ node: to.node, index: to.index });
    const inTex = this.explainer.elemTex({ node: from.node, index: from.index });
    const grad = to.node.grad;
    const upstream = grad === null ? 0 : grad.get(to.index);
    const i = offsetOf(to.node.shape, to.index) + 1;
    const j = offsetOf(from.node.shape, from.index) + 1;
    const card = {
      title: `${inTex}\\to ${outTex}`,
      local: { tex: `\\big(J_{${texTitle(jac.module.title)}}\\big)_{${i},${j}}=\\frac{\\partial ${outTex}}{\\partial ${inTex}}`, value: jac.value },
      upstream: { tex: `\\frac{\\partial\\ell}{\\partial ${outTex}}`, value: upstream },
      product: { tex: `\\frac{\\partial ${outTex}}{\\partial ${inTex}}\\cdot\\frac{\\partial\\ell}{\\partial ${outTex}}`, value: jac.value * upstream },
      note: JACOBIAN_CARD_NOTE,
    };
    return edgeCardElement(card, d.edge.id);
  }

  private linkCard(d: DrawnEdge): DocumentFragment {
    const fragment = document.createDocumentFragment();
    const title = document.createElement('h4');
    title.className = 'vf-pop-title';
    title.dataset.edgeId = d.edge.id;
    title.appendChild(render(d.fromNode.tex));
    const note = document.createElement('p');
    note.className = 'vf-nt';
    note.textContent = LINK_CARD_NOTE;
    fragment.append(title, note);
    return fragment;
  }

  private setTarget(renderable: Renderable, index: number): void {
    const { block } = renderable;
    const label = this.aligned.get(block.node);
    if (label === undefined) return;
    const element = block.elements[index];
    const data = Float64Array.from(label.value.data);
    const row = Math.min(element.row, data.length - 1);
    const sequence = this.options.sequence;
    const ids = sequence?.get();
    const shift = ids === undefined ? -1 : ids.findIndex((_, k) => k + data.length <= ids.length && data.every((id, i) => Math.round(id) === ids[i + k]));
    if (sequence !== undefined && ids !== undefined && shift >= 0) {
      ids[row + shift] = element.col;
      sequence.set(ids);
    } else {
      data[row] = element.col;
      label.set(label.shape.length === 0 ? data[0] : Array.from(data));
    }
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
