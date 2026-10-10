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
import { type Explainer, type FlowEdge, explain, explainOf } from '../explain';
import { render } from '../formula';
import { type StepState, type Steps, allVisible } from '../steps';
import { defaultDigits, formatNumber, offsetOf } from '../tensor';
import { type LinearRgb, type ThemeColors, onThemeChange, readTheme } from '../theme';
import { BarMesh, NodeMesh, PlateMesh, barBottom, barTop } from './blocks';
import { type EditHost, mountEditor } from './edit';
import { editableLayout } from './editconfig';
import { edgeCardElement, nodeCardElement, placePop, placeTip } from './cards';
import { barColor, nodeColor, outlineOpacity } from './colors';
import { LABEL_SAFE_MARGIN, type Label, LabelLayer, anchorOffset } from './labels';
import {
  type Block,
  type BoardGeometry,
  DISC_RADIUS,
  type Element,
  LINE_HEIGHT,
  type LayoutItem,
  type Vec3,
  classAlignment,
  elementAt,
  resolveLayout,
  targetColumn,
  tokenText,
} from './layout';
import { LineBatch } from './lines';
import { HEIGHT_CAP, barHeight, niceTicks, scaleFor } from './scales';
import { type EdgeStyle, edgeStyle } from './style';
import { Transition } from './transition';

export interface CameraOptions {
  direction?: Vec3;
  azimuth?: [number, number];
  polar?: [number, number];
  zoom?: [number, number];
}

export interface SceneOptions {
  steps?: Steps;
  layout: LayoutItem[];
  edit?: boolean;
  camera?: CameraOptions;
  hint?: string;
  labels?: Record<string, [number, number]>;
}

export interface SceneHandle {
  resetView(): void;
  dispose(): void;
}

export const DEFAULT_DIRECTION: Vec3 = [0.42, 0.78, 1];
export const DEFAULT_AZIMUTH: [number, number] = [-0.8, 0.7];
export const DEFAULT_POLAR: [number, number] = [0.3, 1.45];
export const DEFAULT_ZOOM: [number, number] = [0.6, 3];
export const DAMPING = 0.1;
export const PIXEL_RATIO_CAP = 2;
const FRAME_SEARCH_STEPS = 30;
export const EDGE_HOVER_DISTANCE = 6;
export const NO_WEBGL_TEXT = '这个浏览器启用不了 WebGL，立体图显示不出来。下方的分步讲解和右侧数值仍然可用。';

const CAMERA_DISTANCE = 60;
const NARROW_WIDTH = 600;
const SIDE_LABEL_GAP = 0.15;
const AMBIENT_INTENSITY = 0.72;
const DIRECTIONAL_INTENSITY = 0.42;
const LEGACY_LIGHT_FACTOR = Math.PI;
const LIGHT_POSITION: Vec3 = [-4, 12, 8];
const CURVE_SAMPLES = 24;
const CURVE_TANGENT = 0.45;
const CURVE_MIN_TANGENT = 0.5;
const CLICK_SLOP = 5;
const VALUE_LABEL_GAP = 0.12;
const HIGHLIGHT_RADIUS = 0.035;
const TIP_TERMS = 6;

interface Renderable {
  block: Block;
  bars?: BarMesh;
  nodes?: NodeMesh;
  plate?: PlateMesh;
  transition: Transition;
  clipped: boolean[];
  scale: { forward: number; backward: number };
  ticks: { forward: number[]; backward: number[] };
  title: Label | undefined;
  nodeLabels: Label[];
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
  toLoss: boolean;
  zeroGrad: boolean;
  focused: boolean;
  endpoints: () => [Vec3, Vec3];
  flat: boolean;
  points: number[];
  style: EdgeStyle;
  screen: number[];
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

function span(items: FrameItem[], k: number, at: (i: FrameItem) => number, before: (i: FrameItem) => number, after: (i: FrameItem) => number): { lo: number; hi: number; size: number } {
  let lo = Infinity;
  let hi = -Infinity;
  for (const item of items) {
    lo = Math.min(lo, at(item) * k - before(item));
    hi = Math.max(hi, at(item) * k + after(item));
  }
  return { lo, hi, size: hi - lo };
}

function tangentLength(a: Vec3, b: Vec3): number {
  return Math.max(Math.abs(b[1] - a[1]) * CURVE_TANGENT, CURVE_MIN_TANGENT);
}

function sampleCurve(a: Vec3, b: Vec3, flat: boolean): number[] {
  if (flat) return [...a, ...b];
  const t = tangentLength(a, b);
  const c1: Vec3 = [a[0], a[1] + t, a[2]];
  const c2: Vec3 = [b[0], b[1] - t, b[2]];
  const points: number[] = [];
  for (let s = 0; s < CURVE_SAMPLES; s++) {
    const t = s / (CURVE_SAMPLES - 1);
    const r = 1 - t;
    const w = [r * r * r, 3 * r * r * t, 3 * r * t * t, t * t * t];
    for (let axis = 0; axis < 3; axis++) points.push(w[0] * a[axis] + w[1] * c1[axis] + w[2] * c2[axis] + w[3] * b[axis]);
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

function parenthesized(v: number): string {
  return v < 0 ? `(${formatNumber(v)})` : formatNumber(v);
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
  private disposed = false;
  private narrow = false;

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
    this.state = this.currentState();
    renderer.setPixelRatio(Math.min(devicePixelRatio, PIXEL_RATIO_CAP));
    renderer.outputColorSpace = SRGBColorSpace;
    root.appendChild(renderer.domElement);
    this.labels = new LabelLayer(root, () => {
      this.frameAll();
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
    this.controls.enablePan = false;
    const [minZoom, maxZoom] = cameraOptions.zoom ?? DEFAULT_ZOOM;
    this.controls.minZoom = minZoom;
    this.controls.maxZoom = maxZoom;
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
    this.scene.background = new Color(...this.theme.surface);
    this.lines = new LineBatch(this.theme.ink);
    this.scene.add(this.lines.object);
    this.highlightMaterial = new MeshBasicMaterial({ color: new Color(...this.theme.ink) });

    this.buildBlocks();
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

  setDirection(direction: Vec3): void {
    this.direction.set(...direction).normalize();
    this.limitAzimuth();
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

  private buildChrome(): void {
    const key = document.createElement('div');
    key.className = 'vf-key';
    key.innerHTML =
      '<span><i class="vf-sw" style="background:var(--pos)"></i>正</span><span><i class="vf-sw" style="background:var(--neg)"></i>负</span><span>墨线框 = 目标</span>';
    this.root.appendChild(key);
    if (this.options.hint !== undefined) {
      const hint = document.createElement('div');
      hint.className = 'vf-hint';
      hint.textContent = this.options.hint;
      this.root.appendChild(hint);
    }
    const tools = document.createElement('div');
    tools.className = 'vf-stage-tools';
    const reset = document.createElement('button');
    reset.className = 'vf-chip';
    reset.type = 'button';
    reset.textContent = '复位视角';
    reset.addEventListener('click', () => this.resetView());
    tools.appendChild(reset);
    this.root.appendChild(tools);
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
    const scale = {
      forward: block.scales.forward ?? scaleFor(node.value.data),
      backward: block.scales.backward ?? scaleFor(grad === null ? [] : grad.data),
    };
    const renderable: Renderable = {
      block,
      transition: new Transition(block.elements.length, () => this.reduceMotion.matches),
      clipped: block.elements.map(() => false),
      scale,
      ticks: {
        forward: block.axis === undefined ? [] : niceTicks(node.value.data, scale.forward),
        backward: block.axis === undefined || grad === null ? [] : niceTicks(grad.data, scale.backward),
      },
      title: undefined,
      nodeLabels: [],
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
      this.bindCard(renderable.title, node);
    } else {
      renderable.nodes = new NodeMesh(block, this.theme);
      renderable.pickables.push(...renderable.nodes.meshes);
      renderable.nodeLabels = block.elements.map(() => {
        const label = this.labels.add('node');
        this.bindCard(label, node);
        return label;
      });
    }
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

  private refresh(): void {
    if (this.disposed) return;
    this.narrow = this.width <= NARROW_WIDTH;
    this.state = this.currentState();
    this.resolve();
    const now = performance.now();
    const dir = this.state.dir;
    const backward = dir === 'backward';
    for (const renderable of this.renderables) {
      const { block } = renderable;
      const node = block.node;
      const revealed = this.state.revealed(node);
      const active = this.state.active(node);
      const values = this.valuesOf(node);
      const scale = renderable.scale[dir];
      const heights = new Float64Array(block.elements.length);
      for (let i = 0; i < heights.length; i++) {
        const { height, clipped } = barHeight(values[i], scale, revealed);
        heights[i] = height;
        renderable.clipped[i] = revealed && clipped;
      }
      renderable.transition.retarget(heights, now);
      const focusedRow = (row: number) => !block.batched || row === this.state.focus;
      if (renderable.bars !== undefined) {
        renderable.title!.pos = [block.title.x, block.title.y, block.title.z];
        renderable.title!.anchor = block.title.anchor === 'left' ? 'right-middle' : block.title.anchor === 'below' ? 'bottom-center' : 'top-center';
        renderable.title!.visible = revealed;
        renderable.title!.el.dataset.node = node.name;
        renderable.title!.offset = this.options.labels?.[node.name];
        texOf(renderable.title!, backward ? `\\partial\\ell/\\partial ${node.tex}` : node.tex);
        let largest = 0;
        for (let i = 0; i < values.length; i++) largest = Math.max(largest, Math.abs(values[i]));
        const digits = defaultDigits(largest);
        renderable.values.forEach((label, i) => {
          const element = block.elements[i];
          label.setText(formatNumber(values[i], { digits }));
          label.el.dataset.node = node.name;
          label.el.dataset.index = element.index.join(',');
          label.visible = revealed && active && focusedRow(element.row);
          label.setFlag('vf-target', this.targetOf(block, element.row) === element.col);
        });
      }
      if (renderable.nodes !== undefined) {
        renderable.nodeLabels.forEach((label, i) => {
          const element = block.elements[i];
          const elemTex = this.explainer.elemTex({ node, index: element.index });
          const value = formatNumber(values[i], { tex: true });
          texOf(label, backward ? `\\partial\\ell/\\partial ${elemTex} = ${value}` : `${elemTex} = ${value}`);
          const side = block.title.anchor === 'right' && this.narrow;
          const mesh = renderable.nodes!;
          const toward = block.batched ? -1 : 1;
          label.pos = side ? [element.x + toward * (mesh.radius + SIDE_LABEL_GAP), element.y + mesh.height / 2, element.z] : [element.x, element.y + mesh.height / 2 + 0.45, element.z];
          label.anchor = !side ? 'top-center' : block.batched ? 'right-middle' : 'left-middle';
          label.el.dataset.node = node.name;
          label.el.dataset.index = element.index.join(',');
          label.offset = this.options.labels?.[node.name];
          label.visible = revealed && focusedRow(element.row);
        });
        const colors = block.elements.map((element, i) => nodeColor(values[i], this.theme, { dir, revealed, active, focused: focusedRow(element.row) }));
        renderable.nodes.update(block.elements, colors, block.elements.map(() => revealed));
      }
      renderable.classes.forEach((label, i) => {
        const text = block.classLabels[i];
        const target = this.targetOf(block, block.rows - 1);
        const isTarget = this.aligned.has(node) && !block.batched && target === text.col;
        label.setText(block.role === 'planar' && isTarget ? `${text.text} · 目标` : text.text);
        label.setFlag('vf-on', isTarget);
        label.pos = [text.x, text.y, text.z];
        label.anchor = text.anchor === 'above' ? 'top-center' : text.anchor === 'left' ? 'right-middle' : 'bottom-center';
        label.visible = revealed;
      });
      renderable.rows.forEach((label, i) => {
        const text = block.rowLabels[i];
        label.setText(text.text);
        label.pos = [text.x, text.y, text.z];
        label.visible = revealed;
      });
      const ticks = renderable.ticks[dir];
      renderable.tickLabels.forEach((label, i) => {
        const axis = block.axis!;
        if (i < ticks.length) {
          label.setText(String(ticks[i]).replace('-', '\u2212'));
          label.pos = [axis.x - 0.18, block.elements[0].y + ticks[i] * scale, axis.z];
          label.visible = revealed;
        } else label.visible = false;
      });
      if (renderable.plate !== undefined) {
        const frameXs = block.frameRows.map((_, r) => {
          const target = this.targetOf(block, r);
          const column = block.elements.find((e) => e.col === (target ?? 0));
          return column?.x ?? block.elements[0].x;
        });
        renderable.plate.update({ revealed, active, backward }, frameXs, ticks, scale);
      }
    }
    this.buildEdges();
    this.geometryDirty = true;
    this.labelsDirty = true;
    if (this.pinned !== undefined) this.showNodeCard(this.pinned.node, this.pinned.el);
    if (this.hoverEdge !== undefined) {
      const still = this.edges.find((e) => e.edge.id === this.hoverEdge!.edge.id);
      this.setHoverEdge(still !== undefined && this.state.dir === 'backward' && this.state.revealed(still.toNode) ? still : undefined);
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
    const loss = this.g.findLoss();
    const rowOf = (node: Node, index: number[]) => {
      const renderable = this.renderableOf(node);
      return renderable !== undefined && renderable.block.batched ? elementAt(renderable.block, index).row : undefined;
    };
    const drawn: DrawnEdge[] = [];
    for (const edge of this.explainer.edges()) {
      let entry: Omit<DrawnEdge, 'points' | 'style' | 'screen' | 'zeroGrad' | 'focused'> | undefined;
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
          const from: Vec3 = i === 0 ? [aElement.x, y, aElement.z - board.a.footprint[1] / 2 - 0.02] : [tile(k, i - 1).x + tileW / 2 + 0.01, y, tile(k, i - 1).z];
          const to: Vec3 = [tile(k, i).x - tileW / 2 - 0.01, y, tile(k, i).z];
          entry = { edge, fromNode: board.a.node, toNode: edge.op, contract: true, toLoss: false, endpoints: () => [from, to], flat: true };
        } else {
          const last = k === board.b.rows - 1;
          const from: Vec3 = [tile(k, i).x, y, tile(k, i).z + tileD / 2 + 0.01];
          const to: Vec3 = last ? [outElement.x, y, outElement.z - board.out.footprint[1] / 2 - 0.02] : [tile(k + 1, i).x, y, tile(k + 1, i).z - tileD / 2 - 0.01];
          entry = { edge, fromNode: board.b.node, toNode: edge.op, contract: true, toLoss: false, endpoints: () => [from, to], flat: true };
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
          toLoss: to.node === loss,
          endpoints: () => [this.anchorOut(from.node, from.index)!, this.anchorIn(to.node, to.index)!],
          flat: false,
        };
      }
      drawn.push({
        ...entry,
        zeroGrad: this.explainer.zeroGrad(edge),
        focused: rows.every((row) => row === undefined || row === this.state.focus),
        points: [],
        style: { opacity: 0, dashed: false },
        screen: [],
      });
    }
    const overview = this.state.dir === 'forward' && this.state.index === 0;
    for (const d of drawn) {
      d.style = edgeStyle({
        dir: this.state.dir,
        overview,
        contract: d.contract,
        toLoss: d.toLoss,
        zeroGrad: d.zeroGrad,
        focused: d.focused,
        revealedFrom: this.state.revealed(d.fromNode),
        revealedTo: this.state.revealed(d.toNode),
        activeFrom: this.state.active(d.fromNode),
        activeTo: this.state.active(d.toNode),
      });
    }
    this.edges = drawn;
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
      block.elements.forEach((element, i) => {
        const focused = !block.batched || element.row === this.state.focus;
        colors.push(barColor(values[i], this.theme, { active: revealed && active, focused }));
        outline.push(outlineOpacity(revealed, active) * (block.shape === 'tile' ? 0.9 : 1));
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
    const t = tangentLength(a, b);
    const curve = d.flat
      ? new CubicBezierCurve3(new Vector3(...a), new Vector3(...a).lerp(new Vector3(...b), 1 / 3), new Vector3(...a).lerp(new Vector3(...b), 2 / 3), new Vector3(...b))
      : new CubicBezierCurve3(new Vector3(...a), new Vector3(a[0], a[1] + t, a[2]), new Vector3(b[0], b[1] - t, b[2]), new Vector3(...b));
    this.highlight = new Mesh(new TubeGeometry(curve, 40, HIGHLIGHT_RADIUS / Math.sqrt(this.camera.zoom), 8, false), this.highlightMaterial);
    this.scene.add(this.highlight);
  }

  private requestFrame(): void {
    if (this.frameHandle !== 0 || this.disposed) return;
    this.frameHandle = requestAnimationFrame((now) => {
      this.frameHandle = 0;
      this.frame(now);
    });
  }

  private frame(now: number): void {
    if (this.disposed) return;
    let stepped = false;
    for (const renderable of this.renderables) {
      if (renderable.transition.step(now)) stepped = true;
    }
    const running = this.renderables.some((r) => r.transition.active);
    if (this.geometryDirty || stepped) {
      this.updateGeometry();
      this.geometryDirty = false;
      this.labelsDirty = true;
    }
    const moved = this.controls.update();
    this.renderer.render(this.scene, this.camera);
    if (this.labelsDirty) {
      this.labels.project(this.camera, this.width, this.height);
      this.labelsDirty = false;
      this.screenVersion++;
    }
    if (running || moved) this.requestFrame();
  }

  private resize(): void {
    this.width = Math.max(1, this.root.clientWidth);
    this.height = Math.max(1, this.root.clientHeight);
    this.renderer.setSize(this.width, this.height, false);
    if (this.narrow !== this.width <= NARROW_WIDTH) this.refresh();
    this.frameAll();
    this.labelsDirty = true;
  }

  private framingPoints(): Vec3[] {
    const points: Vec3[] = [];
    for (const renderable of this.renderables) {
      const { block } = renderable;
      const [w, d] = block.footprint;
      const values = block.node.value.data;
      const grad = block.node.grad;
      let high = 0;
      let low = 0;
      for (let i = 0; i < values.length; i++) {
        for (const h of [values[i] * renderable.scale.forward, grad === null ? 0 : grad.data[i] * renderable.scale.backward]) {
          high = Math.max(high, Math.min(h, HEIGHT_CAP));
          low = Math.min(low, Math.max(h, -HEIGHT_CAP));
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

  private center(): Vector3 {
    const points = this.framingPoints();
    const min = [Infinity, Infinity, Infinity];
    const max = [-Infinity, -Infinity, -Infinity];
    for (const p of points) for (let axis = 0; axis < 3; axis++) {
      min[axis] = Math.min(min[axis], p[axis]);
      max[axis] = Math.max(max[axis], p[axis]);
    }
    return new Vector3((min[0] + max[0]) / 2, (min[1] + max[1]) / 2, (min[2] + max[2]) / 2);
  }

  private placeCamera(): void {
    const center = this.center();
    this.camera.position.copy(center).addScaledVector(this.direction, CAMERA_DISTANCE);
    this.camera.lookAt(center);
    this.controls.target.copy(center);
    this.camera.zoom = 1;
    this.camera.updateProjectionMatrix();
    this.controls.update();
  }

  private frameAll(): void {
    const center = this.center();
    const eye = center.clone().addScaledVector(this.direction, CAMERA_DISTANCE);
    const view = new Matrix4().lookAt(eye, center, this.camera.up).setPosition(eye).invert();
    const marginX = LABEL_SAFE_MARGIN;
    const marginY = LABEL_SAFE_MARGIN;
    const items: FrameItem[] = this.framingPoints().map((p) => {
      tmpVector.set(...p).applyMatrix4(view);
      return { x: tmpVector.x, y: tmpVector.y, left: marginX, right: marginX, up: marginY, down: marginY };
    });
    const geometry = { x: span(items, 1, (i) => i.x, () => 0, () => 0), y: span(items, 1, (i) => i.y, () => 0, () => 0) };
    for (const label of this.labels.labels) {
      if (label.pos === undefined) continue;
      if (label.size === undefined) label.size = [label.el.offsetWidth, label.el.offsetHeight];
      const [w, h] = label.size;
      if (w === 0 && h === 0) continue;
      const [ax, ay] = anchorOffset(label.anchor);
      tmpVector.set(...label.pos).applyMatrix4(view);
      const flippable = label.kind === 'node' && label.offset === undefined;
      const [dx, dy] = label.offset ?? [0, 0];
      items.push({
        x: tmpVector.x,
        y: tmpVector.y,
        left: LABEL_SAFE_MARGIN + (flippable ? w : -ax * w - dx),
        right: LABEL_SAFE_MARGIN + (flippable ? 0 : (ax + 1) * w + dx),
        up: LABEL_SAFE_MARGIN - ay * h - dy,
        down: (ay + 1) * h + dy + LABEL_SAFE_MARGIN,
      });
    }
    const reserve = this.chromeBottom();
    const availableH = Math.max(this.height - reserve, this.height / 2);
    const spanX = (k: number) => span(items, k, (i) => i.x, (i) => i.left, (i) => i.right);
    const spanY = (k: number) => span(items, k, (i) => -i.y, (i) => i.up, (i) => i.down);
    const fits = (k: number) => spanX(k).size <= this.width && spanY(k).size <= availableH;
    let k = Math.min((this.width - 2 * marginX) / geometry.x.size, (availableH - 2 * marginY) / geometry.y.size);
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
    this.camera.left = cx - (this.width * unit) / 2;
    this.camera.right = cx + (this.width * unit) / 2;
    this.camera.top = cy + (reserve + availableH / 2) * unit;
    this.camera.bottom = this.camera.top - this.height * unit;
    this.camera.updateProjectionMatrix();
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
    this.placeCamera();
    this.frameAll();
    this.labelsDirty = true;
    this.requestFrame();
  }

  private applyTheme(): void {
    this.theme = readTheme();
    this.scene.background = new Color(...this.theme.surface);
    for (const renderable of this.renderables) {
      renderable.bars?.setTheme(this.theme);
      renderable.nodes?.setTheme(this.theme);
      renderable.plate?.setTheme(this.theme);
    }
    this.lines.setColor(this.theme.ink);
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
      if (!this.state.revealed(d.toNode)) continue;
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
    const ref = { node, index: element.index };
    const elemTex = this.explainer.elemTex(ref);
    const value = this.valuesOf(node)[index];
    const label = this.aligned.get(node);
    const token = label?.tokens?.[element.col];
    const fragment = document.createDocumentFragment();
    const head = document.createElement('div');
    head.className = 'vf-f';
    const formula = backward ? '' : this.forwardFormula(node, element);
    head.appendChild(render(backward ? `\\partial\\ell/\\partial ${elemTex}` : `${elemTex}${formula}`));
    if (token !== undefined) head.append(`，${tokenText(token)}`);
    fragment.appendChild(head);
    const line = document.createElement('div');
    line.className = 'vf-m';
    line.textContent = `= ${backward ? this.backwardTerms(node, element) : ''}${formatNumber(value)}`;
    fragment.appendChild(line);
    const target = this.targetOf(block, element.row);
    if (label !== undefined && target !== element.col) {
      const note = document.createElement('div');
      note.className = 'vf-s';
      note.textContent = label.tokens === undefined ? '点击设为目标类' : '点击设为目标 token';
      fragment.appendChild(note);
    }
    return fragment;
  }

  private backwardTerms(node: Node, element: Element): string {
    const out = this.explainer.outEdges({ node, index: element.index });
    if (out.length < 2 || out.length > TIP_TERMS) return '';
    return `${out.map((e) => parenthesized(this.explainer.contribution(e))).join(' + ')} = `;
  }

  private forwardFormula(node: Node, element: Element): string {
    if (node.kind !== 'op') return '';
    const def = explainOf(node.op as string);
    if (def === undefined) return '';
    const key = element.index.join(',');
    const incoming = this.explainer.edges(node).filter((e) => !('cell' in e.to) && e.to.index.join(',') === key);
    if (incoming.length === 0 || incoming.length > TIP_TERMS || incoming.some((e) => 'cell' in e.from)) return '';
    const perArg = node.inputs.map((_, arg) => incoming.filter((e) => e.arg === arg));
    if (perArg.some((list) => list.length !== 1)) return '';
    return ` = ${def.tex.forward({ out: this.explainer.elemTex({ node, index: element.index }), in: perArg.map((list) => this.explainer.elemTex(list[0].from)), attrs: node.attrs })}`;
  }

  private showTip(renderable: Renderable, index: number, ev: PointerEvent): void {
    this.tip.replaceChildren(this.tipContent(renderable, index));
    this.tip.hidden = false;
    const rect = this.root.getBoundingClientRect();
    placeTip(this.root, this.tip, ev.clientX - rect.left, ev.clientY - rect.top);
  }

  private showNodeCard(node: Node, el: HTMLElement): void {
    this.pinned = { node, el };
    this.pop.replaceChildren(nodeCardElement(this.explainer.nodeCard(node, this.state.dir)));
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
      const clickable = this.aligned.has(hit.renderable.block.node);
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
      if (hit === undefined || !this.aligned.has(hit.renderable.block.node)) return;
      this.setTarget(hit.renderable, hit.index);
      this.showTip(hit.renderable, hit.index, ev);
    };
    canvas.addEventListener('pointermove', onMove);
    canvas.addEventListener('pointerleave', onLeave);
    canvas.addEventListener('pointerdown', onDown);
    canvas.addEventListener('pointerup', onUp);
    this.unsubscribe.push(() => {
      canvas.removeEventListener('pointermove', onMove);
      canvas.removeEventListener('pointerleave', onLeave);
      canvas.removeEventListener('pointerdown', onDown);
      canvas.removeEventListener('pointerup', onUp);
    });
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    if (this.frameHandle !== 0) cancelAnimationFrame(this.frameHandle);
    this.resizeObserver.disconnect();
    this.unsubscribe.forEach((fn) => fn());
    this.controls.dispose();
    for (const renderable of this.renderables) {
      renderable.bars?.dispose();
      renderable.nodes?.dispose();
      renderable.plate?.dispose();
    }
    this.lines.dispose();
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
