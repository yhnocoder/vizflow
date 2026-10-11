import { AmbientLight, Color, DirectionalLight, type OrthographicCamera, SRGBColorSpace, Scene, type WebGLRenderer } from 'three';
import type { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import type { Graph, Node } from '../autograd';
import { type Explainer, explain } from '../explain';
import { type StepInfo, type StepState, type Steps, allVisible } from '../steps';
import { type ThemeColors, onThemeChange, readTheme } from '../theme';
import { PlateMesh } from './blocks';
import { markFar, markSeeThrough, refreshBlock, updateBars } from './blockview';
import { FOLLOW_RATE, CameraRig } from './camera';
import { type OverviewNav, buildChrome, buildTools, updateNav } from './chrome';
import { TokenChips } from './chips';
import { FoldBoxes } from './boxes';
import { type EditHost, mountEditor } from './edit';
import { EdgeLines } from './edgelines';
import { buildEdges } from './edges';
import { LabelLayer } from './labels';
import { type BoardGeometry, type Vec3, classAlignment, resolveLayout, targetColumn } from './layout';
import type { LevelLayout } from './levels';
import { checkModules } from './modules';
import { FormulaPanel } from './formulapanel';
import { ModuleBraces } from './moduleview';
import { SceneNotes } from './notes';
import { blockAt, bindPointer } from './pointer';
import { ScenePopups } from './popups';
import { type Renderable, createRenderable } from './renderable';
import type { SceneOptions } from './scene';

export type PageStepInfo = StepInfo & { kind?: string; sub?: string };

export const PIXEL_RATIO_CAP = 2;

const MAX_FRAME_SECONDS = 0.05;
const AMBIENT_INTENSITY = 0.72;
const DIRECTIONAL_INTENSITY = 0.42;
const LEGACY_LIGHT_FACTOR = Math.PI;
const LIGHT_POSITION: Vec3 = [-4, 12, 8];

export class SceneController implements EditHost {
  readonly renderer: WebGLRenderer;
  readonly scene = new Scene();
  readonly explainer: Explainer;
  readonly aligned: Map<Node, Node>;
  readonly labels: LabelLayer;
  readonly follow: boolean;
  readonly reduceMotion = matchMedia('(prefers-reduced-motion: reduce)');
  readonly rig: CameraRig;
  readonly popups: ScenePopups;
  readonly edgeLines: EdgeLines;
  readonly moduleBraces: ModuleBraces;
  readonly boxes: FoldBoxes;
  readonly chips: TokenChips;
  private readonly nav: OverviewNav | undefined;
  private readonly formula: FormulaPanel | undefined;
  readonly notes: SceneNotes;
  private readonly resizeObserver: ResizeObserver;
  readonly unsubscribe: (() => void)[] = [];
  private stepKey: string | undefined;
  private lastFrame: number | undefined;
  private far = true;
  private seeThrough = true;
  private frontRow = true;
  lod = false;
  stepNodes = new Set<Node>();
  renderables: Renderable[] = [];
  boards: BoardGeometry[] = [];
  levelLayout: LevelLayout | undefined;
  theme: ThemeColors;
  state: StepState;
  width = 1;
  height = 1;
  private frameHandle = 0;
  private geometryDirty = true;
  private labelsDirty = true;
  screenVersion = 0;
  private disposed = false;

  constructor(
    readonly root: HTMLDivElement,
    readonly g: Graph,
    readonly options: SceneOptions,
    renderer: WebGLRenderer,
  ) {
    this.renderer = renderer;
    this.theme = readTheme();
    this.explainer = explain(g);
    this.aligned = classAlignment(g.nodes);
    this.follow = options.camera?.follow === true;
    root.classList.toggle('vf-follow', this.follow);
    this.state = this.currentState();
    renderer.setPixelRatio(Math.min(devicePixelRatio, PIXEL_RATIO_CAP));
    renderer.outputColorSpace = SRGBColorSpace;
    root.appendChild(renderer.domElement);
    this.labels = new LabelLayer(root, () => {
      this.rig.reframe();
      this.labelsDirty = true;
      this.requestFrame();
    });
    buildChrome(root, options.hint);
    this.popups = new ScenePopups(this);
    this.rig = new CameraRig(this);

    this.scene.add(new AmbientLight(0xffffff, AMBIENT_INTENSITY * LEGACY_LIGHT_FACTOR));
    const sun = new DirectionalLight(0xffffff, DIRECTIONAL_INTENSITY * LEGACY_LIGHT_FACTOR);
    sun.position.set(...LIGHT_POSITION);
    this.scene.add(sun);
    this.scene.background = new Color(...this.theme.surface).convertSRGBToLinear();
    this.edgeLines = new EdgeLines(this);
    this.moduleBraces = new ModuleBraces(this);
    this.boxes = new FoldBoxes(this);
    this.chips = new TokenChips(this);

    this.buildBlocks();
    const modules = options.modules ?? [];
    checkModules(modules, this.renderables.map((r) => r.block.node));
    this.moduleBraces.mountLabels(modules);
    this.nav = buildTools(this);
    this.formula = this.nav === undefined ? undefined : new FormulaPanel(this);
    this.notes = new SceneNotes(this);
    this.refresh();
    this.resize();
    this.rig.placeCamera();
    this.controls.saveState();

    this.unsubscribe.push(g.subscribe(() => this.refresh()));
    if (options.steps !== undefined) this.unsubscribe.push(options.steps.subscribe(() => this.refresh()));
    this.unsubscribe.push(onThemeChange(() => this.applyTheme()));
    this.resizeObserver = new ResizeObserver(() => {
      this.resize();
      this.requestFrame();
    });
    this.resizeObserver.observe(root);
    this.unsubscribe.push(bindPointer(this));
    if (options.edit === true) this.unsubscribe.push(mountEditor(this));
    this.requestFrame();
  }

  get camera(): OrthographicCamera {
    return this.rig.camera;
  }

  get controls(): OrbitControls {
    return this.rig.controls;
  }

  get canvas(): HTMLCanvasElement {
    return this.renderer.domElement;
  }

  get visible(): readonly Node[] {
    return this.options.steps?.visible ?? this.g.nodes;
  }

  get steps(): Steps | undefined {
    return this.options.steps as Steps | undefined;
  }

  setDirection(direction: Vec3): void {
    this.rig.setDirection(direction);
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
    return blockAt(this, clientX, clientY);
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

  currentInfo(): PageStepInfo | undefined {
    const steps = this.options.steps;
    return steps === undefined ? undefined : (steps.list[steps.current] as PageStepInfo);
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
      this.renderables = resolved.blocks.map((block) => createRenderable(this, block));
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

  refresh(): void {
    if (this.disposed) return;
    this.state = this.currentState();
    this.resolve();
    const now = performance.now();
    const dir = this.state.dir;
    const info = this.currentInfo();
    const overview = dir === 'forward' && this.state.index === 0;
    this.lod = this.follow && (overview || info?.kind === 'inputs');
    this.stepNodes = new Set(info?.nodes ?? this.visible);
    markFar(this);
    markSeeThrough(this);
    const stepKey = `${dir}:${this.state.index}:${info?.id ?? ''}`;
    if (this.follow && this.stepKey !== undefined && stepKey !== this.stepKey) this.rig.followStep();
    this.stepKey = stepKey;
    const zoom = this.rig.zoomNow();
    for (const renderable of this.renderables) refreshBlock(this, renderable, now, zoom);
    this.moduleBraces.update();
    this.boxes.update();
    this.chips.update();
    this.edgeLines.edges = buildEdges(this);
    if (this.nav !== undefined) updateNav(this, this.nav);
    this.formula?.update();
    this.notes.update(info?.notes ?? [], info?.id ?? 'all');
    this.geometryDirty = true;
    this.labelsDirty = true;
    this.popups.repin();
    const hover = this.edgeLines.hover;
    if (hover !== undefined) {
      const still = this.edgeLines.edges.find((e) => e.edge.id === hover.edge.id);
      this.edgeLines.setHover(still !== undefined && this.state.dir === 'backward' && this.state.revealed(still.fromNode) ? still : undefined);
      if (this.edgeLines.hover === undefined) this.popups.hideUnpinned();
    }
    this.requestFrame();
  }

  private updateGeometry(): void {
    for (const renderable of this.renderables) {
      if (renderable.bars !== undefined) updateBars(this, renderable, renderable.bars);
    }
    this.edgeLines.redraw();
  }

  requestFrame(): void {
    if (this.frameHandle !== 0 || this.disposed) return;
    this.frameHandle = requestAnimationFrame((now) => {
      this.frameHandle = 0;
      this.frame(now);
    });
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
    const following = this.rig.applyGoal(k);
    const running = this.renderables.some((r) => r.transition.active) || this.rig.goal !== undefined;
    if (this.geometryDirty || stepped) {
      this.updateGeometry();
      this.geometryDirty = false;
      this.labelsDirty = true;
    }
    const moved = this.controls.update();
    if ((moved || stepped || following || this.labelsDirty) && markSeeThrough(this)) {
      for (const renderable of this.renderables) refreshBlock(this, renderable, now, this.rig.zoomNow());
      this.labelsDirty = true;
    }
    if (this.labelsDirty || moved || following) this.moduleBraces.place();
    this.renderer.render(this.scene, this.camera);
    if (this.labelsDirty || moved) {
      this.labels.project(this.camera, this.width, this.height, !this.follow);
      this.notes.place();
      this.labelsDirty = false;
      this.screenVersion++;
    }
    if (running || moved) this.requestFrame();
    else this.lastFrame = undefined;
  }

  private resize(): void {
    this.width = Math.max(1, this.root.clientWidth);
    this.height = Math.max(1, this.root.clientHeight);
    this.renderer.setSize(this.width, this.height, false);
    this.rig.reframe();
    this.labelsDirty = true;
  }

  get farOn(): boolean {
    return this.far;
  }

  toggleFar(): boolean {
    this.far = !this.far;
    return this.far;
  }

  get seeThroughOn(): boolean {
    return this.seeThrough;
  }

  toggleSeeThrough(): boolean {
    this.seeThrough = !this.seeThrough;
    return this.seeThrough;
  }

  get frontRowOnly(): boolean {
    return this.frontRow;
  }

  toggleFrontRow(): boolean {
    this.frontRow = !this.frontRow;
    return this.frontRow;
  }

  markLabelsDirty(): void {
    this.labelsDirty = true;
  }

  markGeometryDirty(): void {
    this.geometryDirty = true;
  }

  resetView(): void {
    this.rig.reset();
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
    this.edgeLines.setTheme(this.theme);
    this.moduleBraces.setTheme(this.theme);
    this.refresh();
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    if (this.frameHandle !== 0) cancelAnimationFrame(this.frameHandle);
    clearTimeout(this.rig.wheelTimer);
    this.resizeObserver.disconnect();
    this.unsubscribe.forEach((fn) => fn());
    this.controls.dispose();
    for (const renderable of this.renderables) {
      renderable.bars?.dispose();
      renderable.nodes?.dispose();
      renderable.plate?.dispose();
    }
    this.edgeLines.dispose();
    this.moduleBraces.dispose();
    this.labels.dispose();
    this.notes.dispose();
    this.renderer.dispose();
    this.renderer.forceContextLoss();
    this.root.remove();
  }
}
