import { OrthographicCamera, Vector3 } from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import type { SceneController } from './controller';
import { CAMERA_DISTANCE, allPoints, center, frameAll } from './framing';
import type { Vec3 } from './layout';

export const DEFAULT_DIRECTION: Vec3 = [0.42, 0.55, 1];
export const DEFAULT_AZIMUTH: [number, number] = [-0.8, 0.7];
export const DEFAULT_POLAR: [number, number] = [0.3, 1.5];
export const DEFAULT_ZOOM: [number, number] = [0.1, 4];
export const DAMPING = 0.1;
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

const FOLLOW_SETTLE_Y = 0.01;
const FOLLOW_SETTLE_ZOOM = 0.002;

interface FollowGoal {
  y: number;
  zoom: number;
}

export class CameraRig {
  readonly camera = new OrthographicCamera(-1, 1, 1, -1, 0.1, CAMERA_DISTANCE * 5);
  readonly controls: OrbitControls;
  readonly direction: Vector3;
  private readonly zoomRange: [number, number];
  goal: FollowGoal | undefined;
  wheelTimer = 0;

  constructor(private readonly c: SceneController) {
    const cameraOptions = c.options.camera ?? {};
    this.zoomRange = cameraOptions.zoom ?? DEFAULT_ZOOM;
    this.direction = new Vector3(...(cameraOptions.direction ?? DEFAULT_DIRECTION)).normalize();
    this.controls = new OrbitControls(this.camera, c.renderer.domElement);
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
      c.markLabelsDirty();
      c.requestFrame();
    });
  }

  private limitAzimuth(): void {
    const azimuth = Math.atan2(this.direction.x, this.direction.z);
    const [azLo, azHi] = this.c.options.camera?.azimuth ?? DEFAULT_AZIMUTH;
    this.controls.minAzimuthAngle = azimuth + azLo;
    this.controls.maxAzimuthAngle = azimuth + azHi;
  }

  setDirection(direction: Vec3): void {
    this.direction.set(...direction).normalize();
    this.limitAzimuth();
  }

  zoomNow(): number {
    return this.goal?.zoom ?? this.camera.zoom;
  }

  followStep(): void {
    this.goal = this.followGoal();
    if (this.c.reduceMotion.matches) this.applyGoal(1);
  }

  jumpTo(y: number): void {
    this.goal = { y: this.clampFollowY(y, this.halfHeight()), zoom: 1 };
  }

  applyGoal(k: number): boolean {
    const goal = this.goal;
    if (goal === undefined) return false;
    const dy = (goal.y - this.controls.target.y) * k;
    this.controls.target.y += dy;
    this.camera.position.y += dy;
    this.camera.zoom += (goal.zoom - this.camera.zoom) * k;
    this.camera.updateProjectionMatrix();
    if (Math.abs(goal.y - this.controls.target.y) < FOLLOW_SETTLE_Y && Math.abs(goal.zoom - this.camera.zoom) < FOLLOW_SETTLE_ZOOM) this.goal = undefined;
    this.c.markLabelsDirty();
    return true;
  }

  private halfHeight(): number {
    return (this.camera.top - this.camera.bottom) / 2;
  }

  reframe(): void {
    if (this.c.follow) this.followFrustum();
    else frameAll(this.c);
  }

  private followFrustum(): void {
    const aspect = this.c.width / this.c.height;
    const half = Math.max(this.c.height / (2 * FOLLOW_PIXELS_PER_UNIT), FOLLOW_MIN_HALF_WIDTH / aspect);
    this.camera.left = -half * aspect;
    this.camera.right = half * aspect;
    this.camera.top = half;
    this.camera.bottom = -half;
    this.camera.updateProjectionMatrix();
  }

  private levelRange(): { lo: number; top: number } {
    const levels = this.c.levelLayout;
    if (levels !== undefined) return { lo: levels.lo, top: levels.top };
    const ys = this.c.renderables.map((r) => r.block.elements[0].y);
    return { lo: Math.min(...ys), top: Math.max(...ys) + FOLLOW_TOP_ROOM };
  }

  private clampFollowY(y: number, half: number): number {
    const { lo, top } = this.levelRange();
    return Math.min(Math.max(y, lo + half * FOLLOW_EDGE_FILL - FOLLOW_BOTTOM_SLACK), top - half * FOLLOW_EDGE_FILL + FOLLOW_TOP_SLACK);
  }

  private followGoal(): FollowGoal {
    const c = this.c;
    const hh = this.halfHeight();
    const { lo, top } = this.levelRange();
    const keys = c.lod ? [] : c.renderables.filter((r) => r.block.copyOf === undefined && c.state.active(r.block.node));
    if (keys.length === 0) return { y: (lo + top - FOLLOW_TOP_ROOM) / 2, zoom: Math.min(1, (2 * hh) / (top - lo + FOLLOW_OVERVIEW_MARGIN)) };
    const ys = keys.map((r) => r.block.elements[0].y);
    const kLo = Math.min(...ys);
    const kHi = Math.max(...ys);
    const zoom = Math.min(1, (2 * hh) / (kHi - kLo + FOLLOW_STEP_MARGIN));
    return { y: this.clampFollowY((kLo + kHi) / 2 + FOLLOW_STEP_LIFT, hh / zoom), zoom };
  }

  placeCamera(): void {
    const middle = center(allPoints(this.c));
    let zoom = 1;
    if (this.c.follow) {
      const goal = this.followGoal();
      middle.y = goal.y;
      zoom = goal.zoom;
    }
    this.goal = undefined;
    this.camera.position.copy(middle).addScaledVector(this.direction, CAMERA_DISTANCE);
    this.camera.lookAt(middle);
    this.controls.target.copy(middle);
    this.camera.zoom = zoom;
    this.camera.updateProjectionMatrix();
    this.controls.update();
  }

  viewDirection(): Vector3 {
    return this.camera.position.clone().sub(this.controls.target).normalize();
  }

  reset(): void {
    this.controls.enableDamping = false;
    this.controls.update();
    this.controls.enableDamping = true;
    this.controls.reset();
    this.reframe();
    this.placeCamera();
  }

  onWheel(ev: WheelEvent): void {
    if (!(ev.ctrlKey || ev.metaKey)) return;
    ev.preventDefault();
    const [lo, hi] = this.zoomRange;
    this.camera.zoom = Math.min(hi, Math.max(lo, this.camera.zoom * Math.exp(-ev.deltaY * WHEEL_ZOOM_RATE)));
    this.camera.updateProjectionMatrix();
    this.goal = undefined;
    this.c.markLabelsDirty();
    this.c.requestFrame();
    clearTimeout(this.wheelTimer);
    this.wheelTimer = window.setTimeout(() => this.c.refresh(), WHEEL_SETTLE_MS);
  }
}
