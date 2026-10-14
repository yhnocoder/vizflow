import { BoxGeometry, Color, InstancedMesh, Matrix4, MeshLambertMaterial, type Object3D, type Scene } from 'three';
import type { ThemeColors } from '../theme';
import { PLATE_THICKNESS, type Plate } from './layout';
import { LineBatch, type Segments } from './lines';
import { plateFill, seeThroughFill } from './blocks';
import type { Renderable } from './renderable';

export const BAR_INK_ORDER = 1;

const matrix = new Matrix4();

class PlateFills {
  mesh: InstancedMesh;
  owners: Renderable[] = [];
  readonly material: MeshLambertMaterial;

  constructor(
    private readonly scene: Scene,
    private readonly geometry: BoxGeometry,
    theme: ThemeColors,
    seeThrough: boolean,
  ) {
    this.material = new MeshLambertMaterial({ color: new Color(...theme.line) });
    seeThroughFill(this.material, seeThrough);
    this.mesh = this.create(1);
  }

  private create(capacity: number): InstancedMesh {
    const mesh = new InstancedMesh(this.geometry, this.material, capacity);
    mesh.count = 0;
    this.scene.add(mesh);
    return mesh;
  }

  set(owners: Renderable[], plates: Plate[]): void {
    if (plates.length > this.mesh.instanceMatrix.count) {
      this.scene.remove(this.mesh);
      this.mesh.dispose();
      this.mesh = this.create(plates.length);
    }
    plates.forEach((plate, i) => this.mesh.setMatrixAt(i, plateFill(plate, matrix)));
    this.mesh.count = plates.length;
    this.mesh.instanceMatrix.needsUpdate = true;
    this.mesh.computeBoundingSphere();
    this.owners = owners;
  }

  dispose(): void {
    this.scene.remove(this.mesh);
    this.mesh.dispose();
    this.material.dispose();
  }
}

export class SceneBatches {
  private readonly barInk: LineBatch;
  private readonly ink: LineBatch;
  private readonly geometry = new BoxGeometry(1, PLATE_THICKNESS, 1);
  private readonly solid: PlateFills;
  private readonly clear: PlateFills;

  constructor(
    private readonly scene: Scene,
    theme: ThemeColors,
  ) {
    this.solid = new PlateFills(scene, this.geometry, theme, false);
    this.clear = new PlateFills(scene, this.geometry, theme, true);
    this.ink = new LineBatch(theme.ink, undefined, true);
    this.barInk = new LineBatch(theme.ink, theme.surface);
    this.barInk.object.renderOrder = BAR_INK_ORDER;
    scene.add(this.ink.object, this.barInk.object);
  }

  setTheme(theme: ThemeColors): void {
    this.ink.setColor(theme.ink);
    this.barInk.setColor(theme.ink, theme.surface);
    this.solid.material.color.setRGB(...theme.line);
    this.clear.material.color.setRGB(...theme.line);
  }

  update(renderables: readonly Renderable[]): void {
    const barInk: Segments[] = [];
    const ink: Segments[] = [];
    const solid: { owners: Renderable[]; plates: Plate[] } = { owners: [], plates: [] };
    const clear: { owners: Renderable[]; plates: Plate[] } = { owners: [], plates: [] };
    for (const renderable of renderables) {
      if (renderable.block.hidden === true) continue;
      if (renderable.bars !== undefined) barInk.push(renderable.bars.outline);
      if (renderable.nodes !== undefined) ink.push(renderable.nodes.outline);
      const plate = renderable.plate;
      if (plate === undefined) continue;
      ink.push(plate.ink, plate.axis);
      const fills = plate.seeThrough ? clear : solid;
      for (const p of plate.plates) {
        fills.owners.push(renderable);
        fills.plates.push(p);
      }
    }
    this.barInk.setSegments(barInk);
    this.ink.setSegments(ink);
    this.solid.set(solid.owners, solid.plates);
    this.clear.set(clear.owners, clear.plates);
  }

  plateMeshes(): Object3D[] {
    return [this.solid.mesh, this.clear.mesh];
  }

  plateOwner(object: Object3D, instance: number | undefined): Renderable | undefined {
    const fills = object === this.solid.mesh ? this.solid : object === this.clear.mesh ? this.clear : undefined;
    return instance === undefined ? undefined : fills?.owners[instance];
  }

  dispose(): void {
    for (const batch of [this.ink, this.barInk]) {
      this.scene.remove(batch.object);
      batch.dispose();
    }
    this.solid.dispose();
    this.clear.dispose();
    this.geometry.dispose();
  }
}
