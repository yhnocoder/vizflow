import { BoxGeometry, EdgesGeometry, LineBasicMaterial, LineSegments, Mesh, MeshBasicMaterial } from 'three';
import { renderText } from '../formula';
import type { SceneController } from './controller';
import type { Label } from './labels';
import type { FoldBox } from './levels';

export const BOX_FOLDED_TEXT = '已折叠';
export const BOX_FIXED_TEXT = '占位';
export const BOX_FIXED_NOTE = '数值按恒等';

const BOX_MIN_HEIGHT = 0.01;
const BOX_LABEL_OFFSET: [number, number] = [0.2, 0.15];
const BOX_OPACITY = 0.3;
const BOX_EDGE_OPACITY = 0.6;

export interface SceneBox {
  box: FoldBox;
  mesh: Mesh;
  material: MeshBasicMaterial;
  edgeMaterial: LineBasicMaterial;
  label: Label;
}

export class FoldBoxes {
  private readonly boxes = new Map<string, SceneBox>();
  private readonly geometry = new BoxGeometry(1, 1, 1);
  private readonly edges = new EdgesGeometry(this.geometry);

  constructor(private readonly c: SceneController) {}

  update(): void {
    const theme = this.c.theme;
    const shown = new Set<string>();
    const platforms = this.c.platforms;
    const shifts = platforms?.shifts() ?? [[0, 0]];
    const fold = platforms?.foldStep();
    const boxes = (this.c.levelLayout?.boxes ?? []).filter((box) => platforms === undefined || platforms.involved() === undefined || box.module.id === fold);
    for (const box of boxes) for (const [k, [dx, dy]] of shifts.entries()) {
      const id = k === 0 ? box.module.id : `${k}:${box.module.id}`;
      shown.add(id);
      const entry = this.boxes.get(id) ?? this.create(box, id);
      entry.box = box;
      const [x0, x1] = [box.x[0] + dx, box.x[1] + dx];
      const [y0, y1] = [box.y[0] + dy, box.y[1] + dy];
      const [z0, z1] = box.z;
      entry.mesh.scale.set(x1 - x0, Math.max(BOX_MIN_HEIGHT, y1 - y0), z1 - z0);
      entry.mesh.position.set((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2);
      entry.mesh.visible = true;
      entry.material.color.setRGB(...theme.bg);
      entry.edgeMaterial.color.setRGB(...theme.ink);
      entry.label.pos = [x0 + BOX_LABEL_OFFSET[0], y0 + BOX_LABEL_OFFSET[1], z1];
      entry.label.visible = k === 0;
    }
    for (const [id, entry] of this.boxes) {
      if (shown.has(id)) continue;
      entry.mesh.visible = false;
      entry.label.visible = false;
    }
  }

  private create(box: FoldBox, id: string): SceneBox {
    const c = this.c;
    const material = new MeshBasicMaterial({ transparent: true, opacity: BOX_OPACITY, depthWrite: false });
    const edgeMaterial = new LineBasicMaterial({ transparent: true, opacity: BOX_EDGE_OPACITY });
    const mesh = new Mesh(this.geometry, material);
    mesh.add(new LineSegments(this.edges, edgeMaterial));
    c.scene.add(mesh);
    const module = box.module;
    const label = c.labels.add('boxlb', module.fixed ? 'vf-fixed' : '');
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
    label.el.addEventListener('pointerenter', () => c.popups.showBoxCard(entry));
    label.el.addEventListener('pointerleave', () => c.popups.hideNodeCard());
    if (!module.fixed) label.el.addEventListener('click', () => c.steps?.setFolded(module.id, false));
    this.boxes.set(id, entry);
    return entry;
  }
}
