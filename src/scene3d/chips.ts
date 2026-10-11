import { BufferGeometry, Color, ConeGeometry, Group, Line, LineBasicMaterial, Mesh, MeshBasicMaterial, Vector3 } from 'three';
import { render } from '../formula';
import type { LinearRgb } from '../theme';
import type { SceneController } from './controller';
import type { Label } from './labels';
import { tokenText } from './layout';
import { CHIP_LIFT } from './levels';
import { renderableFor } from './renderable';

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

interface SceneArrow {
  group: Group;
  label: Label;
  setLength(length: number): void;
}

function mixRgb(a: LinearRgb, b: LinearRgb, k: number): LinearRgb {
  return [a[0] + (b[0] - a[0]) * k, a[1] + (b[1] - a[1]) * k, a[2] + (b[2] - a[2]) * k];
}

export class TokenChips {
  private readonly chipLabels = new Map<string, Label[]>();
  private arrow: SceneArrow | undefined;

  constructor(private readonly c: SceneController) {}

  update(): void {
    const c = this.c;
    const chips = c.levelLayout?.chips ?? [];
    let arrowShown = false;
    for (const chip of chips) {
      const renderable = renderableFor(c, chip.block);
      if (renderable === undefined) continue;
      const block = chip.block;
      const rows = block.rows;
      let labels = this.chipLabels.get(block.key ?? block.node.name + chip.label.name);
      if (labels === undefined) {
        labels = Array.from({ length: rows }, () => c.labels.add('tok'));
        this.chipLabels.set(block.key ?? block.node.name + chip.label.name, labels);
      }
      const y = block.elements[0].y;
      const zs = Array.from({ length: rows }, (_, r) => block.elements.find((e) => e.row === r)!.z);
      const shown = !c.lod && !renderable.far && block.hidden !== true && (chip.arrow || c.state.revealed(block.node));
      labels.forEach((label, t) => {
        const id = Math.round(chip.label.value.data[Math.min(t, chip.label.value.size - 1)]);
        const token = chip.label.tokens?.[id] ?? String(id);
        const elem = c.explainer.elemTex({ node: chip.label, index: [t] });
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
    const c = this.c;
    const group = new Group();
    const shaftMaterial = new LineBasicMaterial({ color: new Color(...c.theme.ink), transparent: true, opacity: ARROW_OPACITY });
    const shaft = new Line(new BufferGeometry().setFromPoints([new Vector3(0, 0, 0), new Vector3(0, 0, 1)]), shaftMaterial);
    const headMaterial = new MeshBasicMaterial({ color: new Color(...mixRgb(c.theme.surface, c.theme.ink, ARROW_OPACITY)) });
    const head = new Mesh(new ConeGeometry(ARROW_HEAD[0], ARROW_HEAD[1], ARROW_HEAD_SEGMENTS), headMaterial);
    head.rotation.x = Math.PI / 2;
    group.add(shaft, head);
    c.scene.add(group);
    const label = c.labels.add('kk', 'vf-arrow-tag');
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
}
