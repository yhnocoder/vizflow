import { type OrthographicCamera, Plane, Raycaster, Vector2, Vector3 } from 'three';
import type { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import type { Node } from '../autograd';
import editCss from './edit.css?inline';
import { type MoveTarget, layoutCode, literal, moveTarget, readPosition, round, templateName, writePosition } from './editconfig';
import type { LabelLayer } from './labels';
import { BOARD_ROW_GAP, type LayoutItem, STACK_BACK_OFFSET, STACK_LEVEL, type StackOptions, type Vec3 } from './layout';
import type { SceneOptions } from './scene';

export const EDIT_INPUT_STEP = 0.05;
export const DIRECTION_DIGITS = 2;
export const COPY_NOTE_MS = 2000;

export interface EditHost {
  readonly root: HTMLDivElement;
  readonly canvas: HTMLCanvasElement;
  readonly camera: OrthographicCamera;
  readonly controls: OrbitControls;
  readonly options: SceneOptions;
  readonly labels: LabelLayer;
  readonly visible: readonly Node[];
  blockAt(clientX: number, clientY: number): { node: Node; y: number } | undefined;
  relayout(reframe: boolean): void;
  setDirection(direction: Vec3): void;
}

const styleId = 'vf-edit-style';

const FIELDS: Record<LayoutItem['template'], { key: keyof StackOptions; fallback: number }[]> = {
  board: [
    { key: 'gap', fallback: BOARD_ROW_GAP },
    { key: 'scale', fallback: 1 },
  ],
  stack: [
    { key: 'level', fallback: STACK_LEVEL },
    { key: 'back', fallback: STACK_BACK_OFFSET },
  ],
  node: [],
};

function element<K extends keyof HTMLElementTagNameMap>(tag: K, className: string, text?: string): HTMLElementTagNameMap[K] {
  const el = document.createElement(tag);
  el.className = className;
  if (text !== undefined) el.textContent = text;
  return el;
}

function describe(target: MoveTarget, n: Node): string {
  const where = target.key === 'at' ? `${templateName(target.item)} 的 at` : `${templateName(target.item)} 的 offsets['${n.name}']`;
  return `${n.name}：${where} = ${literal(readPosition(target, n))}`;
}

export function mountEditor(host: EditHost): () => void {
  if (!document.getElementById(styleId)) {
    const style = document.createElement('style');
    style.id = styleId;
    style.textContent = editCss;
    document.head.appendChild(style);
  }
  const { root, canvas, options } = host;
  const layout = options.layout;
  const labelOffsets = options.labels!;
  const camera = options.camera!;
  root.classList.add('vf-editing');

  const panel = element('details', 'vf-edit');
  panel.open = true;
  panel.appendChild(element('summary', 'vf-edit-head', '编辑布局'));
  const body = element('div', 'vf-edit-body');
  panel.appendChild(body);
  layout.forEach((item, index) => {
    const fields = FIELDS[item.template];
    if (fields.length === 0) return;
    const row = element('div', 'vf-edit-item');
    row.dataset.item = String(index);
    row.appendChild(element('div', 'vf-edit-name', templateName(item)));
    for (const { key, fallback } of fields) {
      const field = element('label', 'vf-edit-field', key);
      const input = element('input', 'vf-edit-input');
      input.type = 'number';
      input.step = String(EDIT_INPUT_STEP);
      input.name = key;
      input.value = String((item.options as StackOptions)[key] ?? fallback);
      input.addEventListener('input', () => {
        const value = Number(input.value);
        if (input.value === '' || !Number.isFinite(value)) return;
        (item.options as Record<string, unknown>)[key] = value;
        host.relayout(true);
      });
      field.appendChild(input);
      row.appendChild(field);
    }
    body.appendChild(row);
  });
  const status = element('div', 'vf-edit-status', '拖动块或块名调整位置');
  const actions = element('div', 'vf-edit-actions');
  const setView = element('button', 'vf-chip', '设为默认视角');
  const copy = element('button', 'vf-chip', '复制配置');
  setView.type = 'button';
  copy.type = 'button';
  const note = element('span', 'vf-edit-note');
  note.setAttribute('role', 'status');
  actions.append(setView, copy, note);
  const output = element('textarea', 'vf-edit-code');
  output.readOnly = true;
  output.spellcheck = false;
  output.wrap = 'off';
  output.hidden = true;
  body.append(status, actions, output);
  root.appendChild(panel);
  panel.addEventListener('toggle', () => host.relayout(true));
  let noteTimer = 0;
  const flash = (text: string) => {
    note.textContent = text;
    note.classList.add('vf-on');
    clearTimeout(noteTimer);
    noteTimer = window.setTimeout(() => note.classList.remove('vf-on'), COPY_NOTE_MS);
  };

  setView.addEventListener('click', () => {
    const direction = host.camera.position.clone().sub(host.controls.target).normalize();
    const rounded = direction.toArray().map((v) => round(v, DIRECTION_DIGITS)) as Vec3;
    camera.direction = rounded;
    host.setDirection(rounded);
    status.textContent = `camera.direction = ${literal(rounded)}`;
  });
  copy.addEventListener('click', () => {
    const code = layoutCode({ layout, camera, labels: labelOffsets });
    output.value = code;
    output.hidden = false;
    output.focus();
    output.select();
    const written = navigator.clipboard?.writeText(code) ?? Promise.reject(new Error('no clipboard'));
    written.then(
      () => flash('已复制'),
      () => flash('请从文本框复制'),
    );
  });

  const raycaster = new Raycaster();
  const ndc = new Vector2();
  const groundPoint = (ev: PointerEvent, plane: Plane): Vector3 | undefined => {
    const rect = canvas.getBoundingClientRect();
    ndc.set(((ev.clientX - rect.left) / rect.width) * 2 - 1, -((ev.clientY - rect.top) / rect.height) * 2 + 1);
    raycaster.setFromCamera(ndc, host.camera);
    return raycaster.ray.intersectPlane(plane, new Vector3()) ?? undefined;
  };

  let dragging: ((ev: PointerEvent) => void) | undefined;
  let dragged = false;

  const startBlockDrag = (ev: PointerEvent): boolean => {
    const hit = host.blockAt(ev.clientX, ev.clientY);
    if (hit === undefined) return false;
    const target = moveTarget(layout, hit.node, host.visible);
    if (target === undefined) return false;
    const plane = new Plane(new Vector3(0, 1, 0), -hit.y);
    const start = groundPoint(ev, plane);
    if (start === undefined) return false;
    const origin = readPosition(target, hit.node);
    dragging = (move) => {
      const point = groundPoint(move, plane);
      if (point === undefined) return;
      writePosition(target, hit.node, [origin[0] + point.x - start.x, origin[1], origin[2] + point.z - start.z]);
      status.textContent = describe(target, hit.node);
      host.relayout(false);
    };
    status.textContent = describe(target, hit.node);
    return true;
  };

  const startLabelDrag = (ev: PointerEvent): boolean => {
    const el = (ev.target as HTMLElement).closest<HTMLElement>('.vf-lb.vf-ttl, .vf-lb.vf-node');
    const name = el?.dataset.node;
    const label = host.labels.labels.find((l) => l.el === el);
    if (label === undefined || name === undefined) return false;
    const origin = label.offset ?? [label.placed[0] - label.base[0], label.placed[1] - label.base[1]];
    const [x0, y0] = [ev.clientX, ev.clientY];
    dragging = (move) => {
      labelOffsets[name] = [Math.round(origin[0] + move.clientX - x0), Math.round(origin[1] + move.clientY - y0)];
      status.textContent = `labels['${name}'] = ${literal(labelOffsets[name])}`;
      host.relayout(false);
    };
    return true;
  };

  const onDown = (ev: PointerEvent) => {
    if (ev.button !== 0 || panel.contains(ev.target as globalThis.Node)) return;
    const started = ev.target === canvas ? startBlockDrag(ev) : startLabelDrag(ev);
    if (!started) return;
    dragged = false;
    ev.stopPropagation();
    ev.preventDefault();
    root.classList.add('vf-dragging');
  };
  const onMove = (ev: PointerEvent) => {
    if (dragging !== undefined) {
      dragged = true;
      dragging(ev);
      return;
    }
    if (ev.target !== canvas) return;
    canvas.classList.toggle('vf-cursor-move', host.blockAt(ev.clientX, ev.clientY) !== undefined);
  };
  const onUp = () => {
    if (dragging === undefined) return;
    dragging = undefined;
    root.classList.remove('vf-dragging');
    host.relayout(true);
  };
  const onClick = (ev: MouseEvent) => {
    if (!dragged) return;
    dragged = false;
    ev.stopPropagation();
  };
  root.addEventListener('pointerdown', onDown, { capture: true });
  root.addEventListener('click', onClick, { capture: true });
  window.addEventListener('pointermove', onMove);
  window.addEventListener('pointerup', onUp);
  window.addEventListener('pointercancel', onUp);
  host.relayout(true);
  return () => {
    root.removeEventListener('pointerdown', onDown, { capture: true });
    root.removeEventListener('click', onClick, { capture: true });
    window.removeEventListener('pointermove', onMove);
    window.removeEventListener('pointerup', onUp);
    window.removeEventListener('pointercancel', onUp);
    clearTimeout(noteTimer);
    panel.remove();
  };
}
