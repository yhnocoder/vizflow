import { type InstancedMesh, type Object3D, Raycaster, Vector2 } from 'three';
import type { Node } from '../autograd';
import type { SceneController } from './controller';
import type { Block } from './layout';
import { type Renderable, renderableFor, shown } from './renderable';

const CLICK_SLOP = 5;

const raycaster = new Raycaster();
const pointer = new Vector2();

function aim(c: SceneController, clientX: number, clientY: number): void {
  const rect = c.renderer.domElement.getBoundingClientRect();
  pointer.set(((clientX - rect.left) / rect.width) * 2 - 1, -((clientY - rect.top) / rect.height) * 2 + 1);
  raycaster.setFromCamera(pointer, c.camera);
}

export function blockAt(c: SceneController, clientX: number, clientY: number): { node: Node; y: number } | undefined {
  aim(c, clientX, clientY);
  const owners = new Map<Object3D, Renderable>();
  for (const renderable of c.renderables) {
    for (const object of renderable.pickables) owners.set(object, renderable);
    if (renderable.plate !== undefined) owners.set(renderable.plate.mesh, renderable);
  }
  const hits = raycaster.intersectObjects([...owners.keys()], false);
  if (hits.length === 0) return undefined;
  const block = owners.get(hits[0].object)!.block;
  return { node: block.node, y: block.elements[0].y };
}

function hit(c: SceneController, ev: PointerEvent): { renderable: Renderable; index: number } | undefined {
  aim(c, ev.clientX, ev.clientY);
  const pickables = c.renderables.filter((r) => c.state.revealed(r.block.node) && shown(r.block)).flatMap((r) => r.pickables);
  const hits = raycaster.intersectObjects(pickables, false);
  if (hits.length === 0) return undefined;
  const object = hits[0].object;
  const block = object.userData.block as Block;
  const renderable = renderableFor(c, block)!;
  const index = (object as InstancedMesh).isInstancedMesh ? (hits[0].instanceId ?? 0) : (object.userData.element as number);
  return { renderable, index };
}

function setTarget(c: SceneController, renderable: Renderable, index: number): void {
  const { block } = renderable;
  const label = c.aligned.get(block.node);
  if (label === undefined) return;
  const element = block.elements[index];
  const data = Float64Array.from(label.value.data);
  const row = Math.min(element.row, data.length - 1);
  const sequence = c.options.sequence;
  const ids = sequence?.get();
  const shift = ids === undefined ? -1 : ids.findIndex((_, k) => k + data.length <= ids.length && data.every((id, i) => Math.round(id) === ids[i + k]));
  if (sequence !== undefined && ids !== undefined && shift >= 0) {
    ids[row + shift] = element.col;
    sequence.set(ids);
  } else {
    data[row] = element.col;
    label.set(label.shape.length === 0 ? data[0] : Array.from(data));
  }
  if (block.batched) c.options.steps?.setFocus(element.row);
}

export function bindPointer(c: SceneController): () => void {
  const canvas = c.renderer.domElement;
  const { popups, edgeLines } = c;
  let downAt: [number, number] | undefined;
  const onMove = (ev: PointerEvent) => {
    const found = hit(c, ev);
    const rect = c.root.getBoundingClientRect();
    edgeLines.setBar(found);
    if (found === undefined) {
      popups.hideTip();
      const edge = edgeLines.edgeAt(ev.clientX - rect.left, ev.clientY - rect.top);
      edgeLines.setHover(edge);
      canvas.classList.toggle('vf-cursor-help', edge !== undefined);
      canvas.classList.remove('vf-cursor-pointer');
      if (edge !== undefined) popups.showEdgeCard(edge, ev);
      else popups.hideUnpinned();
      return;
    }
    edgeLines.setHover(undefined);
    popups.hideUnpinned();
    const clickable = c.aligned.has(found.renderable.block.node) && c.state.dir === 'forward';
    canvas.classList.toggle('vf-cursor-pointer', clickable);
    canvas.classList.remove('vf-cursor-help');
    popups.showTip(found.renderable, found.index, ev);
  };
  const onLeave = () => {
    popups.hideTip();
    edgeLines.setBar(undefined);
    edgeLines.setHover(undefined);
    popups.hideUnpinned();
  };
  const onDown = (ev: PointerEvent) => {
    downAt = [ev.clientX, ev.clientY];
  };
  const onUp = (ev: PointerEvent) => {
    if (downAt === undefined) return;
    const moved = Math.hypot(ev.clientX - downAt[0], ev.clientY - downAt[1]);
    downAt = undefined;
    if (moved > CLICK_SLOP) return;
    const found = hit(c, ev);
    if (found === undefined || !c.aligned.has(found.renderable.block.node) || c.state.dir !== 'forward') return;
    setTarget(c, found.renderable, found.index);
    popups.showTip(found.renderable, found.index, ev);
  };
  const onWheel = (ev: WheelEvent) => c.rig.onWheel(ev);
  canvas.addEventListener('pointermove', onMove);
  canvas.addEventListener('pointerleave', onLeave);
  canvas.addEventListener('pointerdown', onDown);
  canvas.addEventListener('pointerup', onUp);
  c.root.addEventListener('wheel', onWheel, { passive: false });
  return () => {
    canvas.removeEventListener('pointermove', onMove);
    canvas.removeEventListener('pointerleave', onLeave);
    canvas.removeEventListener('pointerdown', onDown);
    canvas.removeEventListener('pointerup', onUp);
    c.root.removeEventListener('wheel', onWheel);
  };
}
