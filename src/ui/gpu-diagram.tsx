import { For, Index, Show, createEffect, createMemo, createSignal, createUniqueId, onCleanup, type Accessor } from 'solid-js';
import { render } from 'solid-js/web';
import {
  GPU_LINE_HEIGHT,
  GPU_NARROW_WIDTH,
  GPU_TITLE_HEIGHT,
  GPU_WIDE_WIDTH,
  gpuUnitIds,
  layoutGpuDiagram,
  pathLength,
  pointAt,
  type GpuBadgeLine,
  type GpuBox,
  type GpuColor,
  type GpuDiagramProps,
  type GpuKind,
  type Point,
} from './gpu-layout';
import { resolveElement } from './mount';
import type { Mounted } from './mounted';

export interface GpuDiagram extends Mounted {
  set(props: GpuDiagramProps): void;
  replay(): void;
}

const GPU_KINDS: GpuKind[] = ['compute', 'nvlink', 'rdma', 'reduce', 'copy'];
const SERIES = 6;
const NARROW_BELOW = 600;
const PACKET_MS = 1400;
const PACKET = 10;
const CHIP = 10;

const colorClass = (color: GpuColor) => (typeof color === 'number' ? `vf-gpu-s${((color % SERIES) + SERIES) % SERIES}` : `vf-gpu-${color}`);
const pointsAttr = (points: Point[]) => points.map((p, i) => `${i === 0 ? 'M' : 'L'}${p.x.toFixed(1)} ${p.y.toFixed(1)}`).join(' ');

function reducedMotion(): boolean {
  return typeof matchMedia === 'function' && matchMedia('(prefers-reduced-motion: reduce)').matches;
}

function BadgeLine(props: { line: GpuBadgeLine; x: number; y: number }) {
  const chips = () => (typeof props.line === 'string' ? [] : props.line.chips);
  const text = () => (typeof props.line === 'string' ? props.line : props.line.text);
  return (
    <>
      <Index each={chips()}>
        {(chip, j) => <rect class={`vf-gpu-chip ${colorClass(chip())}`} x={props.x + j * (CHIP + 3)} y={props.y - CHIP + 1} width={CHIP} height={CHIP} rx={2} />}
      </Index>
      <text class="vf-gpu-line" x={props.x + chips().length * (CHIP + 3) + (chips().length > 0 ? 3 : 0)} y={props.y}>
        {text()}
      </text>
    </>
  );
}

function Unit(props: { box: GpuBox; state: string; frame: boolean }) {
  const b = () => props.box;
  return (
    <g class={`vf-gpu-unit ${props.state}`} classList={{ 'vf-gpu-frame': props.frame }} data-unit={b().id}>
      <rect class="vf-gpu-box" x={b().x} y={b().y} width={b().w} height={b().h} rx={props.frame ? 12 : 6} />
      <Show when={b().text === 'vertical'}>
        <text class="vf-gpu-vertical" x={b().x + b().w / 2} y={b().y + b().h / 2} transform={`rotate(-90 ${b().x + b().w / 2} ${b().y + b().h / 2})`} text-anchor="middle" dominant-baseline="central">
          {b().title}
        </text>
      </Show>
      <Show when={b().text === 'full'}>
        <text class="vf-gpu-title" x={b().x + (props.frame ? 12 : 8)} y={b().y + (props.frame ? 22 : 16)}>
          {b().title}
        </text>
        <Show when={b().note !== ''}>
          <Show
            when={props.frame}
            fallback={
              <text class="vf-gpu-note" x={b().x + b().w - 8} y={b().y + 16} text-anchor="end">
                {b().note}
              </text>
            }
          >
            <text class="vf-gpu-note" x={b().x + 12} y={b().y + 40}>
              {b().note}
            </text>
          </Show>
        </Show>
        <Index each={b().lines}>{(line, i) => <BadgeLine line={line()} x={b().x + 8} y={b().y + GPU_TITLE_HEIGHT + i * GPU_LINE_HEIGHT + 8} />}</Index>
      </Show>
    </g>
  );
}

function GpuDiagramView(props: { spec: Accessor<GpuDiagramProps>; width: Accessor<number>; replays: Accessor<number> }) {
  const markerBase = `vf-gpu-${createUniqueId()}`;
  const layout = createMemo(() => {
    const spec = props.spec();
    const ids = new Set(gpuUnitIds(spec));
    for (const [field, map] of [['highlight', spec.highlight ?? {}], ['badges', spec.badges ?? {}]] as const) {
      for (const id of Object.keys(map)) if (!ids.has(id)) throw new Error(`gpu diagram: ${field} names unknown unit '${id}' at level ${spec.level}`);
    }
    const width = props.width();
    return layoutGpuDiagram(spec, width > 0 && width < NARROW_BELOW ? GPU_NARROW_WIDTH : GPU_WIDE_WIDTH);
  });
  const state = (id: string) => {
    const highlight = props.spec().highlight ?? {};
    const kind = highlight[id];
    if (kind !== undefined) return `vf-gpu-hl vf-gpu-${kind}`;
    return Object.keys(highlight).length > 0 ? 'vf-gpu-muted' : '';
  };
  const flows = createMemo(() => {
    const l = layout();
    return (props.spec().flows ?? []).map((flow) => {
      const points = l.route(flow);
      return { flow, points, mid: pointAt(points, 0.5) };
    });
  });
  const packets = createMemo(() => {
    const spec = props.spec();
    const l = layout();
    const drawn = flows();
    return (spec.packets ?? []).map((packet, i) => {
      if (typeof packet.flow === 'number') {
        const found = drawn[packet.flow];
        if (found === undefined) throw new Error(`gpu diagram: packet ${i} names flow ${packet.flow}, but there are ${drawn.length} flows`);
        return { points: found.points, color: packet.color ?? found.flow.kind, delay: packet.delay ?? 0 };
      }
      return { points: l.route(packet.flow), color: packet.color ?? packet.flow.kind, delay: packet.delay ?? 0 };
    });
  });
  const packetEls: SVGRectElement[] = [];
  let frame: number | undefined;
  const place = (el: SVGRectElement, points: Point[], fraction: number) => {
    const p = pointAt(points, fraction);
    el.setAttribute('x', (p.x - PACKET / 2).toFixed(1));
    el.setAttribute('y', (p.y - PACKET / 2).toFixed(1));
  };
  const stop = () => {
    if (frame !== undefined) cancelAnimationFrame(frame);
    frame = undefined;
  };
  createEffect(() => {
    props.replays();
    const list = packets();
    stop();
    const still = reducedMotion() || typeof requestAnimationFrame !== 'function';
    list.forEach((p, i) => place(packetEls[i], p.points, still ? 1 : 0));
    if (still || list.length === 0) return;
    const start = performance.now();
    const step = (now: number) => {
      let busy = false;
      list.forEach((p, i) => {
        const f = pathLength(p.points) === 0 ? 1 : (now - start - p.delay) / PACKET_MS;
        if (f < 1) busy = true;
        place(packetEls[i], p.points, f);
      });
      frame = busy ? requestAnimationFrame(step) : undefined;
    };
    frame = requestAnimationFrame(step);
  });
  onCleanup(stop);
  return (
    <svg class="vf-gpu" viewBox={`0 0 ${layout().width} ${layout().height}`} role="img" aria-label="GPU 结构简图">
      <defs>
        <For each={GPU_KINDS}>
          {(kind) => (
            <marker id={`${markerBase}-${kind}`} class={`vf-gpu-${kind}`} viewBox="0 0 10 10" refX="9" refY="5" markerUnits="userSpaceOnUse" markerWidth="10" markerHeight="10" orient="auto-start-reverse">
              <path class="vf-gpu-arrow" d="M0 0L10 5L0 10Z" />
            </marker>
          )}
        </For>
      </defs>
      <For each={layout().frames}>{(b) => <Unit box={b} state={state(b.id)} frame={true} />}</For>
      <For each={layout().wires}>{(points) => <path class="vf-gpu-wire" d={pointsAttr(points)} />}</For>
      <For each={layout().boxes}>{(b) => <Unit box={b} state={state(b.id)} frame={false} />}</For>
      <For each={layout().texts}>
        {(t) => (
          <text class="vf-gpu-label" x={t.x} y={t.y} text-anchor={t.anchor}>
            {t.text}
          </text>
        )}
      </For>
      <Show when={layout().tiles}>
        {(tiles) => (
          <g class="vf-gpu-tiles">
            <For each={tiles().cells}>
              {(cell) => (
                <g class={`vf-gpu-tile vf-gpu-tile-${cell.state}`}>
                  <rect x={cell.x} y={cell.y} width={cell.w} height={cell.h} rx={4} />
                  <Index each={cell.lines}>
                    {(line, i) => (
                      <text x={cell.x + cell.w / 2} y={cell.y + (cell.lines.length === 1 ? cell.h / 2 + 4 : 16 + i * 14)} text-anchor="middle" classList={{ 'vf-gpu-tile-sub': i > 0 }}>
                        {line()}
                      </text>
                    )}
                  </Index>
                </g>
              )}
            </For>
            <Show when={tiles().group}>
              {(group) => (
                <g class="vf-gpu-group">
                  <rect x={group().x} y={group().y} width={group().w} height={group().h} rx={6} />
                  <Show when={group().label}>
                    <text x={group().x + group().w / 2} y={group().labelY} text-anchor="middle">
                      {group().label}
                    </text>
                  </Show>
                </g>
              )}
            </Show>
          </g>
        )}
      </Show>
      <For each={flows()}>
        {(f) => (
          <g class={`vf-gpu-flow vf-gpu-${f.flow.kind}`}>
            <path d={pointsAttr(f.points)} marker-end={`url(#${markerBase}-${f.flow.kind})`} />
            <Show when={f.flow.label}>
              <text class="vf-gpu-flowlabel" x={f.mid.x} y={f.mid.y - 6} text-anchor="middle">
                {f.flow.label}
              </text>
            </Show>
          </g>
        )}
      </For>
      <Index each={packets()}>{(p, i) => <rect ref={(el) => (packetEls[i] = el)} class={`vf-gpu-packet ${colorClass(p().color)}`} width={PACKET} height={PACKET} rx={2} x={-PACKET} y={-PACKET} />}</Index>
    </svg>
  );
}

export function mountGpuDiagram(el: string | HTMLElement, props: GpuDiagramProps): GpuDiagram {
  const target = resolveElement(el, 'mountGpuDiagram');
  const [spec, setSpec] = createSignal(props);
  const [width, setWidth] = createSignal(target.clientWidth);
  const [replays, setReplays] = createSignal(0);
  const observer = typeof ResizeObserver === 'function' ? new ResizeObserver(() => setWidth(target.clientWidth)) : undefined;
  observer?.observe(target);
  const dispose = render(() => <GpuDiagramView spec={spec} width={width} replays={replays} />, target);
  return {
    set: (next) => setSpec(() => next),
    replay: () => setReplays((n) => n + 1),
    dispose: () => {
      observer?.disconnect();
      dispose();
    },
  };
}
