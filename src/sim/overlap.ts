export type OverlapPhase = 'attention' | 'dispatch' | 'moe' | 'combine';
export type OverlapStream = 'compute' | 'comm';

export interface OverlapDurations {
  attention: number;
  dispatch: number;
  moe: number;
  combine: number;
}

export interface OverlapItem {
  phase: OverlapPhase;
  stream: OverlapStream;
  batch: number;
  layer: number;
  start: number;
  end: number;
}

const ORDER: [OverlapPhase, number][] = [
  ['attention', 0],
  ['dispatch', 0],
  ['attention', 1],
  ['dispatch', 1],
  ['moe', 0],
  ['combine', 0],
  ['moe', 1],
  ['combine', 1],
];

export function streamOf(phase: OverlapPhase): OverlapStream {
  return phase === 'dispatch' || phase === 'combine' ? 'comm' : 'compute';
}

export function overlapSchedule(durations: OverlapDurations, numLayers: number): OverlapItem[] {
  const free: Record<OverlapStream, number> = { compute: 0, comm: 0 };
  const ready = [0, 0];
  const items: OverlapItem[] = [];
  for (let layer = 0; layer < numLayers; layer += 1) {
    for (const [phase, batch] of ORDER) {
      const stream = streamOf(phase);
      const start = Math.max(free[stream], ready[batch]);
      const end = start + durations[phase];
      free[stream] = end;
      ready[batch] = end;
      items.push({ phase, stream, batch, layer, start, end });
    }
  }
  return items;
}
