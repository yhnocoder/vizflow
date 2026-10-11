export interface ServingRequest {
  id: string;
  arrive: number;
  prompt: number;
  shared?: number;
  group?: string;
  out: number;
}

export type Policy = 'fcfs' | 'lpm';

export interface ServingConfig {
  requests: ServingRequest[];
  budget: number;
  kvTokens: number;
  overlap: boolean;
  policy?: Policy;
  maxSeqs?: number;
  maxSteps?: number;
}

export type RequestStatus = 'pending' | 'waiting' | 'running' | 'preempted' | 'finished';

export interface BatchRow {
  id: string;
  from: number;
  n: number;
  duplicate: number[];
  sampled: boolean;
}

export interface RequestView {
  id: string;
  status: RequestStatus;
  prompt: number;
  outputs: number;
  inflight: number;
  computed: number;
  hit: number;
}

export type EventKind = 'launch' | 'process' | 'preempt' | 'retract' | 'resume' | 'blocked' | 'finish';

export interface EngineEvent {
  kind: EventKind;
  ids: string[];
  batch?: number;
  hit?: number;
}

export type BatchMode = 'prefill' | 'mixed' | 'decode' | 'idle';

export type GraphKind = 'full' | 'piecewise' | 'decode' | 'prefill' | 'eager' | 'none';

export interface EngineStep {
  step: number;
  mode: BatchMode;
  rows: BatchRow[];
  tokens: number;
  graph: GraphKind;
  padded: number;
  requests: RequestView[];
  events: EngineEvent[];
  kvUsed: number;
  newTokenRatio?: number;
}

export interface RequestSummary {
  id: string;
  firstToken: number | null;
  finished: number | null;
  prefillTokens: number;
  decodeTokens: number;
  duplicateTokens: number;
  evictions: number;
}

export type EngineName = 'vllm' | 'sglang';

export interface EngineRun {
  engine: EngineName;
  kvCapacity: number;
  steps: EngineStep[];
  requests: RequestSummary[];
}
