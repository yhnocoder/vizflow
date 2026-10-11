import type { Tensor } from '../tensor';
import type { Dtype } from './dtype';
import type { Node } from './node';

export type Pattern = 'elementwise' | 'reduce' | 'dense' | 'contract' | 'select';

export interface OpCtx {
  node: Node;
  inputs: Tensor[];
  out: Tensor;
  attrs: object;
}

export interface ForwardTexArgs {
  out: string;
  in: string[];
  attrs: object;
}

export interface LocalTexArgs {
  out: string;
  in: string;
  ins: string[];
  arg: number;
  attrs: object;
  outIndex: number[];
  inIndex: number[];
  ctx: OpCtx;
  elem(arg: number, index: number[]): string;
}

export interface OpExplain {
  pattern: Pattern;
  local(ctx: OpCtx, out: number[], arg: number, input: number[]): number;
  tex: {
    forward(s: ForwardTexArgs): string;
    local(s: LocalTexArgs): string;
  };
  selected?(ctx: OpCtx, out: number[]): { arg: number; index: number[] }[];
  note?(ctx: OpCtx, tex: { in: string[] }): string;
}

export interface OpDef<A extends object = Record<string, unknown>> {
  name: string;
  args: string[];
  tensors: string[];
  list?: string;
  rest?: string;
  scalars?: string[];
  sizes?: string[];
  defaults?: Partial<A>;
  shape(inputs: number[][], attrs: A): number[];
  forward(inputs: Tensor[], attrs: A): Tensor;
  backward(gradOut: Tensor, inputs: Tensor[], out: Tensor, attrs: A): (Tensor | null)[];
  differentiable?: boolean[];
  dtype?: Dtype;
  explain?: OpExplain;
}

export type AnyOpDef = OpDef<any>;
