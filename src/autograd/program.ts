import type { Tensor } from '../tensor';
import { opDef } from './registry';

export type ProgramStep = { k: 'read'; name: string; grad: boolean } | { k: 'op'; op: string; inputs: number[]; attrs: Record<string, unknown> };

export interface Program {
  steps: ProgramStep[];
  out: number;
}

export type ReadValue = (name: string, grad: boolean) => Tensor;

export function runOp(op: string, inputs: Tensor[], attrs: Record<string, unknown>): Tensor {
  const def = opDef(op);
  if (def === undefined) throw new Error(`unknown op '${op}'`);
  def.shape(inputs.map((t) => [...t.shape]), attrs);
  return def.forward(inputs, attrs);
}

export function runProgram(program: Program, read: ReadValue): Tensor {
  const values: Tensor[] = [];
  for (const step of program.steps) values.push(step.k === 'read' ? read(step.name, step.grad) : runOp(step.op, step.inputs.map((i) => values[i]), step.attrs));
  return values[program.out];
}

export class ProgramBuilder {
  readonly steps: ProgramStep[] = [];

  read(name: string, grad: boolean): number {
    const found = this.steps.findIndex((s) => s.k === 'read' && s.name === name && s.grad === grad);
    if (found >= 0) return found;
    this.steps.push({ k: 'read', name, grad });
    return this.steps.length - 1;
  }

  op(op: string, inputs: number[], attrs: Record<string, unknown>): number {
    this.steps.push({ k: 'op', op, inputs, attrs });
    return this.steps.length - 1;
  }

  program(out: number): Program {
    const used = new Set<number>();
    const visit = (i: number) => {
      if (used.has(i)) return;
      used.add(i);
      const step = this.steps[i];
      if (step.k === 'op') step.inputs.forEach(visit);
    };
    visit(out);
    const kept = [...used].sort((a, b) => a - b);
    const at = new Map(kept.map((i, k) => [i, k]));
    const steps = kept.map((i): ProgramStep => {
      const step = this.steps[i];
      return step.k === 'read' ? step : { ...step, inputs: step.inputs.map((j) => at.get(j) as number) };
    });
    return { steps, out: at.get(out) as number };
  }
}
