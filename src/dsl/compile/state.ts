import { type Graph, type Node, graph } from '../../autograd';
import type { Nested } from '../../tensor';
import { type Where, fail } from '../source';
import type { Env, Num } from './types';

const INIT_DIGITS = 2;

function rng(seed: number): () => number {
  let a = seed;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export interface GraphSetup {
  dims: Map<string, Num>;
  real: ({ name: string } & Record<string, number | string>) | undefined;
  dtype: string;
  seed: number;
}

export class CompilerState {
  g!: Graph;
  readonly nodes = new Map<string, Node>();
  private random: () => number = rng(0);
  private realPage = false;
  private readonly namedDims = new Map<string, string>();
  private readonly derivedDims = new Set<string>();

  setupGraph(setup: GraphSetup, where: Where): void {
    const dims: Record<string, number> = {};
    for (const [name, value] of setup.dims) {
      dims[name] = value.toy;
      this.namedDims.set(`${value.toy}:${value.real}`, name);
    }
    this.realPage = setup.real !== undefined;
    try {
      this.g = graph({ dims, ...(setup.real === undefined ? {} : { real: setup.real }), dtype: setup.dtype });
    } catch (error) {
      fail(where, (error as Error).message);
    }
    this.random = rng(setup.seed);
  }

  dimSpec(value: Num): number | string {
    if (!this.realPage) return value.toy;
    if (value.dim !== undefined) return value.dim;
    if (value.toy === value.real && value.derive === undefined) return value.toy;
    const key = `${value.toy}:${value.real}`;
    const name = this.namedDims.get(key);
    if (name !== undefined && (value.derive === undefined || this.derivedDims.has(name))) return name;
    const derived = value.derive === undefined ? key : this.derivedName(key);
    this.g.addDim(derived, value.toy, value.real, value.derive);
    if (value.derive !== undefined) this.derivedDims.add(derived);
    if (name === undefined) this.namedDims.set(key, derived);
    return derived;
  }

  private derivedName(key: string): string {
    if (this.g.dims[key] === undefined) return key;
    let n = 2;
    while (this.g.dims[`${key}#${n}`] !== undefined) n++;
    return `${key}#${n}`;
  }

  initValues(kind: string, shape: number[]): Nested {
    const size = shape.reduce((a, b) => a * b, 1);
    const round = (x: number) => Math.round(x * 10 ** INIT_DIGITS) / 10 ** INIT_DIGITS;
    const flat = Array.from({ length: size }, () => {
      if (kind === 'ones') return 1;
      if (kind === 'zeros') return 0;
      if (kind === 'normal') {
        const u = 1 - this.random();
        const v = this.random();
        return round(Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v));
      }
      return round(this.random() * 2 - 1);
    });
    const nest = (offset: number, dims: number[]): Nested =>
      dims.length === 0 ? flat[offset] : Array.from({ length: dims[0] }, (_, i) => nest(offset + i * dims.slice(1).reduce((a, b) => a * b, 1), dims.slice(1)));
    return nest(0, shape);
  }

  fail(env: Env, message: string, offset?: number): never {
    return fail(env.where, message, offset);
  }
}
