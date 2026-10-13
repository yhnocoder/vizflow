import '../autograd/ops';
import { type CompiledPage, compileSpec } from './compile';
import { defaultRead } from './load';
import { parseYaml } from './source';
import type { PageFile } from './types';

export interface CompileOptions {
  base?: string;
  read?: (url: string) => Promise<string>;
}

export type { CompiledPage } from './compile';
export { DslError, parseYaml } from './source';
export type * from './types';

export function compilePage(spec: PageFile, options: CompileOptions = {}): Promise<CompiledPage> {
  return compileSpec(spec, options.base ?? '<page>', options);
}

export async function loadPage(url: string, options: CompileOptions = {}): Promise<CompiledPage> {
  const absolute = typeof location === 'undefined' ? url : new URL(url, location.href).href;
  const read = options.read ?? defaultRead;
  const spec = parseYaml(await read(absolute), absolute);
  return compileSpec(spec, absolute, { ...options, base: absolute, read });
}
