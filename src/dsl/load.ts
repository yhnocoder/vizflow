import { type Where, child, fail, parseYaml } from './source';

const bundled: Record<string, string> = import.meta.glob(['../nn/*.yaml', '../models/*/*.yaml'], { query: '?raw', import: 'default', eager: true });

export const PACKAGE_PREFIX = 'vizflow/';
const BUILTINS: Readonly<Record<string, string>> = { Linear: `${PACKAGE_PREFIX}nn/linear`, Embedding: `${PACKAGE_PREFIX}nn/embedding` };

export interface ModuleDef {
  name: string;
  file: string;
  decl: Record<string, unknown>;
  where: Where;
  namespace: Map<string, ModuleDef>;
}

export interface LoadOptions {
  base?: string;
  read?: (url: string) => Promise<string>;
}

function bundledSource(name: string): string | undefined {
  return bundled[`../${name}.yaml`];
}

function packageRelative(ref: string, from: string): string | undefined {
  if (!ref.startsWith('./') && !ref.startsWith('../')) return undefined;
  const parts = from.slice(PACKAGE_PREFIX.length).split('/').slice(0, -1);
  for (const part of ref.replace(/\.yaml$/, '').split('/')) {
    if (part === '..') parts.pop();
    else if (part !== '.') parts.push(part);
  }
  return `${PACKAGE_PREFIX}${parts.join('/')}`;
}

export async function defaultRead(url: string): Promise<string> {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`cannot read ${url}: HTTP ${response.status}`);
  return response.text();
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

export class Loader {
  private readonly cache = new Map<string, Promise<ModuleDef>>();
  private readonly read: (url: string) => Promise<string>;

  constructor(options: LoadOptions) {
    this.read = options.read ?? defaultRead;
  }

  resolve(ref: string, from: string | undefined, where: Where): string {
    if (ref.startsWith(PACKAGE_PREFIX)) {
      if (bundledSource(ref.slice(PACKAGE_PREFIX.length)) === undefined) fail(where, `module '${ref}' is not part of the package`);
      return ref;
    }
    if (from !== undefined && from.startsWith(PACKAGE_PREFIX)) {
      const inside = packageRelative(ref, from);
      if (inside === undefined || bundledSource(inside.slice(PACKAGE_PREFIX.length)) === undefined) fail(where, `cannot resolve '${ref}' from package module '${from}'`);
      return inside;
    }
    if (from === undefined) {
      if (!/^[a-z]+:/.test(ref)) fail(where, `cannot resolve '${ref}': the page has no base URL; pass options.base`);
      return ref;
    }
    return new URL(ref, from).href;
  }

  text(url: string): Promise<string> {
    return this.read(url);
  }

  async module(url: string, chain: readonly string[], where: Where): Promise<ModuleDef> {
    if (chain.includes(url)) fail(where, `circular import: ${[...chain, url].join(' -> ')}`);
    const cached = this.cache.get(url);
    if (cached !== undefined) return cached;
    const loading = this.loadModule(url, chain);
    this.cache.set(url, loading);
    return loading;
  }

  private async loadModule(url: string, chain: readonly string[]): Promise<ModuleDef> {
    const text = url.startsWith(PACKAGE_PREFIX) ? (bundledSource(url.slice(PACKAGE_PREFIX.length)) as string) : await this.read(url);
    const decl = parseYaml(text, url);
    const where: Where = { file: url, path: '', obj: isRecord(decl) ? decl : undefined };
    if (!isRecord(decl)) fail(where, 'a module file must be a mapping');
    if (typeof decl.module !== 'string') fail(where, "a module file needs 'module: <name>'");
    const namespace = await this.namespace(decl, url, [...chain, url], where);
    return { name: decl.module, file: url, decl, where, namespace };
  }

  async namespace(decl: Record<string, unknown>, file: string | undefined, chain: readonly string[], where: Where): Promise<Map<string, ModuleDef>> {
    const namespace = new Map<string, ModuleDef>();
    const imports = decl.import;
    if (imports !== undefined) {
      const importsWhere = child(where, decl, 'import');
      if (!isRecord(imports)) fail(importsWhere, 'import must map module names to references');
      for (const [name, ref] of Object.entries(imports)) {
        const at = child(importsWhere, imports, name);
        if (typeof ref !== 'string') fail(at, 'a module reference must be a string');
        namespace.set(name, await this.module(this.resolve(ref, file, at), chain, at));
      }
    }
    for (const [name, ref] of Object.entries(BUILTINS)) {
      if (namespace.has(name) || chain.includes(ref)) continue;
      namespace.set(name, await this.module(ref, chain, where));
    }
    return namespace;
  }
}
