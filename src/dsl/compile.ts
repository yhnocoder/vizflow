import { PAGE_KEYS } from './compile/keys';
import { type CompiledPage, assemblePage } from './compile/page';
import type { SectionSources } from './compile/sections';
import { Compiler } from './compile/text';
import { NeedDim, type Num } from './compile/types';
import { checkKeys, isRecord, record } from './compile/util';
import { type LoadOptions, Loader, type ModuleDef } from './load';
import { sectionRows } from './section-md';
import { type Where, child, fail } from './source';

export type { CompiledPage, PageSequence } from './compile/page';

function rootDimsOf(page: Record<string, unknown>, where: Where): Map<string, Num> {
  const dimsWhere = child(where, page, 'dims');
  const dims = record(page.dims, dimsWhere, 'dims');
  const real = isRecord(page.real) ? page.real : {};
  const out = new Map<string, Num>();
  for (const [name, value] of Object.entries(dims)) {
    if (typeof value !== 'number' || !Number.isInteger(value) || value <= 0) fail(child(dimsWhere, dims, name), `dimension '${name}' must be a positive integer`);
    const realValue = real[name];
    out.set(name, { toy: value, real: typeof realValue === 'number' ? realValue : value });
  }
  return out;
}

export async function compileSpec(spec: unknown, file: string, options: LoadOptions): Promise<CompiledPage> {
  const where: Where = { file, path: '', obj: isRecord(spec) ? spec : undefined };
  if (!isRecord(spec)) fail(where, 'a page must be a mapping');
  checkKeys(spec, PAGE_KEYS, where);
  if (typeof spec.title !== 'string') fail(where, "a page needs 'title'");
  const loader = new Loader(options);
  const pageFile = options.base;
  const namespace = await loader.namespace(spec, pageFile, pageFile === undefined ? [] : [pageFile], where);
  const modulesWhere = child(where, spec, 'modules');
  const locals = record(spec.modules, modulesWhere, 'modules');
  for (const [name, decl] of Object.entries(locals)) {
    const at = child(modulesWhere, locals, name);
    if (!isRecord(decl)) fail(at, `module '${name}' must be a mapping`);
    if (namespace.has(name)) fail(at, `module '${name}' is already imported`);
    namespace.set(name, { name, file, decl, where: { ...at, obj: decl, key: undefined }, namespace });
  }
  for (const [name, decl] of Object.entries(locals)) {
    await loader.directUses(decl as Record<string, unknown>, pageFile, pageFile === undefined ? [] : [pageFile], child(modulesWhere, locals, name), namespace);
  }
  const sectionSources: SectionSources = new Map();
  if (Array.isArray(spec.sections)) {
    const sectionsWhere = child(where, spec, 'sections');
    for (const [i, section] of spec.sections.entries()) {
      if (!isRecord(section) || section.src === undefined) continue;
      const at = child(child(sectionsWhere, spec.sections, i), section, 'src');
      if (typeof section.src !== 'string') fail(at, 'src must be a path to a Markdown file');
      const url = loader.resolve(section.src, pageFile, at);
      sectionSources.set(section, { file: url, rows: sectionRows(await loader.text(url), url) });
    }
  }
  const rootDef: ModuleDef = { name: String(spec.title), file, decl: spec, where, namespace };
  const rootDims = rootDimsOf(spec, where);
  const extra = new Map<string, Num>();
  for (;;) {
    try {
      return assemblePage(new Compiler(spec, where, rootDef, rootDims, extra), sectionSources);
    } catch (error) {
      if (!(error instanceof NeedDim)) throw error;
      if (extra.has(error.name)) throw new Error(`dsl: dimension ${error.name} could not be registered`);
      extra.set(error.name, error.value);
    }
  }
}
