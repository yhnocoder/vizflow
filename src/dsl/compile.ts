import { type CompiledPage, PAGE_KEYS, PageCompiler, compilePage } from './compile/page';
import type { SectionSources } from './compile/sections';
import { parsePreamble } from './compile/texts';
import packagePreamble from '../models/preamble.sty?raw';
import { checkKeys, isRecord, record } from './compile/util';
import { type LoadOptions, Loader } from './load';
import { sectionRows } from './section-md';
import { type Where, child, fail } from './source';

export type { CompiledPage, PageSequence } from './compile/page';

export async function compileSpec(spec: unknown, file: string, options: LoadOptions): Promise<CompiledPage> {
  const where: Where = { file, path: '', obj: isRecord(spec) ? spec : undefined };
  if (!isRecord(spec)) fail(where, 'a page must be a mapping');
  checkKeys(spec, PAGE_KEYS, where);
  if (typeof spec.title !== 'string') fail(where, "a page needs 'title'");
  if (typeof spec.model !== 'string') fail(where, "a page needs 'model', the name of its root module");
  const loader = new Loader(options);
  const pageFile = options.base;
  const chain = pageFile === undefined ? [] : [pageFile];
  const namespace = await loader.namespace(spec, pageFile, chain, where);
  const modulesWhere = child(where, spec, 'modules');
  const locals = record(spec.modules, modulesWhere, 'modules');
  for (const [name, decl] of Object.entries(locals)) {
    const at = child(modulesWhere, locals, name);
    if (!isRecord(decl)) fail(at, `module '${name}' must be a mapping`);
    if (namespace.has(name)) fail(at, `module '${name}' is already imported or built in`);
    namespace.set(name, { name, file, decl, where: { ...at, obj: decl, key: undefined }, namespace });
  }
  const root = namespace.get(spec.model);
  if (root === undefined) fail(child(where, spec, 'model'), `module '${spec.model}' is neither defined in modules nor imported`);
  const compiler = new PageCompiler(spec, where);
  if (options.config !== undefined) compiler.overrides = options.config;
  compiler.macros = parsePreamble(packagePreamble, 'vizflow/models/preamble.sty');
  if (spec.preamble !== undefined) {
    const at = child(where, spec, 'preamble');
    if (typeof spec.preamble !== 'string') fail(at, 'preamble must be a path to a file of \\newcommand definitions');
    const url = loader.resolve(spec.preamble, pageFile, at);
    for (const [name, macro] of parsePreamble(await loader.text(url), url)) compiler.macros.set(name, macro);
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
  return compilePage(compiler, root, sectionSources);
}
