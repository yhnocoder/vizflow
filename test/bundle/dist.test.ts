import { existsSync, readFileSync, rmSync } from 'node:fs';
import { dirname, join, normalize } from 'node:path';
import { afterAll, beforeAll, describe, expect, test } from 'vitest';
import packageJson from '../../package.json';
import { buildDistToTemp } from './build';

const libraries = {
  three: ['__THREE__', 'WebGLRenderer'],
  solid: ['solid-track', 'solid-js/web'],
  katex: ['katex-html', 'KaTeX parse error'],
};

type Library = keyof typeof libraries;

let workspace = '';
let dist = '';

beforeAll(() => {
  ({ workspace, dist } = buildDistToTemp());
}, 180_000);

afterAll(() => {
  if (workspace) rmSync(workspace, { recursive: true, force: true });
});

function staticGraph(entry: string): string[] {
  const seen = new Set<string>();
  const pending = [entry];
  while (pending.length > 0) {
    const file = pending.pop()!;
    if (seen.has(file)) continue;
    seen.add(file);
    const code = readFileSync(join(dist, file), 'utf8');
    for (const [, path] of code.matchAll(/^(?:import|export)\b[^'"]*?["']\.\/([^"']+\.js)["'];?$/gm)) {
      pending.push(path);
    }
  }
  return [...seen].sort();
}

function graphCode(entry: string): string {
  return staticGraph(entry)
    .map((file) => readFileSync(join(dist, file), 'utf8'))
    .join('\n');
}

function bundledLibraries(entry: string): Library[] {
  const code = graphCode(entry);
  return (Object.keys(libraries) as Library[]).filter((library) => libraries[library].some((marker) => code.includes(marker)));
}

function typeGraph(entry: string): string[] {
  const seen = new Set<string>();
  const pending = [entry];
  while (pending.length > 0) {
    const file = pending.pop()!;
    if (seen.has(file)) continue;
    seen.add(file);
    for (const [, path] of readFileSync(file, 'utf8').matchAll(/from '(\.[^']+)'/g)) {
      const base = normalize(join(dirname(file), path));
      const next = [`${base}.d.ts`, join(base, 'index.d.ts')].find((candidate) => existsSync(candidate));
      if (next !== undefined) pending.push(next);
    }
  }
  return [...seen];
}

const exportsMap = packageJson.exports as Record<string, { types: string; import: string }>;

describe('build output', () => {
  test('core, explain and steps load none of three.js, Solid and KaTeX', () => {
    expect(staticGraph('core.js').length).toBeGreaterThan(1);
    expect(bundledLibraries('core.js')).toEqual([]);
    expect(bundledLibraries('explain.js')).toEqual([]);
    expect(bundledLibraries('steps.js')).toEqual([]);
  });

  test('ui loads Solid and KaTeX but not three.js', () => {
    expect(bundledLibraries('ui.js')).toEqual(['solid', 'katex']);
  });

  test('scene3d loads three.js and formula loads only KaTeX', () => {
    expect(bundledLibraries('scene3d.js')).toContain('three');
    expect(bundledLibraries('formula.js')).toEqual(['katex']);
  });

  test('ui, formula and scene3d keep the modules that inject their styles', () => {
    expect(graphCode('ui.js')).toContain('vf-theme-style');
    expect(graphCode('ui.js')).toContain('vf-ui-style');
    expect(graphCode('ui.js')).toContain('vf-katex-style');
    expect(graphCode('formula.js')).toContain('vf-katex-style');
    expect(graphCode('theme.js')).toContain('vf-theme-style');
    expect(graphCode('scene3d.js')).toContain('vf-theme-style');
  });

  test('the KaTeX symbol leftthreetimes is not taken for three.js', () => {
    expect(graphCode('formula.js')).toContain('leftthreetimes');
    expect(bundledLibraries('formula.js')).not.toContain('three');
  });

  test('every package entry has its module and type declarations', () => {
    expect(Object.keys(exportsMap).sort()).toEqual(['./core', './explain', './formula', './scene3d', './steps', './theme', './ui']);
    for (const [entry, { types, import: module }] of Object.entries(exportsMap)) {
      expect(existsSync(join(workspace, module)), `${entry} module`).toBe(true);
      expect(existsSync(join(workspace, types)), `${entry} types`).toBe(true);
    }
  });

  test('public type declarations do not import the bundled libraries', () => {
    for (const [entry, { types }] of Object.entries(exportsMap)) {
      const importing = typeGraph(join(workspace, types)).filter((file) => /from '(three|solid-js|katex|@fontsource)/.test(readFileSync(file, 'utf8')));
      expect(importing, entry).toEqual([]);
    }
  });

  test('fonts are emitted to dist/fonts', () => {
    for (const font of ['KaTeX_Main-Regular.woff2', 'geist-latin-400-normal.woff2', 'geist-mono-latin-500-normal.woff2', 'source-serif-4-latin-400-italic.woff2']) {
      expect(existsSync(join(dist, 'fonts', font)), font).toBe(true);
    }
  });
});
