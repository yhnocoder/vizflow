import { resolve } from 'node:path';
import type { Plugin } from 'vite';
import { defineConfig } from 'vitest/config';
import solid from 'vite-plugin-solid';
import dts from 'vite-plugin-dts';

const root = import.meta.dirname;
const src = (path: string) => resolve(root, 'src', path);

const entries = {
  core: src('core.ts'),
  explain: src('explain/index.ts'),
  steps: src('steps/index.ts'),
  ui: src('ui/index.ts'),
  scene3d: src('scene3d/index.ts'),
  formula: src('formula/index.ts'),
  theme: src('theme/index.ts'),
};

const katexWoff2Only: Plugin = {
  name: 'vizflow-katex-woff2-only',
  enforce: 'pre',
  transform(code, id) {
    if (!/[\\/]katex[\\/]dist[\\/][^\\/?]+\.css(\?|$)/.test(id)) return null;
    return { code: code.replace(/,\s*url\([^)]+\.(?:woff|ttf)\)\s*format\("(?:woff|truetype)"\)/g, ''), map: null };
  },
};

export default defineConfig({
  plugins: [katexWoff2Only, solid(), dts({ include: ['src'], outDirs: 'dist/types', entryRoot: 'src' })],
  resolve: {
    alias: Object.entries(entries).map(([name, path]) => ({
      find: `@yhnocoder/vizflow/${name}`,
      replacement: path,
    })),
  },
  build: {
    target: 'es2022',
    lib: { entry: entries, formats: ['es'] },
  },
  test: {
    include: ['test/**/*.test.ts'],
    environment: 'node',
  },
});
