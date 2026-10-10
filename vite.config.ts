import { existsSync, readFileSync } from 'node:fs';
import { basename, dirname, resolve } from 'node:path';
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
  element: src('element/index.ts'),
};

const fontCssId = /[\\/]node_modules[\\/](?:katex[\\/]dist|@fontsource[\\/][^\\/]+)[\\/][^\\/?]+\.css\?raw$/;
const fontRequest = /\/fonts\/([^/?]+\.woff2)(?:\?|$)/;

function bundledFonts(): Plugin {
  const fonts = new Map<string, string>();
  return {
    name: 'vizflow-bundled-fonts',
    transform(_code, id) {
      if (!fontCssId.test(id)) return null;
      const cssFile = id.slice(0, id.indexOf('?'));
      for (const [, url] of readFileSync(cssFile, 'utf8').matchAll(/url\(\s*['"]?([^'")]+\.woff2)['"]?\s*\)/g)) {
        const fontFile = resolve(dirname(cssFile), url);
        const name = basename(fontFile);
        const known = fonts.get(name);
        if (known !== undefined && known !== fontFile) this.error(`font ${name} is provided by both ${known} and ${fontFile}`);
        fonts.set(name, fontFile);
      }
      return null;
    },
    configureServer(server) {
      server.middlewares.use((request, response, next) => {
        const name = fontRequest.exec(request.url ?? '')?.[1];
        const fontFile = name === undefined ? undefined : fonts.get(name);
        if (fontFile === undefined) return next();
        response.setHeader('content-type', 'font/woff2');
        response.end(readFileSync(fontFile));
      });
    },
    generateBundle() {
      for (const [name, fontFile] of fonts) {
        this.emitFile({ type: 'asset', fileName: `fonts/${name}`, source: readFileSync(fontFile) });
      }
    },
  };
}

const externalPackages = /^(?:three|solid-js|katex|@fontsource\/[^/]+)(?:\/|$)/;

function moduleVariants(): Plugin {
  return {
    name: 'vizflow-module-variants',
    enforce: 'pre',
    async resolveId(source, importer, options) {
      if (importer === undefined || !source.startsWith('.')) return null;
      const resolved = await this.resolve(source, importer, { ...options, skipSelf: true });
      if (resolved === null || !resolved.id.startsWith(src('')) || !resolved.id.endsWith('.ts')) return resolved;
      const variant = resolved.id.replace(/\.ts$/, '.module.ts');
      return existsSync(variant) ? variant : resolved;
    },
  };
}

const moduleConfig = defineConfig({
  plugins: [moduleVariants(), solid()],
  build: {
    target: 'es2022',
    outDir: 'dist/module',
    lib: { entry: entries, formats: ['es'] },
    rolldownOptions: { external: (id: string) => externalPackages.test(id) },
  },
});

const standaloneConfig = defineConfig({
  plugins: [bundledFonts(), solid(), dts({ include: ['src'], exclude: ['src/**/*.module.ts'], outDirs: 'dist/types', entryRoot: 'src' })],
  optimizeDeps: { entries: ['examples/*.html'] },
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

export default defineConfig(({ mode }) => (mode === 'module' ? moduleConfig : standaloneConfig));
