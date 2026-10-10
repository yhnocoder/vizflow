import { readFileSync, readdirSync, rmSync } from 'node:fs';
import { join } from 'node:path';
import { afterAll, beforeAll, describe, expect, test } from 'vitest';
import { buildViteProjectToTemp } from './build';

let workspace = '';
let project = '';

beforeAll(() => {
  ({ workspace, project } = buildViteProjectToTemp());
}, 300_000);

afterAll(() => {
  for (const directory of [workspace, project]) if (directory) rmSync(directory, { recursive: true, force: true });
});

function assets(): string[] {
  return readdirSync(join(project, 'dist/assets'));
}

function bundledJs(): string {
  return assets()
    .filter((file) => file.endsWith('.js'))
    .map((file) => readFileSync(join(project, 'dist/assets', file), 'utf8'))
    .join('\n');
}

describe('examples/vite-project production build', () => {
  test('the page script is bundled with a single copy of three.js', () => {
    expect(readFileSync(join(project, 'dist/index.html'), 'utf8')).toMatch(/<script type="module" crossorigin src="\/assets\/index-[^"]+\.js">/);
    expect(bundledJs().split('Multiple instances of Three.js being imported').length - 1).toBe(1);
  });

  test('the consumer build emits the theme and KaTeX fonts as assets', () => {
    for (const font of ['KaTeX_Main-Regular', 'KaTeX_Math-Italic', 'geist-latin-400-normal', 'geist-mono-latin-500-normal', 'source-serif-4-latin-400-italic']) {
      expect(assets().some((file) => file.startsWith(`${font}-`) && file.endsWith('.woff2')), font).toBe(true);
    }
  });

  test('NODE_ENV is replaced by the consumer build', () => {
    expect(bundledJs()).not.toContain('process.env.NODE_ENV');
  });
});
