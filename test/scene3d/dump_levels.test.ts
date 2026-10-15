// @vitest-environment jsdom
import { mkdirSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { it } from 'vitest';
import { levelPlan } from '../../src/scene3d/levels';
import { loadExample } from '../dsl/load';

const out = process.env.VIZFLOW_DUMP_OUT;
const pages = (process.env.VIZFLOW_DUMP_PAGES ?? '').split(',').filter((p) => p !== '');

it.skipIf(out === undefined)('dumps the levels and columns of the given pages', async () => {
  mkdirSync(out!, { recursive: true });
  for (const name of pages) {
    const page = await loadExample(name);
    writeFileSync(join(out!, `${name.replace(/\//g, '_')}.txt`), levelPlan(page.steps, page.levels).dump());
  }
}, 300000);
