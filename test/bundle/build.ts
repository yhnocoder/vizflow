import { execFileSync } from 'node:child_process';
import { cpSync, mkdtempSync, symlinkSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';

const repo = resolve(import.meta.dirname, '../..');

export function buildDistToTemp(): { workspace: string; dist: string } {
  const workspace = mkdtempSync(join(tmpdir(), 'vizflow-build-'));
  for (const name of ['src', 'package.json', 'tsconfig.json', 'vite.config.ts']) {
    cpSync(join(repo, name), join(workspace, name), { recursive: true });
  }
  symlinkSync(join(repo, 'node_modules'), join(workspace, 'node_modules'), 'dir');
  execFileSync(process.execPath, [join(repo, 'node_modules/vite/bin/vite.js'), 'build', '--logLevel', 'error'], {
    cwd: workspace,
    stdio: 'pipe',
  });
  return { workspace, dist: join(workspace, 'dist') };
}
