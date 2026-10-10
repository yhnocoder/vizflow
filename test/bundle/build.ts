import { execFileSync } from 'node:child_process';
import { cpSync, mkdtempSync, readFileSync, symlinkSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { basename, join, resolve } from 'node:path';

const repo = resolve(import.meta.dirname, '../..');

export function buildDistToTemp(): { workspace: string; dist: string } {
  const workspace = mkdtempSync(join(tmpdir(), 'vizflow-build-'));
  for (const name of ['src', 'package.json', 'tsconfig.json', 'vite.config.ts']) {
    cpSync(join(repo, name), join(workspace, name), { recursive: true });
  }
  symlinkSync(join(repo, 'node_modules'), join(workspace, 'node_modules'), 'dir');
  for (const mode of ['production', 'module']) {
    execFileSync(process.execPath, [join(repo, 'node_modules/vite/bin/vite.js'), 'build', '--mode', mode, '--logLevel', 'error'], {
      cwd: workspace,
      stdio: 'pipe',
    });
  }
  return { workspace, dist: join(workspace, 'dist') };
}

const skippedInExample = new Set(['node_modules', 'dist', 'package-lock.json']);

export function buildViteProjectToTemp(): { workspace: string; project: string } {
  const { workspace } = buildDistToTemp();
  const project = mkdtempSync(join(tmpdir(), 'vizflow-vite-project-'));
  cpSync(join(repo, 'examples/vite-project'), project, {
    recursive: true,
    filter: (source) => !skippedInExample.has(basename(source)),
  });
  const manifestPath = join(project, 'package.json');
  const manifest = JSON.parse(readFileSync(manifestPath, 'utf8'));
  manifest.dependencies['@yhnocoder/vizflow'] = `file:${workspace}`;
  writeFileSync(manifestPath, JSON.stringify(manifest, null, 2));
  const npm = { cwd: project, stdio: 'pipe' as const };
  execFileSync('npm', ['install', '--prefer-offline', '--no-audit', '--no-fund'], npm);
  execFileSync('npm', ['run', 'build'], npm);
  return { workspace, project };
}
