import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { type CompiledPage, loadPage } from '../../src/dsl';

export const examplesUrl = pathToFileURL(resolve(import.meta.dirname, '../../examples/')).href + '/';

export function readFileUrl(url: string): Promise<string> {
  return readFile(new URL(url), 'utf8');
}

export function loadExample(name: string): Promise<CompiledPage> {
  return loadPage(new URL(name, examplesUrl).href, { read: readFileUrl });
}
