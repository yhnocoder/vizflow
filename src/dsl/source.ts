import { LineCounter, isAlias, isMap, isScalar, isSeq, parseDocument, type Document, type Node as YamlNode } from 'yaml';

export interface Pos {
  file: string;
  line: number;
  col: number;
  shift: number;
}

const positions = new WeakMap<object, Map<string | number, Pos>>();
const ownPositions = new WeakMap<object, Pos>();

export class DslError extends Error {
  override name = 'DslError';
}

export interface Where {
  file: string;
  path: string;
  obj?: object;
  key?: string | number;
}

export function child(where: Where, obj: object | undefined, key: string | number): Where {
  const step = typeof key === 'number' ? `[${key}]` : where.path === '' ? key : `.${key}`;
  return { file: where.file, path: `${where.path}${step}`, obj, key };
}

export function posOf(where: Where): Pos | undefined {
  if (where.obj === undefined) return undefined;
  if (where.key === undefined) return ownPositions.get(where.obj);
  return positions.get(where.obj)?.get(where.key) ?? ownPositions.get(where.obj);
}

export function fail(where: Where, message: string, offset?: number): never {
  const pos = posOf(where);
  const label = where.path === '' ? '' : ` ${where.path}`;
  if (pos === undefined) throw new DslError(`${where.file}${label}: ${message}`);
  const col = offset === undefined ? pos.col : pos.col + pos.shift + offset;
  throw new DslError(`${pos.file}:${pos.line}:${col}${label}: ${message}`);
}

function quoteShift(node: YamlNode): number {
  if (!isScalar(node)) return 0;
  return node.type === 'QUOTE_DOUBLE' || node.type === 'QUOTE_SINGLE' ? 1 : 0;
}

export function place(obj: object, key: string | number | undefined, file: string, line: number): void {
  const pos = { file, line, col: 1, shift: 0 };
  if (key === undefined) {
    ownPositions.set(obj, pos);
    return;
  }
  const keys = positions.get(obj) ?? new Map<string | number, Pos>();
  keys.set(key, pos);
  positions.set(obj, keys);
}

export function parseYaml(text: string, file: string, firstLine = 1): unknown {
  const lineCounter = new LineCounter();
  const doc = parseDocument(text, { lineCounter, uniqueKeys: true });
  const linePos = (offset: number) => {
    const { line, col } = lineCounter.linePos(offset);
    return { line: line + firstLine - 1, col };
  };
  const posAt = (offset: number, shift = 0): Pos => ({ file, ...linePos(offset), shift });
  const first = doc.errors[0];
  if (first !== undefined) {
    const { line, col } = linePos(first.pos[0]);
    throw new DslError(`${file}:${line}:${col}: ${first.message.split('\n')[0]}`);
  }
  const convert = (node: unknown): unknown => {
    const resolved = isAlias(node) ? node.resolve(doc as Document) : node;
    if (isMap(resolved)) {
      const out: Record<string, unknown> = {};
      const keys = new Map<string | number, Pos>();
      for (const pair of resolved.items) {
        const key = isScalar(pair.key) ? String(pair.key.value) : String(pair.key);
        out[key] = convert(pair.value);
        const valueNode = pair.value as YamlNode | null;
        const keyNode = pair.key as YamlNode;
        const range = valueNode?.range ?? keyNode.range;
        if (range) keys.set(key, posAt(range[0], valueNode ? quoteShift(valueNode) : 0));
      }
      positions.set(out, keys);
      if (resolved.range) ownPositions.set(out, posAt(resolved.range[0]));
      return out;
    }
    if (isSeq(resolved)) {
      const out: unknown[] = [];
      const keys = new Map<string | number, Pos>();
      resolved.items.forEach((item, i) => {
        out.push(convert(item));
        const itemNode = item as YamlNode;
        if (itemNode?.range) keys.set(i, posAt(itemNode.range[0], quoteShift(itemNode)));
      });
      positions.set(out, keys);
      if (resolved.range) ownPositions.set(out, posAt(resolved.range[0]));
      return out;
    }
    if (isScalar(resolved)) return resolved.value;
    return resolved ?? null;
  };
  return convert(doc.contents);
}
