import { type Node, opDef } from '../../autograd';
import type { LayoutEntry } from '../../scene3d';
import { DslError, type Where, child, fail } from '../source';
import { addMark, defaultTex, elemOf } from '../tex';
import { reshapeNested } from './buffer-functions';
import type { Compiler } from './text';
import { type BufferValue, type Env, type Num, type OpValue, type ParamType, type Scope } from './types';
import { checkKeys, dtypeOption, isNum, isRecord, join, num, record } from './util';

const PARAM_TYPES: ReadonlySet<string> = new Set(['int', 'float', 'dim', 'op', 'tex', 'text']);
const INITS = new Set(['ones', 'zeros', 'normal', 'uniform']);
const DEFAULT_INIT = 'uniform';

export function define(c: Compiler, scope: Scope, name: string, node: Node, env: Env): void {
  if (scope.local(name) !== undefined || scope.groups.has(name)) c.fail(env, `name '${name}' is already defined in ${c.scopeLabel(scope)}`);
  scope.entries.set(name, node);
  c.nodes.set(join(scope.path, name), node);
}

export function marked(tex: string, scope: Scope, hasOf: boolean): string {
  if (hasOf) return tex;
  return scope.marks.reduce((acc, mark) => addMark(acc, mark), tex);
}

function paramValue(c: Compiler, decl: Record<string, unknown>, given: unknown, name: string, env: Env, parentEnv: Env | undefined, where: Where, instance?: string): { type: ParamType; value: Num | string | OpValue } {
  const type = decl.type;
  if (typeof type !== 'string' || !PARAM_TYPES.has(type)) fail(where, `parameter '${name}' needs a type: int, float, dim, op, tex or text`);
  if (decl.default_key !== undefined && (decl.default_key !== true || type !== 'text')) fail(where, `parameter '${name}': default_key must be true and needs type text`);
  const keyDefault = decl.default_key === true ? instance : undefined;
  const raw = given !== undefined ? given : (keyDefault ?? decl.default);
  if (raw === undefined) fail(where, `parameter '${name}' has no value and no default`);
  const sourceEnv = given !== undefined && parentEnv !== undefined ? parentEnv : env;
  const t = type as ParamType;
  if (t === 'op') {
    if (typeof raw !== 'string') fail(where, `parameter '${name}' must name an op`);
    const parentParam = given !== undefined ? parentEnv?.scope.params.get(raw) : undefined;
    const op = parentParam?.type === 'op' ? (parentParam.value as OpValue).op : raw;
    const allowed = Array.isArray(decl.values) ? (decl.values as string[]) : undefined;
    if (allowed !== undefined && !allowed.includes(op)) fail(where, `parameter '${name}' must be one of ${allowed.join(', ')}, got '${op}'`);
    if (opDef(op) === undefined) fail(where, `parameter '${name}': op '${op}' is not defined`);
    return { type: t, value: { op } };
  }
  if (t === 'tex' || t === 'text') {
    if (typeof raw !== 'string') fail(where, `parameter '${name}' must be a string`);
    return { type: t, value: raw };
  }
  const value = c.expression(raw, { ...sourceEnv, where }, undefined);
  if (!isNum(value)) fail(where, `parameter '${name}' must be a number`);
  if ((t === 'int' || t === 'dim') && !Number.isInteger(value.toy)) fail(where, `parameter '${name}' must be an integer, got ${value.toy}`);
  if (t === 'dim' && value.toy <= 0) fail(where, `parameter '${name}' must be a positive dimension, got ${value.toy}`);
  return { type: t, value: t === 'dim' ? value : num(value.toy) };
}

export function declareParams(c: Compiler, scope: Scope, decl: Record<string, unknown>, env: Env, parentEnv: Env | undefined, given: Record<string, unknown>, where: Where, instance?: string): void {
  const paramsWhere = child(where, decl, 'params');
  const params = record(decl.params, paramsWhere, 'params');
  for (const key of Object.keys(given)) {
    if (!(key in params)) fail(child(parentEnv?.where ?? where, given, key), `module ${scope.def.name} has no parameter '${key}'`);
  }
  for (const [name, paramDecl] of Object.entries(params)) {
    const at = given[name] !== undefined && parentEnv !== undefined ? child(parentEnv.where, given, name) : child(paramsWhere, params, name);
    if (!isRecord(paramDecl)) fail(at, `parameter '${name}' must be a mapping`);
    const value = paramValue(c, paramDecl, given[name], name, env, parentEnv, at, instance);
    scope.params.set(name, value);
    if (value.type === 'dim') scope.dims.set(name, value.value as Num);
  }
}

function shapeOf(c: Compiler, spec: unknown, env: Env, where: Where): Num[] {
  if (!Array.isArray(spec)) fail(where, 'shape must be a list');
  return spec.map((entry, i) => {
    const at = child(where, spec, i);
    const value = c.expression(entry, { ...env, where: at }, undefined);
    if (!isNum(value) || !Number.isInteger(value.toy) || value.toy <= 0) fail(at, `shape entry ${JSON.stringify(entry)} is not a positive integer`);
    return value;
  });
}

function leafSpec(c: Compiler, shape: Num[]): (number | string)[] {
  return shape.map((d) => c.dimSpec(d));
}

export function declareWeights(c: Compiler, scope: Scope, decl: Record<string, unknown>, env: Env, bound: { map: Record<string, unknown>; env: Env } | undefined, entries: LayoutEntry[]): void {
  const weightsWhere = child(env.where, decl, 'weights');
  const weights = record(decl.weights, weightsWhere, 'weights');
  for (const key of Object.keys(bound?.map ?? {})) {
    if (!(key in weights)) fail(child(bound!.env.where, bound!.map, key), `module ${scope.def.name} has no weight '${key}'`);
  }
  for (const [name, weightDecl] of Object.entries(weights)) {
    const at = child(weightsWhere, weights, name);
    if (!isRecord(weightDecl)) fail(at, `weight '${name}' must be a mapping`);
    checkKeys(weightDecl, new Set(['shape', 'init', 'tex', 'dtype']), at);
    const shape = shapeOf(c, weightDecl.shape, { ...env, where: child(at, weightDecl, 'shape') }, child(at, weightDecl, 'shape'));
    const binding = bound?.map[name];
    if (binding !== undefined) {
      const bindWhere = child(bound!.env.where, bound!.map, name);
      const node = c.node(binding, { ...bound!.env, where: bindWhere });
      if (node.shape.join(',') !== shape.map((d) => d.toy).join(',')) {
        fail(bindWhere, `weight '${name}' of ${scope.def.name} has shape (${shape.map((d) => d.toy).join(', ')}), but '${String(binding)}' has shape (${node.shape.join(', ')})`);
      }
      define(c, scope, name, node, { ...env, where: at });
      continue;
    }
    const init = weightDecl.init ?? DEFAULT_INIT;
    if (typeof init !== 'string' || !INITS.has(init)) fail(child(at, weightDecl, 'init'), `init must be one of ${[...INITS].join(', ')}`);
    const path = join(scope.path, name);
    const declared = typeof weightDecl.tex === 'string' ? c.ofText(weightDecl.tex, { ...env, where: child(at, weightDecl, 'tex') }) : undefined;
    const tex = scope.texOverride.get(name) ?? marked(declared ?? defaultTex(name), scope, typeof weightDecl.tex === 'string' && c.ofRefersToNode(weightDecl.tex, { ...env, where: child(at, weightDecl, 'tex') }));
    const values = c.takeValue(path) ?? c.initValues(init, shape.map((d) => d.toy));
    let node: Node;
    try {
      node = c.g.param(path, leafSpec(c, shape), values, { tex, elem: elemOf(tex), ...dtypeOption(weightDecl, at) });
    } catch (error) {
      if (error instanceof DslError) throw error;
      fail(at, (error as Error).message);
    }
    define(c, scope, name, node, { ...env, where: at });
    c.setOwner(node, entries);
  }
}

export function declareBuffers(c: Compiler, scope: Scope, decl: Record<string, unknown>, env: Env, entries: LayoutEntry[]): void {
  const buffersWhere = child(env.where, decl, 'buffers');
  const buffers = record(decl.buffers, buffersWhere, 'buffers');
  for (const [name, bufferDecl] of Object.entries(buffers)) {
    const at = child(buffersWhere, buffers, name);
    if (!isRecord(bufferDecl)) fail(at, `buffer '${name}' must be a mapping`);
    checkKeys(bufferDecl, new Set(['value', 'shape', 'tex', 'dtype']), at);
    const bufferDtype = dtypeOption(bufferDecl, at);
    const valueWhere = child(at, bufferDecl, 'value');
    const value = c.expression(bufferDecl.value, { ...env, where: valueWhere, buffers: true }, undefined);
    if (!isRecord(value) || !('buffer' in value)) fail(valueWhere, 'a buffer value must call rope_cos, rope_sin, causal_mask, zeros, full or arange');
    let buffer = value as BufferValue;
    if (bufferDecl.shape !== undefined) {
      const shapeWhere = child(at, bufferDecl, 'shape');
      const shape = shapeOf(c, bufferDecl.shape, { ...env, where: shapeWhere }, shapeWhere);
      const size = (dims: Num[]) => dims.reduce((a, d) => a * d.toy, 1);
      if (size(shape) !== size(buffer.shape)) {
        fail(shapeWhere, `buffer '${name}' has ${size(buffer.shape)} values, which do not fit the shape (${shape.map((d) => d.toy).join(', ')})`);
      }
      buffer = { buffer: true, shape, values: reshapeNested(buffer.values, shape.map((d) => d.toy)) };
    }
    const path = join(scope.path, name);
    const given = c.takeValue(path);
    if (given !== undefined) buffer = { ...buffer, values: given };
    const key = given !== undefined ? `${path}=values` : `${scope.def.file}#${scope.def.name}.${name}=${String(bufferDecl.value)}(${buffer.shape.map((d) => `${d.toy}:${d.real}`).join(',')};${JSON.stringify(buffer.values)})`;
    const tex = typeof bufferDecl.tex === 'string' ? c.ofText(bufferDecl.tex, { ...env, where: child(at, bufferDecl, 'tex') }) : defaultTex(name);
    const node = c.bufferNode(key, () => {
      try {
        const created = c.g.buffer(path, leafSpec(c, buffer.shape), buffer.values, { tex, elem: elemOf(tex), ...bufferDtype });
        c.setOwner(created, entries);
        return created;
      } catch (error) {
        return fail(at, (error as Error).message);
      }
    });
    define(c, scope, name, node, { ...env, where: at });
  }
}

export function matchShape(c: Compiler, scope: Scope, pattern: unknown, node: Node, env: Env, where: Where, givenAt: Where): void {
  if (!Array.isArray(pattern)) fail(where, 'shape must be a list');
  const rest = pattern.indexOf('...');
  const extra = node.shape.length - (pattern.length - 1);
  if (rest >= 0 ? extra < 0 || pattern.lastIndexOf('...') !== rest : pattern.length !== node.shape.length) {
    fail(givenAt, `input has shape (${node.shape.join(', ')}), but ${scope.def.name} expects (${pattern.join(', ')})`);
  }
  pattern.forEach((entry, at) => {
    if (at === rest) return;
    const i = rest >= 0 && at > rest ? at - 1 + extra : at;
    const actual: Num = { toy: node.shape[i], real: node.realShape[i] };
    if (typeof entry === 'string' && /^[A-Za-z_][A-Za-z0-9_]*$/.test(entry) && !scope.dims.has(entry) && !scope.params.has(entry)) {
      scope.dims.set(entry, actual);
      return;
    }
    const expected = c.expression(entry, { ...env, where: child(where, pattern, at) }, undefined);
    if (!isNum(expected) || expected.toy !== actual.toy) {
      fail(givenAt, `input has shape (${node.shape.join(', ')}), but dimension ${i} of ${scope.def.name}'s pattern (${pattern.join(', ')}) is ${isNum(expected) ? expected.toy : '?'}`);
    }
  });
}
