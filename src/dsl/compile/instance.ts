import { Node } from '../../autograd';
import type { LayoutEntry, LayoutTree } from '../../scene3d';
import type { ModuleDef } from '../load';
import { type Where, child, fail } from '../source';
import { compileBody } from './body';
import { registerFormula } from './formula';
import { declareBuffers, declareParams, declareWeights, define, marked, matchShape } from './declare';
import { INSTANCE_KEYS, MODULE_ITEM_KEYS, MODULE_KEYS } from './keys';
import { closeModule, openModule, titleOf } from './modules';
import { braceOf, moduleBackward, moduleSteps, patchSteps } from './steps';
import type { Compiler } from './text';
import { type Env, type ModuleValue, Scope, type Slot } from './types';
import { checkKeys, join, record } from './util';

export function usedModule(scope: Scope, use: string): string {
  const param = scope.params.get(use);
  return param?.type === 'module' ? (param.value as ModuleValue).module : use;
}

export function compileInstance(
  c: Compiler,
  parent: Scope,
  name: string,
  braceTitle: string,
  indices: number[],
  use: Record<string, unknown>,
  env: Env,
  entries: LayoutEntry[],
  counts: Map<ModuleDef, number>,
  ordinals: Map<ModuleDef, number>,
  before: number,
): Slot {
  checkKeys(use, INSTANCE_KEYS, env.where);
  const useWhere = child(env.where, use, 'use');
  if (typeof use.use !== 'string') fail(useWhere, 'use must name a module');
  const def = parent.def.namespace.get(usedModule(parent, use.use));
  if (def === undefined) fail(useWhere, `module '${use.use}' is not imported`);
  if (parent.local(name) !== undefined || parent.groups.has(name)) fail(env.where, `name '${name}' is already defined in ${c.scopeLabel(parent)}`);
  const ordinal = (ordinals.get(def) ?? 0) + 1;
  ordinals.set(def, ordinal);
  let mark: string | undefined;
  if (typeof use.suffix === 'string') mark = c.ofText(use.suffix, { ...env, where: child(env.where, use, 'suffix') }) || undefined;
  else if ((counts.get(def) ?? 0) > 1) mark = indices.length > 0 ? indices.join(',') : String(ordinal);
  const scope = new Scope(def, join(parent.path, name), mark === undefined ? [...parent.marks] : [mark, ...parent.marks]);
  parent.children.set(name, scope);
  const decl = def.decl;
  const where = def.where;
  checkKeys(decl, MODULE_KEYS, where);
  const childEnv: Env = { scope, loops: new Map(), where };
  const given = record(use.params, child(env.where, use, 'params'), 'params');
  declareParams(c, scope, decl, childEnv, { ...env, where: child(env.where, use, 'params') }, given, where, name);
  const texWhere = child(env.where, use, 'tex');
  for (const [key, tex] of Object.entries(record(use.tex, texWhere, 'tex'))) {
    if (typeof tex !== 'string') fail(child(texWhere, use.tex as object, key), 'tex must be a string');
    const texEnv = { ...env, where: child(texWhere, use.tex as object, key) };
    scope.texOverride.set(key, marked(c.ofText(tex, texEnv), parent, c.ofRefersToNode(tex, texEnv)));
  }
  const declInputsWhere = child(where, decl, 'inputs');
  const declInputs = record(decl.inputs, declInputsWhere, 'inputs');
  const inputsWhere = child(env.where, use, 'inputs');
  const givenInputs = record(use.inputs, inputsWhere, 'inputs');
  for (const key of Object.keys(givenInputs)) if (!(key in declInputs)) fail(child(inputsWhere, givenInputs, key), `module ${def.name} has no input '${key}'`);
  for (const [inputName, inputDecl] of Object.entries(declInputs)) {
    const declWhere = child(declInputsWhere, declInputs, inputName);
    const inputSpec = record(inputDecl, declWhere, `input '${inputName}'`);
    checkKeys(inputSpec, new Set(['shape', 'tex', 'when']), declWhere);
    if (!c.enabled(inputSpec, { ...childEnv, where: declWhere })) continue;
    if (!(inputName in givenInputs)) fail(inputsWhere, `input '${inputName}' of ${def.name} is not given`);
    const at = child(inputsWhere, givenInputs, inputName);
    const node = c.node(givenInputs[inputName], { ...env, where: at });
    if (typeof inputSpec.tex === 'string') scope.inputTex.set(inputName, inputSpec.tex);
    if (inputSpec.shape !== undefined) matchShape(c, scope, inputSpec.shape, node, childEnv, child(declWhere, inputSpec, 'shape'), at);
    scope.entries.set(inputName, node);
    c.nodes.set(join(scope.path, inputName), node);
  }
  const outputsWhere = child(where, decl, 'outputs');
  if (!Array.isArray(decl.outputs)) fail(outputsWhere, 'outputs must be a list of names');
  const outputs = decl.outputs as unknown[];
  const boundWhere = child(env.where, use, 'outputs');
  const bound = record(use.outputs, boundWhere, 'outputs');
  const bindings: [string, string, Where][] = [];
  for (const [output, target] of Object.entries(bound)) {
    const at = child(boundWhere, bound, output);
    if (!outputs.includes(output)) fail(at, `module ${def.name} has no output '${output}'`);
    if (typeof target !== 'string') fail(at, 'an output binding must be a name');
    const concrete = c.concreteName(target, { ...env, where: at });
    bindings.push([output, concrete.name, at]);
    const outer = parent.texOverride.get(concrete.name) ?? parent.texOverride.get(concrete.base);
    if (outer !== undefined) scope.texOverride.set(output, outer);
    if (!scope.texOverride.has(output)) scope.outputTex.set(output, marked(c.loopTex(concrete.base, concrete.indices), parent, false));
  }
  if (use.inline !== undefined && typeof use.inline !== 'boolean') fail(child(env.where, use, 'inline'), 'inline must be true or false');
  const inline = use.inline === true;
  for (const key of MODULE_ITEM_KEYS) {
    if (inline && use[key] !== undefined) fail(child(env.where, use, key), `an inline instance has no box of its own, so it cannot have '${key}'`);
  }
  const tree: LayoutTree = inline ? { kind: 'module', children: [] } : { kind: 'module', title: titleOf(c, use, braceTitle, env), children: [] };
  if (!inline && typeof use.subtitle === 'string') tree.subtitle = use.subtitle;
  else if (!inline && typeof decl.subtitle === 'string') tree.subtitle = decl.subtitle;
  entries.push(tree);
  const rec = inline ? undefined : openModule(c, scope.path, tree.title as string, tree.subtitle, use, env);
  if (rec !== undefined) rec.defaults = { decl, env: childEnv };
  declareWeights(c, scope, decl, childEnv, use.weights === undefined ? undefined : { map: record(use.weights, child(env.where, use, 'weights'), 'weights'), env: { ...env, where: child(env.where, use, 'weights') } }, tree.children);
  declareBuffers(c, scope, decl, childEnv, tree.children);
  const { slots } = compileBody(c, scope, decl, childEnv, tree.children);
  for (const output of outputs) {
    if (typeof output !== 'string' || !(scope.entries.get(output) instanceof Node)) fail(outputsWhere, `output '${String(output)}' of ${def.name} is not defined in its body`);
  }
  const steps = moduleSteps(c, scope, decl, childEnv, slots);
  const formulaSources = [{ decl, env: childEnv }, { decl: use, env }];
  if (use.brace !== undefined) {
    if (!inline) fail(child(env.where, use, 'brace'), 'only an inline instance takes brace; a module with a box has its own brace');
    const brace = braceOf(c, use.brace, { ...env, where: child(env.where, use, 'brace') });
    registerFormula(c, scope.path, formulaSources, brace);
    for (const rec of steps) rec.step.brace ??= brace;
  } else if (inline && use.formula !== undefined) {
    fail(child(env.where, use, 'formula'), 'an inline instance has no module of its own, so its formula needs a brace to name it');
  }
  if (rec !== undefined) registerFormula(c, scope.path, formulaSources, rec);
  moduleBackward(c, decl, childEnv);
  for (const [output, target, at] of bindings) define(c, parent, target, scope.entries.get(output) as Node, { ...env, where: at });
  c.pendingPatches[c.pendingPatches.length - 1].push(() => patchSteps(c, scope, steps, use, env));
  const created = c.g.nodes.slice(before);
  if (rec !== undefined) closeModule(c, rec, created);
  return { kind: 'child', def, steps, created };
}
