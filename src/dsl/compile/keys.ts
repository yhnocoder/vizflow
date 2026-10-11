export const ITEM_KEYS = new Set(['op', 'tex', 'dtype']);
export const MODULE_ITEM_KEYS = ['title', 'subtitle', 'fixed', 'folded', 'split', 'fold'];
export const INSTANCE_KEYS = new Set(['use', 'params', 'inputs', 'outputs', 'weights', 'suffix', 'tex', 'steps', 'beside', 'inline', 'brace', 'formula', ...MODULE_ITEM_KEYS]);
export const GROUP_KEYS = new Set(['body', 'steps', 'beside', 'formula', ...MODULE_ITEM_KEYS]);
export const FOR_KEYS = new Set(['for', 'in', 'name', 'brace', 'beside', 'sequential', 'body']);
export const MODULE_KEYS = new Set(['module', 'import', 'subtitle', 'formula', 'params', 'inputs', 'outputs', 'weights', 'buffers', 'body', 'steps', 'backward', 'fold', 'cache']);
export const PAGE_KEYS = new Set([
  'title', 'lede', 'hint', 'camera', 'overview', 'leaves', 'import', 'modules', 'subtitle', 'params', 'dims', 'real', 'dtype', 'vocab', 'init', 'inputs', 'sequence',
  'weights', 'buffers', 'values', 'body', 'steps', 'backward', 'panels', 'sections', 'layout', 'generate', 'optimizer', 'rounding',
]);
export const STEP_KEYS = new Set(['id', 'show', 'active', 'label', 'title', 'text', 'segment', 'brace', 'backward', 'notes']);
export const BRACE_KEYS = new Set(['title', 'subtitle']);
export const PATCH_KEYS = new Set(['title', 'label', 'text', 'append', 'backward', 'notes']);
export const TEXT_KEYS = new Set(['title', 'text', 'label']);
export const FOLD_KEYS = new Set(['title', 'text', 'label', 'backward']);
export const OVERVIEW_KEYS = new Set(['title', 'text', 'label', 'backward', 'notes']);
export const NOTE_KEYS = new Set(['at', 'text', 'side']);
export const NOTE_SIDES: ReadonlySet<string> = new Set(['left', 'right', 'above', 'below']);
export const GENERATE_KEYS = new Set(['steps']);
export const OPTIMIZER_KEYS = new Set(['type', 'lr', 'weight_decay']);
export const SEQUENCE_KEYS = new Set(['tokens', 'input', 'label']);
