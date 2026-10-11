export const ITEM_KEYS = new Set(['op', 'tex']);
export const MODULE_ITEM_KEYS = ['title', 'subtitle', 'fixed', 'folded', 'split', 'fold'];
export const INSTANCE_KEYS = new Set(['use', 'params', 'inputs', 'outputs', 'weights', 'suffix', 'tex', 'steps', 'beside', 'inline', 'brace', ...MODULE_ITEM_KEYS]);
export const GROUP_KEYS = new Set(['body', 'steps', 'beside', ...MODULE_ITEM_KEYS]);
export const FOR_KEYS = new Set(['for', 'in', 'name', 'brace', 'beside', 'body']);
export const MODULE_KEYS = new Set(['module', 'import', 'subtitle', 'params', 'inputs', 'outputs', 'weights', 'buffers', 'body', 'steps', 'backward', 'fold']);
export const PAGE_KEYS = new Set([
  'title', 'lede', 'hint', 'camera', 'overview', 'leaves', 'import', 'modules', 'subtitle', 'params', 'dims', 'real', 'vocab', 'init', 'inputs', 'sequence',
  'weights', 'buffers', 'values', 'body', 'steps', 'backward', 'panels', 'sections', 'layout',
]);
export const STEP_KEYS = new Set(['id', 'show', 'active', 'label', 'title', 'text', 'segment', 'brace', 'backward']);
export const BRACE_KEYS = new Set(['title', 'subtitle']);
export const PATCH_KEYS = new Set(['title', 'label', 'text', 'append', 'backward']);
export const TEXT_KEYS = new Set(['title', 'text', 'label']);
export const FOLD_KEYS = new Set(['title', 'text', 'label', 'backward']);
export const OVERVIEW_KEYS = new Set(['title', 'text', 'label', 'backward']);
export const SEQUENCE_KEYS = new Set(['tokens', 'input', 'label']);
