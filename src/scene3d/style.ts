export interface EdgeStyle {
  opacity: number;
}

export interface EdgeContext {
  dir: 'forward' | 'backward';
  overview: boolean;
  contract: boolean;
  zeroGrad: boolean;
  focused: boolean;
  far: boolean;
  current: boolean;
  revealedFrom: boolean;
  revealedTo: boolean;
}

export const EDGE_OVERVIEW = 0.35;
export const EDGE_OVERVIEW_CONTRACT = 0.2;
export const EDGE_OVERVIEW_ZERO = 0.12;
export const EDGE_ACTIVE = 0.85;
export const EDGE_ACTIVE_CONTRACT = 0.6;
export const EDGE_ACTIVE_BACKWARD = 0.9;
export const EDGE_ACTIVE_CONTRACT_BACKWARD = 0.65;
export const EDGE_ACTIVE_ZERO = 0.35;
export const EDGE_IDLE = 0.13;
export const EDGE_UNFOCUSED_FACTOR = 0.4;

export function edgeStyle(ctx: EdgeContext): EdgeStyle | null {
  if (ctx.far) return null;
  if (ctx.dir === 'forward' ? !ctx.revealedTo : !ctx.revealedFrom) return null;
  let style: EdgeStyle;
  if (ctx.dir === 'forward') {
    if (ctx.overview) style = { opacity: ctx.contract ? EDGE_OVERVIEW_CONTRACT : EDGE_OVERVIEW };
    else if (ctx.current) style = { opacity: ctx.contract ? EDGE_ACTIVE_CONTRACT : EDGE_ACTIVE };
    else style = { opacity: EDGE_IDLE };
  } else if (ctx.overview) style = { opacity: ctx.zeroGrad ? EDGE_OVERVIEW_ZERO : ctx.contract ? EDGE_OVERVIEW_CONTRACT : EDGE_OVERVIEW };
  else if (ctx.current) style = { opacity: ctx.zeroGrad ? EDGE_ACTIVE_ZERO : ctx.contract ? EDGE_ACTIVE_CONTRACT_BACKWARD : EDGE_ACTIVE_BACKWARD };
  else style = { opacity: EDGE_IDLE };
  if (!ctx.focused) style.opacity *= EDGE_UNFOCUSED_FACTOR;
  return style;
}
