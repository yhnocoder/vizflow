export interface EdgeStyle {
  opacity: number;
  dashed: boolean;
}

export interface EdgeContext {
  dir: 'forward' | 'backward';
  overview: boolean;
  contract: boolean;
  toLoss: boolean;
  zeroGrad: boolean;
  focused: boolean;
  revealedFrom: boolean;
  revealedTo: boolean;
  activeFrom: boolean;
  activeTo: boolean;
}

export const EDGE_HIDDEN = 0.1;
export const EDGE_OVERVIEW = 0.42;
export const EDGE_OVERVIEW_LOSS = 0.9;
export const EDGE_ACTIVE = 0.95;
export const EDGE_ACTIVE_CONTRACT_FORWARD = 0.75;
export const EDGE_ACTIVE_CONTRACT_BACKWARD = 0.8;
export const EDGE_IDLE = 0.3;
export const EDGE_UNFOCUSED_FACTOR = 0.4;
export const DASH_LENGTH = 0.12;
export const DASH_GAP = 0.1;

export function edgeStyle(ctx: EdgeContext): EdgeStyle {
  let style: EdgeStyle;
  if (ctx.dir === 'forward') {
    if (!ctx.revealedTo) style = { opacity: EDGE_HIDDEN, dashed: true };
    else if (ctx.overview) style = { opacity: ctx.toLoss ? EDGE_OVERVIEW_LOSS : EDGE_OVERVIEW, dashed: false };
    else if (ctx.activeTo) style = { opacity: ctx.contract ? EDGE_ACTIVE_CONTRACT_FORWARD : EDGE_ACTIVE, dashed: false };
    else style = { opacity: EDGE_IDLE, dashed: false };
  } else if (!ctx.revealedFrom) style = { opacity: EDGE_HIDDEN, dashed: ctx.zeroGrad };
  else if (ctx.zeroGrad) style = { opacity: EDGE_IDLE, dashed: true };
  else if (ctx.activeFrom) style = { opacity: ctx.contract ? EDGE_ACTIVE_CONTRACT_BACKWARD : EDGE_ACTIVE, dashed: false };
  else style = { opacity: EDGE_IDLE, dashed: false };
  if (!ctx.focused) style.opacity *= EDGE_UNFOCUSED_FACTOR;
  return style;
}
