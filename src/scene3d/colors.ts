import type { LinearRgb, ThemeColors } from '../theme';
import { ZERO_THRESHOLD } from './scales';

export const INACTIVE_MIX = 0.55;
export const FAR_MIX = 0.88;
export const UNFOCUSED_MIX = 0.45;
export const NODE_HIDDEN_MIX = 0.5;
export const NODE_FAR_MIX = 0.85;
export const NODE_IDLE_MIX = 0.6;
export const OUTLINE_ACTIVE = 0.85;
export const OUTLINE_IDLE = 0.32;
export const OUTLINE_FAR = 0.08;
export const OUTLINE_TILE_FACTOR = 0.85;

export function mix(a: LinearRgb, b: LinearRgb, k: number): LinearRgb {
  return [a[0] + (b[0] - a[0]) * k, a[1] + (b[1] - a[1]) * k, a[2] + (b[2] - a[2]) * k];
}

export interface BarState {
  active: boolean;
  far: boolean;
  focused: boolean;
}

function signColor(value: number, theme: ThemeColors): LinearRgb {
  return Math.abs(value) < ZERO_THRESHOLD ? theme.lineStrong : value > 0 ? theme.pos : theme.neg;
}

export function barColor(value: number, theme: ThemeColors, state: BarState): LinearRgb {
  const base = signColor(value, theme);
  let color = state.active && !state.far ? base : mix(base, theme.surface, state.far ? FAR_MIX : INACTIVE_MIX);
  if (!state.focused) color = mix(color, theme.surface, UNFOCUSED_MIX);
  return color;
}

export interface NodeState {
  dir: 'forward' | 'backward';
  revealed: boolean;
  active: boolean;
  far: boolean;
  focused: boolean;
}

export function nodeColor(grad: number, theme: ThemeColors, state: NodeState): LinearRgb {
  if (!state.revealed || state.far) return mix(theme.bg, theme.surface, state.far ? NODE_FAR_MIX : NODE_HIDDEN_MIX);
  let color: LinearRgb;
  if (state.dir === 'forward') color = state.active ? theme.bg : mix(theme.bg, theme.surface, NODE_IDLE_MIX);
  else {
    const base = signColor(grad, theme);
    color = state.active ? base : mix(base, theme.surface, INACTIVE_MIX);
  }
  if (!state.focused) color = mix(color, theme.surface, UNFOCUSED_MIX);
  return color;
}

export function outlineOpacity(state: { active: boolean; far: boolean; tile: boolean }): number {
  const base = state.far ? OUTLINE_FAR : state.active ? OUTLINE_ACTIVE : OUTLINE_IDLE;
  return base * (state.tile ? OUTLINE_TILE_FACTOR : 1);
}
