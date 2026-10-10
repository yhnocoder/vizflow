import type { LinearRgb, ThemeColors } from '../theme';
import { ZERO_THRESHOLD } from './scales';

export const INACTIVE_MIX = 0.55;
export const UNFOCUSED_MIX = 0.45;
export const OUTLINE_ACTIVE = 0.9;
export const OUTLINE_HIDDEN = 0.35;

export function mix(a: LinearRgb, b: LinearRgb, k: number): LinearRgb {
  return [a[0] + (b[0] - a[0]) * k, a[1] + (b[1] - a[1]) * k, a[2] + (b[2] - a[2]) * k];
}

export interface BarState {
  active: boolean;
  focused: boolean;
}

export function barColor(value: number, theme: ThemeColors, state: BarState): LinearRgb {
  const base = Math.abs(value) < ZERO_THRESHOLD ? theme.lineStrong : value > 0 ? theme.pos : theme.neg;
  let color = state.active ? base : mix(base, theme.surface, INACTIVE_MIX);
  if (!state.focused) color = mix(color, theme.surface, UNFOCUSED_MIX);
  return color;
}

export interface NodeState {
  dir: 'forward' | 'backward';
  revealed: boolean;
  active: boolean;
  focused: boolean;
}

export function nodeColor(grad: number, theme: ThemeColors, state: NodeState): LinearRgb {
  if (!state.revealed) return mix(theme.bg, theme.surface, 0.5);
  let color: LinearRgb;
  if (state.dir === 'forward') color = state.active ? theme.bg : mix(theme.bg, theme.surface, 0.6);
  else {
    const base = grad >= 0 ? theme.pos : theme.neg;
    color = state.active ? base : mix(base, theme.surface, INACTIVE_MIX);
  }
  if (!state.focused) color = mix(color, theme.surface, UNFOCUSED_MIX);
  return color;
}

export function outlineOpacity(revealed: boolean, active: boolean): number {
  return revealed && active ? OUTLINE_ACTIVE : OUTLINE_HIDDEN;
}
