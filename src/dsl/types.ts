import type { Nested } from '../tensor';

export type Name = string;
export type Expr = string;
export type Tex = string;
export type Text = string;
export type Path = string;
export type DimExpr = Name | number | string;
export type ModuleRef = string;
export type StepPath = string;

export type ParamDecl =
  | { type: 'int' | 'float'; default?: number }
  | { type: 'dim'; default?: number }
  | { type: 'op'; values?: Name[]; default?: Name }
  | { type: 'tex' | 'text'; default?: string };

export interface InputDecl {
  shape?: DimExpr[];
  tex?: Tex;
}

export interface WeightDecl {
  shape: DimExpr[];
  init?: 'ones' | 'zeros' | 'normal' | 'uniform';
  tex?: Tex;
}

export interface BufferDecl {
  value: Expr;
  tex?: Tex;
}

export interface NodeDecl {
  op: Expr;
  tex?: Tex;
}

export interface Instance {
  use: ModuleRef;
  params?: Record<Name, number | string>;
  inputs: Record<Name, Expr>;
  outputs?: Record<Name, Name>;
  weights?: Record<Name, Name>;
  suffix?: Tex;
  tex?: Record<Name, Tex>;
  steps?: Record<StepPath, StepPatch>;
}

export interface Group {
  body: Item[];
  steps?: Step[];
  subtitle?: Text;
}

export interface ForItem {
  for: Name;
  in: string;
  brace?: Text;
  body: Item[];
}

export type Item = Record<Name, Expr | NodeDecl | Instance | Group> | ForItem;

export interface NoteSpec {
  at: Path;
  text: Text;
  side?: 'left' | 'right' | 'above' | 'below';
}

export interface Step {
  id: Name;
  show: Expr | Expr[];
  notes?: NoteSpec[];
  active?: Expr[];
  label?: Tex;
  title?: Text;
  text?: Text;
}

export interface StepText {
  title?: Text;
  text?: Text;
}

export interface StepPatch {
  title?: Text;
  label?: Tex;
  text?: Text;
  append?: Text;
  notes?: NoteSpec[];
}

export interface ModuleFile {
  module: Name;
  import?: Record<Name, ModuleRef>;
  subtitle?: Text;
  params?: Record<Name, ParamDecl>;
  inputs: Record<Name, InputDecl>;
  outputs: Name[];
  weights?: Record<Name, WeightDecl>;
  buffers?: Record<Name, BufferDecl>;
  body: Item[];
  steps?: Step[];
  backward?: Record<Name, StepText>;
}

export type PanelSpec = 'loss' | Record<string, unknown>;

export type SectionBlockSpec =
  | { text: Text }
  | { lead: Text }
  | { buttons: { label: Text; go: Path }[] }
  | 'rope_dial'
  | 'rope_curve'
  | { rope_compare: { scores: Path | Path[]; weights: Path | Path[]; ids: Path } };

export type SectionRowSpec = SectionBlockSpec[] | { columns?: (number | string)[]; cells: SectionBlockSpec[][] };

export interface SectionSpec {
  title?: Text;
  rope?: { q: Path | Path[]; k: Path | Path[]; rows?: [number, number]; head?: number; theta: number; dh: number; positions?: number };
  src?: string;
  rows?: SectionRowSpec[];
}

export interface PageFile extends Omit<ModuleFile, 'module' | 'inputs' | 'outputs'> {
  title: Text;
  lede?: Text;
  hint?: Text;
  camera?: Record<string, unknown>;
  overview?: StepText & { notes?: NoteSpec[] };
  modules?: Record<Name, Omit<ModuleFile, 'module'>>;
  dims: Record<Name, number>;
  real?: Record<Name, number | string>;
  vocab?: string[];
  init?: { seed: number };
  inputs: Record<Name, { tokens: string[] } | { value: Nested }>;
  values?: Record<Path, Nested>;
  panels?: PanelSpec[];
  sections?: SectionSpec[];
}
