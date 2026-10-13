export type Expression = string;
export type Reference = string;
export type TemplateText = string;
export type Latex = string;
export type ModuleList = `list[${string}]`;
export type LoopBinding = `${string} = ${string}[${string}]`;
export type TensorDtype = 'bool' | 'uint8' | 'int8' | 'int16' | 'int32' | 'int64' | 'fp8_e4m3' | 'fp8_e5m2' | 'fp16' | 'bf16' | 'fp32' | 'fp64';
export type Scalar = string | number | boolean;
export type Value = Scalar | Value[] | ExpressionValue;
export type Dimension = number | Expression;
export type ArgType = 'dim' | 'int' | 'float' | 'bool' | 'dtype' | 'module';
export type ArgDeclaration = ArgType | ArgField;
export type ArgValue = Expression | number | boolean;
export type Args = Record<string, ArgValue> | (string | Record<string, ArgValue>)[];
export type InputDeclaration = TensorField | ListField;
export type ModuleInitializer = string | ModuleInstance | ModuleListInstance;
export type Show = Reference | Reference[] | ShowAll | ShowInputs | ShowProducer;
export type Step = NamedStep | LoopStep;
export type Panel = 'cost' | LossPanel | SequencePanel | AttentionPanel | MatrixPanel | CodePanel;

export declare class ExpressionValue {
  expr: Expression;
}

export declare class ArgField {
  type: ArgType;
  default?: Value;
}

export declare class TensorField {
  type: 'tensor';
  shape: Dimension[];
  dtype: TensorDtype | Expression;
}

export declare class ListField {
  type: 'list';
  dtype: 'str' | TensorDtype | Expression;
}

export declare class Page {
  title: string;
  camera?: Camera;
  import?: Record<string, string>;
  preamble?: string;
  model: string;
  config: PageConfig;
  inputs: Record<string, Value>;
  modules: Record<string, Module>;
  panels?: Panel[];
  sections?: Section[];
  values?: Record<Reference, Value>;
}

export declare class PageConfig {
  [arg: string]: Value | Tokenizer | Record<string, RealConfig> | undefined;
  seed?: number;
  tokenizer?: Tokenizer;
  real?: Record<string, RealConfig>;
}

export declare class RealConfig {
  [dim: string]: string | number;
  name: string;
}

export declare class Tokenizer {
  vocab: string[];
}

export declare class Camera {
  direction?: [number, number, number];
  polar?: [number, number];
  follow?: boolean;
}

export declare class Module {
  import?: Record<string, string>;
  args?: Record<string, ArgDeclaration>;
  inputs: Record<string, InputDeclaration>;
  outputs: Record<string, InputDeclaration>;
  init?: Initializers;
  forward: Assignment[];
  tex?: Record<Reference, Latex>;
  formula?: Latex;
  fold?: Fold;
  steps: Step[];
  unshown?: Reference[];
}

export declare class ModuleFile extends Module {
  module: string;
}

export declare class Initializers {
  [name: string]: ModuleInitializer | Record<string, Weight | Buffer> | undefined;
  weights?: Record<string, Weight>;
  buffers?: Record<string, Buffer>;
}

export declare class Weight extends TensorField {
  init?: string;
}

export declare class Buffer extends TensorField {
  value: Expression;
}

export declare class ModuleInstance {
  use: string;
  args?: Args;
  count?: never;
}

export declare class ModuleListInstance {
  use: ModuleList;
  count: Dimension;
  args?: Args;
}

export declare class Assignment {
  [resultName: string]: Expression;
}

export declare class ShowAll {
  all: true;
}

export declare class ShowInputs {
  inputs?: boolean;
  parameters?: boolean;
}

export declare class ShowProducer {
  producer_of: Reference;
}

export declare class Explanation {
  label?: TemplateText;
  title?: TemplateText;
  subtitle?: TemplateText;
  text?: TemplateText;
  append?: TemplateText;
}

export declare class Backward extends Explanation {
  notes?: Note[];
}

export declare class Fold extends Explanation {
  backward?: Backward;
}

export declare class StepOptions extends Explanation {
  show?: Show;
  segment?: TemplateText;
  brace?: Brace;
  notes?: Note[];
  formula?: Latex;
  folded?: boolean;
  beside?: Reference;
  fold?: Fold;
  backward?: Backward;
  steps?: Record<string, StepOptions>;
}

export declare class NamedStep extends StepOptions {
  id: string;
  loop?: never;
}

export declare class LoopStep {
  id: string;
  loop: LoopBinding;
  defaults: StepOptions;
  rules?: StepRule[];
}

export declare class StepRule extends StepOptions {
  at: number;
}

export declare class Brace {
  title: TemplateText;
  subtitle?: TemplateText;
}

export declare class Note {
  at: Reference;
  text: TemplateText;
  side?: 'left' | 'right';
}

export declare class PanelText {
  title?: TemplateText;
  note?: TemplateText;
}

export declare class LossPanel extends PanelText {
  loss: LossOptions;
}

export declare class LossOptions {
  parts?: Reference;
  hint?: TemplateText;
}

export declare class SequencePanel extends PanelText {
  sequence: true;
  tokens: Reference;
  input: Reference;
  label: Reference;
  labels?: TemplateText[];
  sgd?: Sgd;
  random?: boolean;
  reset?: boolean;
}

export declare class Sgd {
  params: 'all' | Reference[];
  lr: number;
  label?: TemplateText;
}

export declare class AttentionPanel extends PanelText {
  attention: Reference;
  ids: Reference;
}

export declare class MatrixPanel extends PanelText {
  matrices: MatrixDisplay[];
  highlight?: Highlight;
  rows?: Reference;
}

export declare class MatrixDisplay {
  matrix: Reference;
  grad?: boolean;
  caption?: TemplateText;
  limit?: number;
}

export declare class Highlight {
  node: Reference;
  all?: boolean;
}

export declare class CodePanel extends PanelText {
  code: string;
}

export declare class Section {
  title: TemplateText;
  src: string;
  rope?: RopeDisplay;
}

export declare class RopeDisplay {
  q: Reference;
  k: Reference;
  rows: [number, number];
  head: number;
  theta: number;
  d_head: Expression | number;
  positions: number;
}
