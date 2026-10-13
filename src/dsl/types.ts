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

export interface ExpressionValue {
  expr: Expression;
}

export interface ArgField {
  type: ArgType;
  default?: Value;
}

export interface TensorField {
  type: 'tensor';
  shape: Dimension[];
  dtype: TensorDtype | Expression;
}

export interface ListField {
  type: 'list';
  dtype: 'str' | TensorDtype | Expression;
}

export interface PageFile {
  title: string;
  camera?: Camera;
  import?: Record<string, string>;
  preamble?: string;
  model: string;
  config: PageConfig;
  data: Record<string, Value>;
  modules?: Record<string, Module>;
  panels?: Panel[];
  sections?: Section[];
}

export interface PageConfig {
  [arg: string]: Value | Tokenizer | Record<string, RealConfig> | undefined;
  seed?: number;
  tokenizer?: Tokenizer;
  real?: Record<string, RealConfig>;
}

export interface RealConfig {
  [dim: string]: string | number;
  name: string;
}

export interface Tokenizer {
  vocab: string[];
}

export interface Camera {
  direction?: [number, number, number];
  polar?: [number, number];
  follow?: boolean;
}

export interface Module {
  import?: Record<string, string>;
  args?: Record<string, ArgDeclaration>;
  inputs: Record<string, InputDeclaration>;
  outputs: Record<string, InputDeclaration>;
  init?: Initializers;
  forward: Assignment[];
  tex?: Record<Reference, Latex>;
  formula?: Latex;
  fold?: Fold;
  steps?: Step[];
  unshown?: Reference[];
}

export interface ModuleFile extends Module {
  module: string;
}

export interface Initializers {
  [name: string]: ModuleInitializer | Record<string, Weight | Buffer> | undefined;
  weights?: Record<string, Weight>;
  buffers?: Record<string, Buffer>;
}

export interface Weight {
  shape: Dimension[];
  init?: 'ones' | 'zeros' | 'normal' | 'uniform';
  present?: Expression;
}

export interface Buffer extends TensorField {
  value: Expression;
}

export interface ModuleInstance {
  use: string;
  args?: Args;
  count?: never;
}

export interface ModuleListInstance {
  use: ModuleList;
  count: Dimension;
  args?: Args;
}

export interface Assignment {
  [resultName: string]: Expression;
}

export interface ShowAll {
  all: true;
}

export interface ShowInputs {
  inputs?: boolean;
  parameters?: boolean;
}

export interface ShowProducer {
  producer_of: Reference;
}

export interface Explanation {
  label?: TemplateText;
  title?: TemplateText;
  subtitle?: TemplateText;
  text?: TemplateText;
  append?: TemplateText;
}

export type Backward = Omit<Explanation, 'subtitle'>;

export interface Fold extends Backward {
  backward?: Backward;
}

export interface StepOptions extends Explanation {
  show?: Show;
  segment?: TemplateText;
  brace?: Brace;
  notes?: Note[];
  formula?: Latex;
  folded?: boolean;
  beside?: Reference;
  inline?: boolean;
  fold?: Fold;
  backward?: Backward;
  steps?: Record<string, StepOptions>;
}

export interface NamedStep extends StepOptions {
  id: string;
  loop?: never;
}

export interface LoopStep {
  id: string;
  loop: LoopBinding;
  defaults: StepOptions;
  rules?: StepRule[];
}

export interface StepRule extends StepOptions {
  at: number;
}

export interface Brace {
  title: TemplateText;
  subtitle?: TemplateText;
}

export interface Note {
  at: Reference;
  text: TemplateText;
  side?: 'left' | 'right' | 'above' | 'below';
}

export interface PanelText {
  title?: TemplateText;
  note?: TemplateText;
}

export interface LossPanel extends PanelText {
  loss: LossOptions;
}

export interface LossOptions {
  parts?: Reference;
  hint?: TemplateText;
}

export interface SequencePanel extends PanelText {
  sequence: true;
  tokens: Reference;
  input: Reference;
  label: Reference;
  labels?: TemplateText[];
  sgd?: Sgd;
  random?: boolean;
  reset?: boolean;
}

export interface Sgd {
  params: 'all' | Reference[];
  lr: number;
  label?: TemplateText;
}

export interface AttentionPanel extends PanelText {
  attention: Reference;
  ids: Reference;
}

export interface MatrixPanel extends PanelText {
  matrices: MatrixDisplay[];
  highlight?: Highlight;
  rows?: Reference;
}

export interface MatrixDisplay {
  matrix: Reference;
  grad?: boolean;
  caption?: TemplateText;
  limit?: number;
}

export interface Highlight {
  node: Reference;
  all?: boolean;
}

export interface CodePanel extends PanelText {
  code: string;
}

export interface Section {
  title: TemplateText;
  src: string;
  rope?: RopeDisplay;
}

export interface RopeDisplay {
  q: Reference;
  k: Reference;
  rows: [number, number];
  head: number;
  theta: Expression | number;
  d_head: Expression | number;
  positions: number;
}
