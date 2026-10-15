export type Expression = string;
export type Reference = string;
export type Path = string;
export type TemplateText = string;
export type Latex = string;
export type TensorDtype = 'bool' | 'uint8' | 'int8' | 'int16' | 'int32' | 'int64' | 'fp8_e4m3' | 'fp8_e5m2' | 'fp16' | 'bf16' | 'fp32' | 'fp64';
export type Scalar = string | number | boolean;
export type Value = Scalar | Value[] | ExpressionValue;
export type Dimension = number | Expression;
export type ArgType = 'dim' | 'int' | 'float' | 'bool' | 'str' | 'dtype' | 'module';
export type ArgDeclaration = ArgType | ArgField;
export type ModuleList = `list[${string}]`;
export type LoopBinding = `${string} = ${string}[${string}]`;
export type LoopVariable = string;
export type GridArrangement = 'expand' | 'stack';
export type ArgValue = Expression | number | boolean;
export type Args = Record<string, ArgValue> | (string | Record<string, ArgValue>)[];
export type Declaration = Dimension[] | TensorField | ListField;
export type ModuleInitializer = string | ModuleInstance | ModuleListInstance;
export type Step = NamedStep | LoopStep;
export type Panel = 'cost' | LossPanel | SequencePanel | AttentionPanel | MatrixPanel | CodePanel | ImplementationsPanel;

export interface ExpressionValue {
  expr: Expression;
}

export interface ArgField {
  type: ArgType;
  default?: Value;
  options?: Scalar[];
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

export interface Page {
  title: string;
  camera?: Camera;
  import?: Record<string, string>;
  preamble?: string;
  model: string;
  config: PageConfig;
  data: Record<string, Value>;
  modules: Record<string, Module>;
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
  inputs: Record<string, Declaration>;
  outputs: Record<string, Declaration>;
  init?: Initializers;
  forward: Assignment[];
  tex?: Record<Reference, Latex>;
  formula?: Latex;
  fold?: Fold;
  layout?: Layout;
  steps?: Step[];
  unshown?: Reference[];
  kernel?: boolean;
  grid?: Record<string, Dimension>;
  shared?: Reference[];
  save?: Reference[];
  backward?: Assignment[];
  backward_steps?: Step[];
}

export interface ModuleFile extends Module {
  module: string;
}

export interface Initializers {
  [name: string]: ModuleInitializer | Record<string, TensorInit> | undefined;
  weights?: Record<string, TensorInit>;
  buffers?: Record<string, TensorInit>;
}

export interface TensorInit {
  shape: Dimension[];
  dtype?: TensorDtype | Expression;
  init?: 'ones' | 'zeros' | 'normal' | 'uniform';
  value?: Expression;
  present?: Expression;
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
  [name: string]: Expression;
}

export interface Layout {
  place?: Record<Reference, Placement>;
  merge?: Reference[][];
  grid?: Record<string, GridArrangement>;
}

export interface Placement {
  with?: Reference;
  before?: Reference;
  after?: Reference;
  left?: Reference;
  right?: Reference;
}

export interface Explanation {
  label?: TemplateText;
  title?: TemplateText;
  text?: TemplateText;
  append?: TemplateText;
}

export interface Backward extends Explanation {
  notes?: Note[];
}

export interface Fold extends Explanation {
  backward?: Backward;
}

export interface StepOptions extends Explanation {
  show?: Expression | Expression[];
  subtitle?: TemplateText;
  expand?: boolean | Expression;
  flat?: boolean;
  brace?: boolean;
  formula?: Latex;
  notes?: Note[];
  fold?: Fold;
  backward?: Backward;
  edit?: Record<Path, StepOptions>;
}

export interface NamedStep extends StepOptions {
  id: string;
  steps?: Step[];
  loop?: never;
}

export interface LoopStep {
  id: string;
  loop: LoopBinding | LoopVariable;
  defaults: StepOptions;
  rules?: LoopRule[];
}

export interface LoopRule extends StepOptions {
  at: number;
}

export interface Note {
  at: Reference;
  text: TemplateText;
  side?: 'left' | 'right';
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

export interface ImplementationsPanel extends PanelText {
  implementations: Reference[];
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
  theta: number;
  d_head: Expression | number;
  positions: number;
}
