# Qwen3 DSL 约定

本目录是新 DSL 的设计草稿，解析器和内置模块尚未同步。页面入口是 `qwen3-dsl.html`，模型定义在 `qwen3-dsl.yaml`。

结构定义见 [dsl.ts](dsl.ts)。其中 `declare class` 只描述 YAML 数据结构，不创建运行时对象。每条赋值只有一个结果名、同一作用域名称唯一、引用有效及形状匹配等约束，由后续 DSL 校验器检查。表达式和模板的字符串语法仍以本页约定为准。

## 结构

- 页面用 `model` 指定根模块；顶层 `config` 提供本次示例的配置，`data` 提供根模块的输入值，按名字对应根模块的 `inputs` 声明。根模块的 `args` 按同名从页面 `config` 取值，`seed`、`tokenizer`、`real` 只属于页面。
- 模块的 `args`、`inputs`、`outputs` 声明接口；模块只能读取声明的参数、输入及自身定义的对象。
- `init` 声明参数、buffer 和子模块；`forward` 只描述计算；`steps` 按顺序组织讲解。
- `tex` 集中定义数学名称；`formula` 和 `fold` 描述模块整体公式与折叠展示。子模块不额外生成总览步骤。

参数声明没有额外选项时可以简写为 `d_head: dim`。输入输出统一用字典：

```yaml
inputs:
  tokens: { type: list, dtype: str }
  ids: { type: tensor, shape: [T], dtype: int64 }
outputs:
  loss: { type: tensor, shape: [], dtype: fp32 }
```

维度按 `V、d_model、d_head、d_ff、num_heads、num_kv_heads、num_layers` 排列，其余配置放在后面。分别表示词表大小、隐藏维度、单头维度、MLP 中间维度、query head 数、kv head 数和层数。`real` 是字典，键是规格名，值里的 `T` 表示真实规模下的序列长度。

## 形状绑定

`inputs` 的 `shape` 和 `dtype` 里出现的名字，如果不是 `args` 中声明的参数，就在调用时从实际输入绑定；已经绑定或声明过的名字只做检查。`outputs` 里的名字必须已经绑定，用来检查结果。例如 decoder 的 `x: { shape: [T, d_model], dtype: dtype }` 从输入得到 `T` 和 `dtype`，`d_model` 已在 `args` 中声明，只检查输入的第二维是否等于它。

- 序列长度 `T` 是数据的维度，不作为参数传入。同一个实例可以用不同的 `T` 调用，例如 prefill 和 decode。
- `init` 只能使用 `args` 中的名字，不能使用从输入绑定的名字。这和 HF 的 `__init__` 一致：建参数时需要的维度（`d_model`、`d_head`、`num_heads` 等）都由调用者传入，例如 `Linear` 的 `in_features: d_model`、`RMSNorm` 的 `hidden_size: d_model`。
- 参数的 dtype 由页面 `config.dtype` 统一设定，作用于整个模型，相当于 `model.to(torch.bfloat16)`。模块的 `args` 不声明 dtype，`init` 也不读它。输入的 `dtype` 从实际输入绑定，用来检查输出。输入和权重的 dtype 不一致时，由运算报错，例如 `Linear` 的输入是 fp32、权重是 bf16。有意的转换在 `forward` 里写出来，例如 `z.float()`。
- 依赖输入形状的常量在 `forward` 里计算，例如 `M: causal_mask(T)`，不写成 buffer。

## 计算与引用

字符串按所在的位置分三类：表达式、模板和 LaTeX。`$` 只在模板里出现，表示公式。

```yaml
# 表达式：args 的取值、count、shape、dtype、forward、show、notes.at，以及面板和 section 里的张量引用。
# 直接写表达式，名字就是引用，不加 $。
init:
  q_proj:
    use: Linear
    args: { in_features: d_model, out_features: num_heads * d_head, bias: false }
inputs:
  x: { type: tensor, shape: [T, d_model], dtype: dtype }
forward:
  - M: causal_mask(T)
  - zf: z.float()
steps:
  - id: float
    show: zf

# 模板：label、title、subtitle、text、append、brace、note、hint。
# 普通文字；$...$ 是 LaTeX 公式，{{ ... }} 插入表达式的值。
    label: '$\viz{zf}$'
    title: 转成 float32
    text: |
      q 有 {{ num_heads }} 个 head，每个 head 宽 $d_h={{ d_head }}$。

# LaTeX：tex 的值和 formula。整个值就是 LaTeX，不加 $。
tex:
  h_norm: '\mathbf{h}_{\mathrm{norm}}'
formula: |-
  \step{res1}{\viz{a}=\viz{x}+\viz{att}}
```

表达式里的字符串字面量写引号，dtype 名（`bf16`、`fp32`）和导入的模块名（`SiLU`）是表达式里可用的名字。

页面的 `config` 和 `data` 是数据，按 YAML 字面值读取；需要计算时写 `{ expr: ... }`。含 `[`、`:` 等字符的表达式在 YAML 里加引号。

```yaml
data:
  tokens: [the, cat, sat, on]
  ids: { expr: 'tokens_to_ids(tokens[:-1])' }
```

`forward` 同一作用域内不允许重复定义名字：

```yaml
forward:
  - a: norm(x)
  - b: norm(a)
  - result: nll_loss(logp=logp, y=y)
  - loss: result.loss
```

单输出直接得到张量，多输出得到具名结果。同一个子模块实例可以调用多次，共享参数，每次调用的中间结果独立。

子模块的 `args` 可以写成字典，也可以写成列表。列表里的名字表示用调用者同名的值传入，单键字典表示指定值，名字拼错时报错：

```yaml
self_attn:
  use: Qwen3Attention
  args: [d_head, num_heads, num_kv_heads, theta]
layers:
  use: list[Qwen3DecoderLayer]
  count: num_layers
  args: [d_head, d_ff, num_heads, num_kv_heads, { theta: 10 }]
```

`use: list[Qwen3DecoderLayer]` 配合 `count` 创建参数独立的层列表。调用列表时依次执行各层，返回最后一层输出。`real` 只描述真实规模，不改变示例计算。

内置的 `Linear` 和 `Embedding` 使用 PyTorch 的参数名：`Linear` 的 `args` 是 `in_features`、`out_features`、`bias`，`Embedding` 是 `num_embeddings`、`embedding_dim`；两者的参数都叫 `weight`，形状与 PyTorch 相同。权重的初始化方式写在模块自己的 `weights` 声明里，使用处不写。

## 参数

参数按 `config.seed` 随机初始化。参数的路径沿 `init` 树，和 PyTorch `state_dict` 的键一致，例如 `layers[0].self_attn.q_proj.weight`、`lm_head.weight`。

## 讲解

`show: a` 展示张量；`show: norm` 展示模块调用及其讲解。同一实例调用多次时，用 `show: { producer_of: a }` 指定产生 `a` 的那次调用。步骤的 `id` 就是要展示的名字时，`show` 可以省略。

`show: { all: true }` 展示全图；`show: { inputs: true, parameters: true }` 展示输入张量与参数。`steps` 补丁按步骤 ID 定位，`text` 替换正文，`append` 追加正文。

`forward` 的每个结果都必须被某个步骤展示：直接展示它，或展示产生它的模块调用，或在子模块的步骤里作为输入展示。没有被展示的结果由校验器报错；确实不需要单独展示的结果，写进模块的 `unshown` 列表。

```yaml
- id: layers
  loop: layer = layers[i]
  defaults:
    show: layer
    subtitle: 第 {{i + 1}} 层
  rules:
    - at: -1
      subtitle: 最后一层
```

`layer` 是当前实例，`i` 从 0 开始；可以引用 `layers[i + 1]`。先合并默认配置与匹配规则，再求值模板，越界引用报错。

同一个张量在不同模块入口的重复展示由布局副本负责，不为展示增加 `identity` 运算。副本单独命名的配置尚未确定。

## 公式与插值

- `$...$`、`$$...$$` 包围 LaTeX 公式。
- `\viz{h_norm}` 引用 `tex` 中的数学名称，只做替换，不调整排版。追加下标时写 `{\viz{h_norm}}_t`。
- `{{ ... }}` 插入值；普通标量直接显示，非标量显式指定 `matrix`、`join` 等格式。
- `preamble` 定义数学宏；`\step{步骤 ID}{公式片段}` 关联公式与讲解步骤。长 `formula` 使用 YAML 多行块。

```yaml
tex:
  h_norm: '\mathbf{h}_{\mathrm{norm}}'
text: |
  $$\pfl{\viz{h_norm}}={{ h_norm.grad | matrix:3 }}$$
```
