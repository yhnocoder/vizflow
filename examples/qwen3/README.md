# Qwen3 DSL 约定

本目录是新 DSL 的设计草稿，解析器和内置模块尚未同步。页面入口是 `qwen3-dsl.html`，模型定义在 `qwen3-dsl.yaml`。

结构定义见 [dsl.ts](dsl.ts)。其中 `declare class` 只描述 YAML 数据结构，不创建运行时对象。每条赋值只有一个结果名、同一作用域名称唯一、引用有效及形状匹配等约束，由后续 DSL 校验器检查。表达式和模板的字符串语法仍以本页约定为准。

## 结构

- 页面用 `model` 指定根模块；顶层 `config`、`inputs` 提供本次示例的配置和输入值。
- 模块的 `config`、`inputs`、`outputs` 声明接口；模块只能读取声明的配置、输入及自身定义的对象。
- `init` 声明参数、buffer 和子模块；`forward` 只描述计算；`steps` 按顺序组织讲解。
- `tex` 集中定义数学名称；`formula` 和 `fold` 描述模块整体公式与折叠展示。子模块不额外生成总览步骤。

配置声明没有额外选项时可以简写为 `T: dim`。输入输出统一用字典：

```yaml
inputs:
  tokens: { type: list, dtype: str }
  ids: { type: tensor, shape: [$T], dtype: int64 }
outputs:
  loss: { type: tensor, shape: [], dtype: fp32 }
```

维度按 `T、V、d_model、d_head、d_ff、num_heads、num_kv_heads、num_layers` 排列，其余配置放在后面。分别表示序列长度、词表大小、隐藏维度、单头维度、MLP 中间维度、query head 数、kv head 数和层数。

## 计算与引用

配置中的简单引用写 `$d_model`，计算写 `{ expr: num_heads * d_head }`。`forward` 已经是表达式环境，直接使用变量名；同一作用域内不允许重复定义名字。

```yaml
forward:
  - a: norm(x)
  - b: norm(a)
  - result: nll_loss(logp=logp, y=y)
  - loss: result.loss
```

单输出直接得到张量，多输出得到具名结果。同一个子模块实例可以调用多次，共享参数，每次调用的中间结果独立。

`use: list[Qwen3DecoderLayer]` 配合 `count` 创建参数独立的层列表。调用列表时依次执行各层，返回最后一层输出。`real` 只描述真实规模，不改变示例计算。

## 讲解

`show: a` 展示张量；`show: norm` 展示模块调用及其讲解。同一实例调用多次时，用 `show: { producer_of: a }` 指定产生 `a` 的那次调用。

`show: { all: true }` 展示全图；`show: { inputs: true, parameters: true }` 展示输入张量与参数。`steps` 补丁按步骤 ID 定位，`text` 替换正文，`append` 追加正文。

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
