# Qwen3 示例

页面入口是 `qwen3-dsl.html`。`qwen3-dsl.yaml` 只写配置、输入、面板和 section，模型 `Qwen3ForCausalLM` 及它用到的模块在包内的 `src/models/qwen3/`。`rope.md` 是 RoPE 一节的讲解文本。

YAML 的语法、校验规则和内置模块见 [DSL 设计文档](../../docs/design/pages/dsl.html)。

`qwen3-mini.yaml` 是只有一个 decoder 层的小页面，用 `layout: { levels: nested }`。以 `qwen3-mini-` 开头的几个页面只有 `layout` 一块不同，用来对比布局选项：`prepare-start`、`prepare-use`、`prepare-side` 是 `prepare` 的三个值，`gap-labels`、`gap-2` 是 `gap` 的两种写法，`span-members` 是 `span: members`，`lines` 是默认的按行分层。
