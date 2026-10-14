# Qwen3.5 示例

页面入口是 `qwen3_5-dsl.html`。`qwen3_5-dsl.yaml` 只写配置、输入和面板，模型 `Qwen3_5ForCausalLM` 及它用到的模块在包内的 `src/models/qwen3_5/`，MLP 与 eager attention 直接使用 `src/models/qwen3/` 的定义。

同一目录下的 `attention.yaml` 和 `gated-deltanet.yaml` 是旧写法，供还没有迁移的 Qwen3.5-MoE、Qwen3.8 页面引用，迁移后删除。

YAML 的语法、校验规则和内置模块见 [DSL 设计文档](../../docs/design/pages/dsl.html)。
