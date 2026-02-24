# Dify 与 VeADK 对应规则

你可以通过下面的介绍，来了解 Dify 与 VeADK 对应规则。具体的 VeADK 定义方法可以参照 `references/common/` 目录中的内容。

## Dify 节点类型

- LLM 节点：请参照 VeADK Agent
- 知识检索节点：请参照 `references/common/knowledgebase.md` 中的知识库定义和使用方法
- 直接回复节点：请使用 Python 语言实现
- Agent：请参照 VeADK Agent

逻辑分支节点：

- 条件分支、迭代、并行：请使用 Python 语言实现

其它节点：

- 代码执行：定义一个 Agent ，参考 `references/common/tools.md` 中的代码沙箱执行工具定义和使用方法

当遇到无法直接对应到 VeADK 节点类型的情况时，你可以考虑使用自定义的 Python 代码及 VeADK 逻辑来实现。
