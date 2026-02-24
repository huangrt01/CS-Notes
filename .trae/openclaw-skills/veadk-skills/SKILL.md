---
name: VeADK 技能集合
description: 根据用户的功能需求，完成与 VeADK 相关的功能。
---

# VeADK Agent 生成

本技能可以根据用户的需求，生成符合要求的 VeADK Agent 代码，或完成 VeADK 相关功能。

## 触发条件

1. 用户简要描述了其功能需求，并希望构建一个 Agent 来完成；
2. 用户希望可以将已有的 Langchain/Langgraph 代码转化为 VeADK Agent 代码
3. 用户希望可以将已有的 Dify 工作流转化为 VeADK Agent 代码

## 具体步骤

下面是本技能不同的组件能力。

### 直接根据需求生成 Agent

请你遵循以下步骤：

1. 分析用户需求，生成对应的 Agent 系统结构，参考 `references/generator/analyze.md`
2. 提示词优化，参考 `references/generator/refine_prompt.md`
3. 生成 Agent 代码，参考 `references/generator/coding.md`

### Langchain 代码转换为 VeADK Agent

请你遵循以下步骤：

1. 分析原有 Langchain 或 Langgraph 代码
2. 将原有代码改为 VeADK Agent，对应关系详见 `references/converter/langchain_rules.md`
3. 参照 `references/common/` 目录内的文档来生成 VeADK 代码

### Dify 工作流转换为 VeADK Agent

请你遵循以下步骤：

1. 分析原有 Dify 工作流 DSL（一般为一个 Yaml 格式文件）
2. 将原有代码改为 VeADK Agent，对应关系详见 `references/converter/dify_rules.md`
3. 参照 `references/common/` 目录内的文档来生成 VeADK 代码

## 后续工作

在完成 Agent 代码编写后，调用脚本保存代码产物：

- `agent_name/__init__.py`: 固定内容为 `from . import agent # noqa`
- `agent_name/agent.py`：包含所有智能体的代码

其中，`agent_name` 是你认为合适的 Agent 的名称。

脚本调用方法为：

```bash
python save_file.py --path ... --content ...
```
