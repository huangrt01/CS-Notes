---
name: VeADK-Go 技能集合
description: 根据用户的功能需求，完成与 VeADK-Go 相关的功能; 包括：直接根据需求生成 Agent；将Enio Agent转换为VeADK-Go Agent。
---

# VeADK Agent 生成

本技能可以根据用户的需求，生成符合要求的 VeADK-Go Agent 代码，或完成 VeADK-Go 相关功能。

## 触发条件

1. 用户简要描述了其功能需求，并希望构建一个 Agent 来完成；
2. 用户希望可以将已有的 Enio 代码转化为 VeADK-Go Agent 代码

## 具体步骤

下面是本技能不同的组件能力。

### 直接根据需求生成 Agent

请你遵循以下步骤：

1. 了解VeADK-Go开发框架的代码结构、功能特性以及代码示例，可以参考 `/references/common/` 目录下文档
2. 分析用户需求，生成 Agent 代码。

### Enio 代码转换为 VeADK-Go Agent

请你遵循以下步骤：
1. 了解VeADK-Go开发框架的代码结构、功能特性以及代码示例，可以参考 `/references/common/` 目录下文档
2. 分析原有 Enio 代码
3. 将原有代码改为 VeADK-Go Agent。代码特性对应关系参考 `references/converter/enio_rules.md`
4. 确保 llmagent.Config Name 字段 不包含空格和-等特殊字符。

## 后续工作

在完成 Agent 代码编写后，调用脚本保存代码产物：

- `agent_name/agent.py`：包含所有智能体的代码

其中，`agent_name` 是你认为合适的 Agent 的名称。

