---
name: trae-agent
description: Trae Agent - 基于 LLM 的通用软件工程任务 agent。遇到代码问题可尝试调用它作为另一条路。支持多 LLM 提供商（OpenAI、Anthropic、Doubao、Google Gemini、OpenRouter、Ollama），丰富的工具生态系统，轨迹记录功能。
---

# Trae Agent Skill

Trae Agent 是一个基于 LLM 的通用软件工程任务 agent。遇到代码问题可尝试调用它作为另一条路。

## 核心特性

### 1. 研究友好的设计
- 透明、模块化的架构，易于修改、扩展和分析
- 使其成为研究 AI agent 架构、进行消融研究和开发新型 agent 能力的理想平台

### 2. 多 LLM 支持
- 支持的提供商：
  - OpenAI
  - Anthropic
  - Doubao（豆包）
  - Azure
  - OpenRouter
  - Ollama
  - Google Gemini APIs

### 3. 丰富的工具生态系统
- 文件编辑
- Bash 执行
- 顺序思考
- 更多...

### 4. 交互模式
- 用于迭代开发的对话界面

### 5. 轨迹记录
- 所有 agent 动作的详细日志记录，用于调试和分析

### 6. 灵活配置
- 基于 YAML 的配置，支持环境变量

## 安装

### 要求
- UV（https://docs.astral.sh/uv/）
- 所选提供商的 API 密钥（OpenAI、Anthropic、Google Gemini、OpenRouter 等）

### 设置步骤

```bash
git clone https://github.com/bytedance/trae-agent.git
cd trae-agent
uv sync --all-extras
source .venv/bin/activate
```

## 配置

### YAML 配置（推荐）

1. 复制示例配置文件：
   ```bash
   cp trae_config.yaml.example trae_config.yaml
   ```

2. 编辑 `trae_config.yaml`，填入你的 API 凭据和偏好

## 使用方法

### 基本命令

```bash
# 简单任务执行
trae-cli run "Create a hello world Python script"

# 检查配置
trae-cli show-config

# 交互模式
trae-cli interactive
```

### 提供商特定示例

```bash
# OpenAI
trae-cli run "Fix the bug in main.py" --provider openai --model gpt-4o

# Anthropic
trae-cli run "Add unit tests" --provider anthropic --model claude-sonnet-4-20250514

# Google Gemini
trae-cli run "Optimize this algorithm" --provider google --model gemini-2.5-flash

# OpenRouter（访问多个提供商）
trae-cli run "Review this code" --provider openrouter --model "anthropic/claude-3-5-sonnet"

# Doubao（豆包）
trae-cli run "Refactor the database module" --provider doubao --model doubao-seed-1.6

# Ollama（本地模型）
trae-cli run "Comment this code" --provider ollama --model qwen3
```

## 与 CS-Notes 项目的整合

### 作为另一条路
- 遇到代码问题时，可以尝试调用 trae-agent 作为另一条路
- 与当前的方舟代码模型形成互补

### 轨迹记录
- Trae Agent 提供了详细的轨迹记录功能
- 可以用于任务执行的可观测性和调试

### 研究友好的设计
- Trae Agent 的透明、模块化架构使其成为研究 AI agent 架构的理想平台
- 可以用于研究和实验新的 agent 能力

## 相关链接

- **技术报告**：https://arxiv.org/abs/2507.23370
- **GitHub 仓库**：https://github.com/bytedance/trae-agent
- **调研报告**：`Notes/snippets/code-reading-trae-agent.md`
