# Trae Agent 调研报告

## 概述

本文档记录对 Trae Agent（https://github.com/bytedance/trae-agent）的调研结果。

## 什么是 Trae Agent？

**Trae Agent** 是一个基于 LLM 的通用软件工程任务 agent。它提供了强大的 CLI 接口，可以理解自然语言指令，并使用各种工具和 LLM 提供商执行复杂的软件工程工作流。

**项目状态**：项目仍在积极开发中。

**与其他 CLI Agent 的区别**：
- Trae Agent 提供了透明、模块化的架构，研究人员和开发者可以轻松修改、扩展和分析
- 使其成为**研究 AI agent 架构、进行消融研究和开发新型 agent 能力**的理想平台
- 这种**"研究友好的设计"**使学术界和开源社区能够贡献和构建基础 agent 框架，促进 AI agent 快速发展领域的创新

## ✨ 核心特性

### 1. 🌊 Lakeview
- 为 agent 步骤提供简短、简洁的总结

### 2. 🤖 多 LLM 支持
- 支持的提供商：
  - OpenAI
  - Anthropic
  - Doubao（豆包）
  - Azure
  - OpenRouter
  - Ollama
  - Google Gemini APIs

### 3. 🛠️ 丰富的工具生态系统
- 文件编辑
- Bash 执行
- 顺序思考
- 更多...

### 4. 🎯 交互模式
- 用于迭代开发的对话界面

### 5. 📊 轨迹记录
- 所有 agent 动作的详细日志记录，用于调试和分析

### 6. ⚙️ 灵活配置
- 基于 YAML 的配置，支持环境变量

### 7. 🚀 易于安装
- 简单的基于 pip 的安装

## 🚀 安装

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

## ⚙️ 配置

### YAML 配置（推荐）

1. 复制示例配置文件：
   ```bash
   cp trae_config.yaml.example trae_config.yaml
   ```

2. 编辑 `trae_config.yaml`，填入你的 API 凭据和偏好：

```yaml
agents:
  trae_agent:
    enable_lakeview: true
    model: trae_agent_model  # Trae Agent 的模型配置名称
    max_steps: 200  # agent 最大步骤数
    tools:  # Trae Agent 使用的工具
      - bash
      - str_replace_based_edit_tool
      - sequentialthinking
      - task_done

model_providers:  # 模型提供商配置
  anthropic:
    api_key: your_anthropic_api_key
    provider: anthropic
  openai:
    api_key: your_openai_api_key
    provider: openai

models:
  trae_agent_model:
    model_provider: anthropic
    model: claude-sonnet-4-20250514
    max_tokens: 4096
    temperature: 0.5
```

**注意**：`trae_config.yaml` 文件被 git 忽略，以保护你的 API 密钥。

### 使用 Base URL

在某些情况下，我们需要为 API 使用自定义 URL。只需在 `provider` 后添加 `base_url` 字段，以下面的配置为例：

```
openai:
    api_key: your_openrouter_api_key
    provider: openai
    base_url: https://openrouter.ai/api/v1
```

**注意**：对于字段格式，仅使用空格。不允许使用制表符（\t）。

### 环境变量（替代方案）

你也可以使用环境变量配置 API 密钥，并将它们存储在 .env 文件中：

```bash
export OPENAI_API_KEY="your-openai-api-key"
export OPENAI_BASE_URL="your-openai-base-url"
export ANTHROPIC_API_KEY="your-anthropic-api-key"
export ANTHROPIC_BASE_URL="your-anthropic-base-url"
export GOOGLE_API_KEY="your-google-api-key"
export GOOGLE_BASE_URL="your-google-base-url"
export OPENROUTER_API_KEY="your-openrouter-api-key"
export OPENROUTER_BASE_URL="https://openrouter.ai/api/v1"
export DOUBAO_API_KEY="your-doubao-api-key"
export DOUBAO_BASE_URL="https://ark.cn-beijing.volces.com/api/v3/"
```

### MCP 服务（可选）

要启用模型上下文协议（MCP）服务，在配置中添加 `mcp_servers` 部分：

```yaml
mcp_servers:
  playwright:
    command: npx
    args:
      - "@playwright/mcp@0.0.27"
```

**配置优先级**：命令行参数 > 配置文件 > 环境变量 > 默认值

**传统 JSON 配置**：如果使用旧的 JSON 格式，请参阅 [docs/legacy_config.md](docs/legacy_config.md)。我们建议迁移到 YAML。

## 📖 使用方法

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
trae-cli run "Generate documentation" --provider openrouter --model "openai/gpt-4o"

# Doubao（豆包）
trae-cli run "Refactor the database module" --provider doubao --model doubao-seed-1.6

# Ollama（本地模型）
trae-cli run "Comment this code" --provider ollama --model qwen3
```

### 高级选项

```bash
# 自定义工作目录
trae-cli run "Add tests for utils module" --working-dir /path/to/project

# 保存执行轨迹
trae-cli run "Debug authentication" --trajectory-file debug_session.json

# 强制生成补丁
trae-cli run "Update API endpoints" --must-patch

# 使用自定义设置的交互模式
trae-cli interactive --provider openai --model gpt-4o --max-steps 30
```

## 🐳 Docker 模式命令

### 准备
**重要**：你需要确保在环境中配置了 Docker。

### 使用方法

```bash
# 指定 Docker 镜像在新容器中运行任务
trae-cli run "Add tests for utils module" --docker-image python:3.11

# 指定 Docker 镜像在新容器中运行任务并挂载目录
trae-cli run "write a script to print helloworld" --docker-image python:3.12 --working-dir test_workdir/

# 通过 ID 附加到现有 Docker 容器（`--working-dir` 与 `--docker-container-id` 一起使用时无效）
trae-cli run "Update API endpoints" --docker-container-id 91998a56056c

# 指定 Dockerfile 的绝对路径来构建环境
trae-cli run "Debug authentication" --dockerfile-path test_workspace/Dockerfile

# 指定本地 Docker 镜像文件（tar 归档）的路径来加载
```

## 📁 项目结构

```
trae-agent/
├── CONTRIBUTING.md
├── docs/                    # 文档
├── evaluation/              # 评估
├── .github/                 # GitHub 相关
├── .gitignore
├── LICENSE
├── Makefile
├── .pre-commit-config.yaml
├── pyproject.toml
├── .python-version
├── README.md               # 项目说明（本文档）
├── server/                 # 服务器
├── tests/                  # 测试
├── trae_agent/             # 主代码
├── trae_config.json.example
├── trae_config.yaml.example
├── uv.lock
└── .vscode/               # VS Code 配置
```

## 🔗 相关链接

- **技术报告**：https://arxiv.org/abs/2507.23370
- **GitHub 仓库**：https://github.com/bytedance/trae-agent
- **Discord**：https://discord.gg/VwaQ4ZBHvC
- **路线图**：docs/roadmap.md
- **贡献指南**：CONTRIBUTING.md

## 💡 与 CS-Notes 项目的潜在整合点

### 1. 任务执行
- Trae Agent 可以作为任务执行器，替代或补充当前的方舟代码模型
- 支持 Doubao（豆包）模型，与火山引擎方舟 API 兼容

### 2. 工具生态系统
- Trae Agent 提供了丰富的工具生态系统（文件编辑、Bash 执行、顺序思考等）
- 可以与当前的 todo 管理系统整合

### 3. 轨迹记录
- Trae Agent 提供了详细的轨迹记录功能
- 可以用于任务执行的可观测性和调试

### 4. 研究友好的设计
- Trae Agent 的透明、模块化架构使其成为研究 AI agent 架构的理想平台
- 可以用于研究和实验新的 agent 能力

## 总结

### Trae Agent 的核心价值
1. **研究友好的设计**：透明、模块化的架构，易于修改、扩展和分析
2. **多 LLM 支持**：支持 OpenAI、Anthropic、Doubao、Google Gemini、OpenRouter、Ollama 等
3. **丰富的工具生态系统**：文件编辑、Bash 执行、顺序思考等
4. **轨迹记录**：详细的日志记录，用于调试和分析
5. **灵活配置**：基于 YAML 的配置，支持环境变量

### 与 CS-Notes 项目的整合潜力
- 可以作为任务执行器
- 可以与当前的 todo 管理系统整合
- 可以用于任务执行的可观测性和调试
- 可以用于研究和实验新的 agent 能力
