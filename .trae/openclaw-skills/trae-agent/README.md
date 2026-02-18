# Trae Agent OpenClaw Skill

调用 trae-agent 执行复杂任务，利用其强可观测性和完整轨迹记录。

## 概述

这个 Skill 整合了 OpenClaw 和 trae-agent 的能力：

- **OpenClaw**：作为交互界面（易用、多渠道、自然语言对话）
- **trae-agent**：作为执行复杂任务的引擎（可观测性强、轨迹记录完整）

**一加一大于二！**

## 安装

### 前置条件

1. trae-agent 已安装在 `/root/.openclaw/workspace/trae-agent`
2. trae-agent 已配置好（trae_config.yaml）
3. uv 已安装

### 配置

确保 `trae_config.yaml` 已正确配置：

```yaml
model_providers:
    doubao:
        api_key: YOUR_DOUBAO_API_KEY
        provider: doubao
        base_url: https://ark.cn-beijing.volces.com/api/v3

models:
    trae_agent_model:
        model_provider: doubao
        model: doubao-seed-2-0-code-preview-260215
        max_tokens: 4096
        temperature: 0.5
```

## 使用方法

### 命令行使用

```bash
cd /root/.openclaw/workspace/CS-Notes/.trae/openclaw-skills/trae-agent

# 执行任务
python3 main.py "你的任务描述"

# 指定工作目录
python3 main.py "你的任务描述" --working-dir /path/to/dir
```

### 作为 OpenClaw Skill 使用

在 OpenClaw 中调用这个 Skill 来执行复杂任务。

## 任务类型建议

### OpenClaw 直接执行（简单任务）

- 创建简单文件
- 简单的文件编辑
- 运行简单命令
- 读取和写入文件

### trae-agent 执行（复杂任务）

- 笔记知识整合（需要多步骤、复杂推理）
- 大规模代码重构
- 复杂的文档整理和归纳
- 需要强可观测性的任务
- 需要完整轨迹记录的任务
- 需要详细执行步骤的任务

## 优势对比

| 维度 | OpenClaw | trae-agent | 整合后 |
|------|----------|------------|--------|
| **易用性** | ✅ 优秀 | ⚠️ 一般 | ✅ 优秀 |
| **可观测性** | ⚠️ 弱 | ✅ 强 | ✅ 强 |
| **轨迹记录** | ❌ 无 | ✅ 完整 | ✅ 完整 |
| **自然语言** | ✅ 优秀 | ⚠️ 一般 | ✅ 优秀 |
| **多渠道** | ✅ 优秀 | ❌ 无 | ✅ 优秀 |

## 输出结果

执行任务后会返回：

```json
{
  "success": true,
  "stdout": "trae-agent 的标准输出",
  "stderr": "trae-agent 的标准错误",
  "returncode": 0,
  "trajectory_file": "/path/to/trajectory.json",
  "task_description": "任务描述",
  "working_dir": "/path/to/workspace",
  "timestamp": "2026-02-17T20:00:00"
}
```

## 轨迹文件

trae-agent 会生成完整的轨迹文件（JSON 格式），包含：

- 每个步骤的详细信息
- LLM 的输入输出
- Token 使用统计
- 工具调用记录
- Lakeview 总结

## 示例

### 示例 1：笔记知识整合

```bash
python3 main.py "Read the file Notes/AI-Agent-Product&PE.md, summarize the key points about OpenClaw, and add a new section '## OpenClaw 核心能力总结' at the end of the file"
```

### 示例 2：复杂文档整理

```bash
python3 main.py "Organize the content in .trae/documents/INBOX.md, extract new tasks, and add them to the Pending section in .trae/documents/todos管理系统.md"
```

## 注意事项

1. **任务超时**：默认超时时间是 10 分钟，复杂任务可能需要更长时间
2. **工作目录**：默认工作目录是 CS-Notes 仓库
3. **轨迹文件**：轨迹文件保存在 `trae-agent/trajectories/` 目录
4. **Token 使用**：trae-agent 的 Token 使用会单独统计

## 下一步

- [ ] 集成到 OpenClaw 的 Skill 系统
- [ ] 添加更多示例
- [ ] 优化错误处理
- [ ] 添加任务进度反馈

## 相关链接

- trae-agent GitHub: https://github.com/bytedance/trae-agent
- trae-agent 评估报告: `.trae/documents/trae-agent-评估报告.md`
