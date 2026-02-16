# Top Lean AI 榜单监控 Skill

## 概述

这个 OpenClaw Skill 用于监控 Top Lean AI 榜单的更新，支持手动调用和每日定时任务。

## 功能

### 1. 手动命令

**检查更新**
```
top-lean-ai-monitor check
```

**查看状态**
```
top-lean-ai-monitor status
```

### 2. 定时任务

每天早上 9 点自动检查榜单更新（通过 OpenClaw 的 cron 调度）。

### 3. 飞书通知

- 发现新公司时自动通过 OpenClaw 原生能力发送飞书通知
- 使用 `[OPENCLAW_MESSAGE_SEND]` 标记触发通知

## 文件结构

```
top-lean-ai-monitor/
├── skill.json    # Skill 配置
├── main.py       # Skill 主程序
└── README.md     # 本文档
```

## 配置

### skill.json

```json
{
  "name": "top-lean-ai-monitor",
  "version": "1.0.0",
  "commands": [
    {
      "name": "check",
      "description": "检查榜单更新",
      "handler": "handle_check"
    },
    {
      "name": "status",
      "description": "查看监控状态",
      "handler": "handle_status"
    }
  ],
  "scheduled_tasks": [
    {
      "name": "daily-check",
      "description": "每日检查榜单更新",
      "cron": "0 9 * * *",
      "handler": "handle_daily_check"
    }
  ]
}
```

## 使用示例

### 在 OpenClaw 对话中使用

```
你: "检查一下 Top Lean AI 榜单"
OpenClaw: "好的，我来调用 top-lean-ai-monitor Skill..."
[Skill 执行]
OpenClaw: "没有新公司上榜，当前共有 45 家公司"
```

### 通过 Lark 触发（如果已集成）

```
你（在 Lark 中）: "检查榜单"
Lark Bot → OpenClaw → Skill 执行 → 结果返回
```

### 定时任务

无需手动操作，OpenClaw 会在每天早上 9 点自动运行 `handle_daily_check`。

## 数据来源

- 榜单网站: https://leanaileaderboard.com/
- 数据源: Google Sheets CSV 导出

## 依赖

- 主脚本: `/Users/bytedance/CS-Notes/top-lean-ai-monitor.py`
- 状态文件: `.top-lean-ai-state.json` (自动生成)

## 作者

AI
