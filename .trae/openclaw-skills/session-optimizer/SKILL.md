---
name: session-optimizer
description: OpenClaw Session 优化器 - 监控 session 状态，在需要时提醒用户切换 session。包括 session 长度监控、时间显示等功能。
---

# Session Optimizer Skill

监控 OpenClaw session 状态，在需要时提醒用户切换 session。

## 核心功能

1. **Session 状态检查** - 检查当前 session 的运行时间
2. **自动警告** - 在达到阈值时提醒用户
3. **历史记录** - 记录 session 切换历史

## 使用方法

### 检查 Session 状态

```bash
cd /root/.openclaw/workspace/CS-Notes/Notes/snippets/
python3 session-optimizer.py check
```

### 记录消息

```bash
python3 session-optimizer.py log
```

### 查看历史记录

```bash
python3 session-optimizer.py history
```

### 准备重置 Session

```bash
python3 session-optimizer.py reset
```

## 阈值配置

- **消息数量警告**: 30 条消息
- **消息数量强烈建议**: 50 条消息
- **时间警告**: 12 小时
- **时间强烈建议**: 24 小时

## 状态文件

状态文件保存在：`/root/.openclaw/workspace/CS-Notes/.openclaw-session-optimizer.json`

## 重要原则

- 基于 OpenClaw 现有能力，不侵入内部代码
- 不修改 OpenClaw 源代码
- 不修改 OpenClaw 配置文件
- 仅使用 OpenClaw 已提供的功能
- 推荐使用 OpenClaw 内置的 `/reset` 命令
