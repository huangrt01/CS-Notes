
# HEARTBEAT.md

# 自发性后台任务执行配置

## 配置参数

```
# Heartbeat 间隔（单位：分钟）
HEARTBEAT_INTERVAL_MINUTES = 60  # 调整为 1 小时一次
```

## 整体思考

### Heartbeat 任务简化

**用户反馈**：
- health check 可以精简一点点
- 不用一定要推进 todo
- 调整为一小时一次
- 只需分析跟列举当前的 todos 的一些重要性之类的

**调整后的 Heartbeat 任务**：
1. **Session 状态检查**：检查 session 状态，提醒切换
2. **Todo 状态分析**：分析当前 todos 的重要性，列举给用户
3. **不主动推进 todo**：只分析和列举，不主动执行

---

# Add tasks below when you want the agent to check something periodically.

## 1. Session 状态检查

**触发条件**：每次心跳时检查

**执行逻辑**：
1. 运行 `python3 session-optimizer.py check` 检查 session 状态
2. 如果发现需要切换 session，提醒用户
3. 记录检查结果到状态文件

**检查项**：
- 消息数量（≥30条提醒，≥50条强烈建议切换）
- 运行时间（≥12小时提醒，≥24小时强烈建议切换）

**重要原则**：
- 基于 OpenClaw 现有能力，不侵入内部代码
- 使用 OpenClaw 内置的心跳机制
- 推荐使用 OpenClaw 内置的 `/reset` 命令切换 session

## 2. Todo 状态分析

**触发条件**：每次心跳时检查

**执行逻辑**：
1. 读取 todo manager（`.trae/todos/todos.json`）
2. 分析当前 todos 的状态和重要性
3. 向用户报告：
   - 🔥 高优先级 in-progress 的 AI 任务
   - ⏸️ 待执行的高优先级 AI 任务
   - ✅ 最近完成的任务

**重要经验**：
- 只分析和列举，不主动执行 todo
- 让用户了解当前 todo 状态

