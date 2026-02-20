# HEARTBEAT.md

# 自发性后台任务执行配置

## 配置参数

```
# 在两次用户干预期间最多执行的任务数量（可配置，未来可以调高）
MAX_TASKS_BETWEEN_INTERVENTIONS = 8

# 最多并发的子 agent 数量
MAX_CONCURRENT_SUBAGENTS = 5
```

## 心跳任务

### 1. 检查 todo manager 并执行任务（自发性）

**触发条件**：每次心跳时检查

**执行逻辑**：
1. 检查上次用户干预的时间
2. 检查两次干预期间已执行的任务数量
3. 如果已执行数量 < MAX_TASKS_BETWEEN_INTERVENTIONS：
   - 从 todo manager 中取新任务
   - 优先执行 Assignee: AI 的任务
   - 优先执行 Priority: high 的任务
   - 跳过 Feedback Required: 是 的任务
   - 使用子 agent 执行任务
4. 如果已执行数量 >= MAX_TASKS_BETWEEN_INTERVENTIONS：
   - 等待下次用户干预
   - 用户干预后重置计数

### 2. 闭环自我迭代（新增！）

**触发条件**：在执行任务过程中

**执行逻辑**：
1. 如果发现对齐最终目标的新 todos
2. 自动添加到 todo manager 中
3. 形成闭环自我迭代：执行 → 发现新任务 → 添加新任务 → 继续执行

---

# Add tasks below when you want the agent to check something periodically.

## 3. Session 状态检查（新增！）

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
