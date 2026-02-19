# HEARTBEAT.md

# 自发性后台任务执行配置

## 配置参数

```
# 在两次用户干预期间最多执行的任务数量（可配置，未来可以调高）
MAX_TASKS_BETWEEN_INTERVENTIONS = 8

# 最多并发的子 agent 数量
MAX_CONCURRENT_SUBAGENTS = 5

# Heartbeat 间隔（单位：分钟）
HEARTBEAT_INTERVAL_MINUTES = 30
```

## 整体思考

### 问题：依赖 heartbeat 来触发自主推进

**问题分析：**
- 目前比较依赖 heartbeat 来触发自主推进
- 如果 heartbeat 间隔太长，可能会错过一些需要及时推进的任务
- 如果 heartbeat 间隔太短，可能会导致不必要的检查和资源浪费

**解决方案：**
1. **调整 heartbeat 间隔为30分钟**：平衡及时性和资源消耗
2. **在用户干预之间自主推进任务**：不等待 heartbeat，在用户两次干预之间主动推进任务
3. **闭环自我迭代**：在执行任务过程中，如果发现对齐最终目标的新 todos，自动添加到 todo manager 中

### 自主推进原则

1. **只有需要用户做选择题的时候才找用户确认**
2. **否则尽量自主推进一切事项**
3. **用户希望：起床后，每个 todo 都到了不得不依赖他做些什么或者做决策的阶段**
4. **在用户两次交互之间主动推进任务**，不等待 heartbeat
5. **重要：只要用户和我说话就算一次交互，就可以清零计数！**
6. **Heartbeat 时，把自己想象成永动机，高中低优先级的任务均尽力自主触发推进！**


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

## 4. Todo 状态检查（新增！）

**触发条件**：每次心跳时检查

**执行逻辑**：
1. 读取 todo manager（`.trae/documents/todos管理系统.md`）
2. 识别所有在 "#### AI 需要做的任务" section 下的任务（**不管有没有明确标注 Assignee: AI**）
3. 识别所有明确标注 "Assignee: AI" 的任务
4. 向用户报告有哪些 todo 可考虑做

**报告内容**：
- ✅ 已完成的 AI 任务
- ⏸️ 待执行的 AI 任务
- 🤔 可以考虑让 AI 执行的任务

**重要经验**：
- 只要在 "#### AI 需要做的任务" section 下的任务，就算是 AI 要做的任务！不管有没有明确标注 "Assignee: AI"！
