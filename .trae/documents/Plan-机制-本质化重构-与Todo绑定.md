# Plan 机制本质化重构 - Plan 与 Todo 绑定

## 问题分析

当前 Plan 机制存在以下本质问题：

1. **Plans 是独立状态** - Plans 和 Todos 是两个独立的数据集，没有明确的关联
2. **前端有专门的 Plans 标签页** - 给人一种 Plans 是独立于 Todos 的错觉
3. **缺少绑定关系** - 一个 Plan 对应哪个 Todo？不明确

## 本质化思考

**Plan 不应该是一个独立的状态，而应该是 Todo 的一个属性！**

每个 Todo 可以有 0 或 1 个 Plan，Plan 是 Todo 的补充信息，而不是独立的实体。

## 新的数据模型

### Todo 数据结构（更新）

```json
{
  "id": "todo-xxx",
  "title": "任务标题",
  "status": "pending",
  "priority": "high",
  "assignee": "ai",
  "feedback_required": false,
  "created_at": "...",
  
  "is_plan_todo": true,  // 是否需要 Plan
  "plan": {             // Plan 作为 Todo 的一个字段
    "content": "Plan 的完整内容（Markdown）",
    "status": "pending",  // pending / approved / rejected
    "created_at": "...",
    "review_comment": "审核评论（可选）",
    "reviewed_at": "..."
  }
}
```

### Plan 状态

- `pending` - Plan 待审核
- `approved` - Plan 已审核通过
- `rejected` - Plan 已审核拒绝

## 前端展示变化

### 移除独立的 Plans 标签页

- ❌ 不再有专门的 "Plans" 标签页
- ✅ Plan 信息直接在对应的 Todo 中展示

### Todo 展示时可展开 Plan

当一个 Todo 有 Plan 时：
- 在 Todo 卡片上显示一个 "📋 有 Plan" 的标记
- 用户可以点击展开/折叠查看 Plan 内容
- Plan 的审核状态也在 Todo 卡片上显示

### 统计面板变化

- ❌ 不再有独立的 "待审核 Plans" 统计
- ✅ 统计 Todo 的各种状态，Plan 状态作为 Todo 的补充信息

## 实施计划

### 阶段 1：设计与准备
1. ✅ 创建设计文档
2. 分析现有数据，准备迁移方案

### 阶段 2：后端改造
1. 修改 `server.py`，更新数据模型
2. 更新 API 端点，支持新的数据结构
3. 保持向后兼容（可选）

### 阶段 3：前端改造
1. 修改 `index-enhanced.html`，移除 Plans 标签页
2. 更新 Todo 展示逻辑，支持展开 Plan
3. 更新统计面板

### 阶段 4：数据迁移
1. 尝试将现有的 Plans 与对应的 Todo 绑定
2. 如果无法匹配，保留为备份

## 优势

1. **更本质的模型** - Plan 是 Todo 的属性，不是独立实体
2. **更简洁的 UI** - 不再有独立的 Plans 标签页
3. **更清晰的关联** - Plan 和 Todo 的关系一目了然
4. **更符合直觉** - 用户看到 Todo 就能看到它的 Plan
