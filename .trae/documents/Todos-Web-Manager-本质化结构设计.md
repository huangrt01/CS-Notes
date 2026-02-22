# Todos Web Manager 本质化结构设计

## 概述

从更本质的角度，Todos Web Manager 可以分为两大类：

### 第一类：人需要关注的内容
- **定义**：需要人去 review、去决策的内容
- **内容**：
  1. 待审核的 Plans（待 review 的 plan）
  2. 已完成的 tasks（待 review 的 AI 已完成状态的 tasks）

### 第二类：AI 工作流
- **定义**：AI 正在处理或已经处理完的内容（不需要人立即关注）
- **内容**：
  1. 进行中的 tasks
  2. 待处理的 tasks
  3. 已审核的 Plans
  4. 归档的 tasks

---

## 前端结构设计

### 主标签页设计

```
┌─────────────────────────────────────────────────────────┐
│  👤 我的待办 (默认)  │  🤖 AI 工作流        │
└─────────────────────────────────────────────────────────┘
```

### 第一类：👤 我的待办

**子标签页：**
- **待审核的 Plans** - 所有状态为 `pending` 的 Plans
- **待审核的 Tasks** - 所有状态为 `completed` 的 tasks（需要用户 review）

### 第二类：🤖 AI 工作流

**子标签页：**
- **进行中** - 所有状态为 `in-progress` 的 tasks
- **待处理** - 所有状态为 `pending` 的 tasks
- **已审核** - 所有状态为 `approved` 或 `rejected` 的 Plans
- **归档** - 所有已归档的 tasks

---

## 数据模型

### 第一类：我的待办
```javascript
// 待审核的 Plans
const pendingPlans = plans.filter(p => p.status === 'pending');

// 待审核的 Tasks
const pendingReviewTasks = tasks.filter(t => t.status === 'completed');
```

### 第二类：AI 工作流
```javascript
// 进行中的 Tasks
const inProgressTasks = tasks.filter(t => t.status === 'in-progress');

// 待处理的 Tasks
const pendingTasks = tasks.filter(t => t.status === 'pending');

// 已审核的 Plans
const reviewedPlans = plans.filter(p => p.status === 'approved' || p.status === 'rejected');

// 归档的 Tasks
const archiveTasks = [...archiveTasks];
```

---

## 用户体验优化

### 默认显示
- 打开 Web Manager 时，默认显示「👤 我的待办」标签页
- 「我的待办」标签页下，默认显示「待审核的 Plans」

### 视觉区分
- 「👤 我的待办」标签页用蓝色系配色
- 「🤖 AI 工作流」标签页用紫色系配色
- 清晰的图标和文字说明

---

## 实施计划

1. 修改主标签页结构
2. 调整子标签页逻辑
3. 更新统计数据显示
4. 优化视觉样式和配色

