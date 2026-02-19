# Todos Web Manager - Review 流程优化

**最后更新**: 2026-02-20

## 概述

本次优化为 Todos Web Manager 添加了完整的 Review 流程，支持 AI 完成任务后等待用户 Review，用户可以通过或不通过，并附带 Review 意见。

---

## 新的任务状态


## 新的标签页顺序

```
👀 AI 已完成（待 review） → 🚀 进行中 → ⏸️ 待处理 → 📦 归档 → 📋 全部
```

**设计思路**: 优先展示需要用户关注的内容（待 Review 的任务），然后是正在进行的，再是待处理的。

---

## Review 流程

### 1. AI 标记任务为完成

AI 完成任务后，可以将任务状态标记为 `ai-completed`，进入待 Review 队列。

### 2. 用户 Review 任务

用户在 "👀 AI 已完成（待 review）" 标签页中：
- 点击任务标题查看详情（展开/折叠）
- 点击 "📋 Review" 按钮打开 Review 弹窗
- 输入可选的 Review 意见
- 选择 "✅ 通过" 或 "❌ 不通过"

### 3. Review 通过

- 任务自动归档到当前月份的归档文件中
- 记录 Review 历史（时间、通过、意见）
- 从当前任务列表中移除

### 4. Review 不通过

- 任务状态回到 `in-progress`
- 附带 Review 意见（显示在任务卡片上）
- 记录 Review 历史（时间、不通过、意见）
- AI 可以看到 Review 意见并进一步改善

---

## 待处理标签页细分

在 "⏸️ 待处理" 标签页中，新增子标签页：

- **全部**: 显示所有待处理任务
- **👤 待人处理**: 显示 assignee 为 `user` 或 `user + ai` 的任务
- **🤖 待 AI 处理**: 显示 assignee 为 `ai` 的任务

---

## 新增优化点

### 1. 任务统计面板

顶部新增统计面板，实时显示：
- 待 review 任务数
- 进行中任务数
- 待处理任务数
- 总计任务数

### 2. 任务详情展开/折叠

- 点击任务标题可以展开/折叠详情
- 详情包含：
  - 🎯 Definition of Done
  - 🔗 相关链接
  - 📋 Review 历史
- Review 意见显示在任务卡片上（黄色提示框）

---

## 后端 API 变更

### 新增 API 端点

```
POST /api/tasks/<task_id>/review
```

**请求体**:
```json
{
  "approved": true|false,
  "comment": "Review 意见（可选）"
}
```

**响应**:
```json
{
  "success": true,
  "message": "任务 xxx 已通过 review 并归档"
}
```

### 状态更新 API 增强

`PUT /api/tasks/<task_id>/status` 现在支持新状态 `ai-completed`。

---

## 数据结构变更

### 任务新增字段

- `review_comment`: string - 最新的 Review 意见
- `review_history`: array - Review 历史记录
  ```json
  [
    {
      "reviewed_at": "2026-02-20T10:00:00.000Z",
      "approved": true,
      "comment": "做得很好！"
    }
  ]
  ```

### 归档任务新增字段

- `archived_at`: string - 归档时间

---

## 使用示例

### 场景 1：AI 完成任务，等待 Review

1. AI 完成任务，调用 API 更新状态为 `ai-completed`
2. 用户在 "👀 AI 已完成（待 review）" 标签页看到任务
3. 用户点击任务标题查看详情
4. 用户点击 "📋 Review" 按钮
5. 用户输入意见，点击 "✅ 通过"
6. 任务自动归档

### 场景 2：Review 不通过，任务退回

1. 用户 Review 任务，点击 "❌ 不通过" 并附带意见
2. 任务状态回到 `in-progress`
3. Review 意见显示在任务卡片上
4. AI 看到意见后进一步改善
5. AI 再次标记为 `ai-completed` 等待 Review

---

## 相关文件

- **前端**: `.trae/web-manager/index-enhanced.html`
- **后端**: `.trae/web-manager/server.py`
- **主任务文件**: `.trae/todos/todos.json`
- **归档目录**: `.trae/todos/archive/`

---

## 总结

本次优化实现了完整的 Review 流程，让 AI 和用户之间的协作更加高效和可控。通过：

1. **新的状态**: `ai-completed` 表示 AI 完成但待 Review
2. **Review 功能**: 用户可以通过或不通过，附带意见
3. **归档机制**: 通过的任务自动归档
4. **退回机制**: 不通过的任务回到进行中，附带意见
5. **任务细分**: 待处理任务按 assignee 细分
6. **统计面板**: 实时显示任务统计
7. **详情展开**: 任务详情可展开查看

形成了完整的 AI → Review → 归档/退回的闭环流程。
