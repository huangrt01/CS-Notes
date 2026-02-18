# Todos Markdown 格式规范

## 概述

为 Todos Web Manager 制定严格的 Markdown 格式规范，确保 todos管理系统.md 的格式统一，便于解析和显示。

---

## 格式规范

### 1. 任务条目格式

**标准格式：**

```markdown
* [ ] <任务标题>
  - Priority：<high|medium|low>
  - Assignee：<AI|User|User + AI>
  - Feedback Required：<是|否>
  - Links：<链接列表>
  - Definition of Done：
    * <验收标准 1>
    * <验收标准 2>
    * <验收标准 3>
  - Progress：
    * ✅ <已完成的进度 1>
    * ✅ <已完成的进度 2>
    * ⏳ <待完成的进度>
  - Started At：<YYYY-MM-DD>
  - Completed At：<YYYY-MM-DD>
```

### 2. 优先级

**可选值：**
- `high`：高优先级
- `medium`：中优先级
- `low`：低优先级

**示例：**
```markdown
  - Priority：high
```

### 3. Assignee

**可选值：**
- `AI`：由 AI 执行
- `User`：由用户执行
- `User + AI`：由用户和 AI 共同执行

**示例：**
```markdown
  - Assignee：AI
```

### 4. Feedback Required

**可选值：**
- `是`：需要用户反馈/确认
- `否`：可以直接执行完成

**示例：**
```markdown
  - Feedback Required：否
```

### 5. Links

**格式：**
- 多个链接用逗号分隔
- 相对路径或绝对路径都可以
- 可以是 Markdown 链接格式

**示例：**
```markdown
  - Links：`.trae/documents/INBOX.md`、`.trae/documents/TEMPLATES.md`
  - Links：https://github.com/bytedance/trae-agent
```

### 6. Definition of Done

**格式：**
- 使用 `*` 列表
- 每个验收标准用一行

**示例：**
```markdown
  - Definition of Done：
    * 跑通 trae-agent client
    * 设计一些需求让 trae-agent 来改笔记代码
    * 评估它改的效果如何
    * 对比 OpenClaw + 方舟代码模型的效果
```

### 7. Progress

**格式：**
- 使用 `*` 列表
- 已完成的用 `✅` 标记
- 待完成的用 `⏳` 标记

**示例：**
```markdown
  - Progress：
    * ✅ trae-agent 已成功跑通，创建了测试文件 test_trae_agent.txt
    * ✅ trae-agent 成功修改了笔记文件 test_trae_agent_note.md
    * ⏳ 下一步：设计更多测试需求，评估它改笔记代码的能力
```

### 8. Started At / Completed At

**格式：**
- ISO 日期格式：`YYYY-MM-DD`

**示例：**
```markdown
  - Started At：2026-02-17
  - Completed At：2026-02-17
```

---

## 状态标记

### 待处理（Pending）

```markdown
* [ ] <任务标题>
```

### 进行中（In Progress）

```markdown
* [ ] <任务标题>
```

### 已完成（Completed）

```markdown
* [x] <任务标题>
```

---

## Section 结构

### 标准 Section 结构

```markdown
## 当前任务列表

### 进行中 (In Progress)

#### AI 需要做的任务

* [ ] <任务 1>
  - Priority：high
  - Assignee：AI
  - Feedback Required：否
  ...

#### User 需要做的任务

* [ ] <任务 2>
  - Priority：high
  - Assignee：User
  - Feedback Required：否
  ...

---

### 待处理 (Pending)

#### AI 需要做的任务（优先执行）

* [ ] <任务 3>
  - Priority：high
  - Assignee：AI
  - Feedback Required：否
  ...

#### 其他待处理任务

* [ ] <任务 4>
  - Priority：high
  - Assignee：AI
  - Feedback Required：是
  ...

---

### 已完成 (Completed)

已完成任务已归档至：[TODO_ARCHIVE.md](TODO_ARCHIVE.md)
```

---

## 解析规则

### 任务列表解析

1. 查找所有以 `* [ ]` 或 `* [x]` 开头的行
2. 读取下一行，直到遇到下一个 `* [` 或新的 Section
3. 解析每个字段（Priority、Assignee、Feedback Required、Links、Definition of Done、Progress、Started At、Completed At）

### 字段解析

1. **Priority**：匹配 `- Priority：` 后的内容
2. **Assignee**：匹配 `- Assignee：` 后的内容
3. **Feedback Required**：匹配 `- Feedback Required：` 后的内容
4. **Links**：匹配 `- Links：` 后的内容，逗号分隔
5. **Definition of Done**：匹配 `- Definition of Done：` 后的列表
6. **Progress**：匹配 `- Progress：` 后的列表
7. **Started At**：匹配 `- Started At：` 后的内容
8. **Completed At**：匹配 `- Completed At：` 后的内容

---

## 示例

### 完整示例

```markdown
* [ ] 评估 trae-agent 的能力，对比 OpenClaw + 方舟代码模型，评估它改笔记代码的效果
  - Priority：high
  - Assignee：AI
  - Feedback Required：否
  - Links：trae-agent/、Notes/snippets/code-reading-trae-agent.md、.trae/documents/trae-agent-评估报告.md
  - Definition of Done：
    * 跑通 trae-agent client
    * 设计一些需求让 trae-agent 来改笔记代码
    * 评估它改的效果如何
    * 对比 OpenClaw + 方舟代码模型的效果
  - Progress：
    * ✅ trae-agent 已成功跑通，创建了测试文件 test_trae_agent.txt
    * ✅ trae-agent 成功修改了笔记文件 test_trae_agent_note.md
    * ✅ 评估报告已创建：`.trae/documents/trae-agent-评估报告.md`
    * ✅ 用户已确认：在 CS-Notes 项目中使用 trae-agent
    * ✅ 用户已确认：整合方式 - trae agent 作为较复杂任务时的 skill 工具调用
  - Started At：2026-02-17
  - Completed At：2026-02-17
```

---

## 注意事项

1. **缩进**：使用 2 个空格缩进
2. **冒号**：使用中文冒号 `：`（不是英文冒号 `:`）
3. **列表**：使用 `*` 作为列表标记
4. **状态**：待处理用 `[ ]`，已完成用 `[x]`
5. **日期**：使用 ISO 格式 `YYYY-MM-DD`
6. **链接**：多个链接用顿号 `、` 分隔（不是逗号）
