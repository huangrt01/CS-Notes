# LLM 智能解析 - 重新设计方案

**日期**: 2026-02-19  
**作者**: AI + 用户

---

## 概述

基于用户的想法，重新设计 LLM 智能解析口述式任务模板的方案，减少 hard code 规则，更多依赖 LLM 的智能理解能力。

---

## 核心原则

1. **减少 hard code 规则**，更多依赖 LLM 的智能理解能力
2. **优先使用 OpenClaw 消息触发**，而不是复杂的 API 调用逻辑
3. **先尝试 LLM 智能解析**，如果没通过再走基础流程
4. **与 Todos Web Manager 融合**，提供统一的用户体验

---

## 用户的核心观点

### 1. 关于"解析 LLM 口述式智能模板"的未来价值

- **可能不需要 hard code 的规则**
- **LLM 本身就能理解意思，知道该怎么转写为 Todo**
- **可以尝试先调用 LLM 看能否通过校验，如果没通过再走基础流程**

### 2. 关于交互方式

- **最好是通过触发一条 OpenClaw 消息来触发**，而不是写一套调 API 的逻辑
- **和 Todos Web Manager 融合也有类似的问题**

### 3. 用户的要求

- **重新 review 并设计整个方案**
- **整理出更明确的 Todo**

---

## 重新设计的方案

### 方案一：LLM 优先的智能解析流程

```
1. 用户通过语音/文字输入任务
   ↓
2. 先调用 LLM 智能解析
   ↓
3. 检查解析结果是否通过校验
   ├─ 通过 → 直接使用解析结果
   └─ 不通过 → 走基础流程（正则表达式等）
   ↓
4. 结构化 Todo 写入 todos.json
   ↓
5. 通过 OpenClaw 消息触发后续流程
```

### 方案二：OpenClaw 消息触发的交互方式

```
1. 用户在飞书中发送语音/文字消息
   ↓
2. OpenClaw 接收消息
   ↓
3. OpenClaw 触发智能解析 Skill
   ↓
4. Skill 解析任务并写入 todos.json
   ↓
5. Skill 通过 OpenClaw 消息通知用户
   ↓
6. 用户在 Todos Web Manager 中查看和管理任务
```

### 方案三：与 Todos Web Manager 的融合

```
1. Todos Web Manager 提供语音输入按钮
   ↓
2. 用户点击按钮开始录音
   ↓
3. 语音转写为文字（使用浏览器 Web Speech API 或 Whisper）
   ↓
4. 调用 LLM 智能解析
   ↓
5. 解析结果显示在界面上，用户可以编辑
   ↓
6. 用户确认后写入 todos.json
   ↓
7. 通过 OpenClaw 消息触发后续流程
```

---

## 更明确的 Todo

### Todo 1: 实现 LLM 优先的智能解析流程

**Priority**: High  
**Assignee**: AI  
**Status**: Pending

**Definition of Done**:
- 实现先调用 LLM 智能解析的功能
- 实现解析结果的校验逻辑
- 如果 LLM 解析没通过，自动回退到基础流程（正则表达式等）
- 测试验证 LLM 优先的解析流程

**User Requirements**:
- 可能不需要 hard code 的规则
- LLM 本身就能理解意思，知道该怎么转写为 Todo
- 可以尝试先调用 LLM 看能否通过校验，如果没通过再走基础流程

---

### Todo 2: 实现 OpenClaw 消息触发的交互方式

**Priority**: High  
**Assignee**: AI  
**Status**: Pending

**Definition of Done**:
- 创建智能解析 OpenClaw Skill
- Skill 可以通过 OpenClaw 消息触发
- Skill 可以解析任务并写入 todos.json
- Skill 可以通过 OpenClaw 消息通知用户
- 测试验证 OpenClaw 消息触发的交互方式

**User Requirements**:
- 最好是通过触发一条 OpenClaw 消息来触发，而不是写一套调 API 的逻辑
- 和 Todos Web Manager 融合也有类似的问题

---

### Todo 3: 实现与 Todos Web Manager 的融合

**Priority**: High  
**Assignee**: AI  
**Status**: Pending

**Definition of Done**:
- 在 Todos Web Manager 中提供语音输入按钮
- 实现语音转写为文字的功能（使用浏览器 Web Speech API 或 Whisper）
- 调用 LLM 智能解析
- 解析结果显示在界面上，用户可以编辑
- 用户确认后写入 todos.json
- 通过 OpenClaw 消息触发后续流程
- 测试验证与 Todos Web Manager 的融合

**User Requirements**:
- 和 Todos Web Manager 融合也有类似的问题

---

### Todo 4: 重新设计整体方案并整理明确的 Todo

**Priority**: High  
**Assignee**: AI  
**Status**: In Progress

**Definition of Done**:
- 重新 review 并设计整个方案
- 整理出更明确的 Todo
- 创建方案设计文档
- 同步到 todos.json

**User Requirements**:
- 重新 review 并设计整个方案
- 整理出更明确的 Todo

---

## 下一步行动

1. ✅ 创建方案设计文档（本文档）
2. ⏸️ 实现 LLM 优先的智能解析流程（Todo 1）
3. ⏸️ 实现 OpenClaw 消息触发的交互方式（Todo 2）
4. ⏸️ 实现与 Todos Web Manager 的融合（Todo 3）
5. ⏸️ 将这些 Todo 同步到 todos.json

---

## 总结

本文档基于用户的想法，重新设计了 LLM 智能解析口述式任务模板的方案，核心要点是：

1. **减少 hard code 规则**，更多依赖 LLM 的智能理解能力
2. **优先使用 OpenClaw 消息触发**，而不是复杂的 API 调用逻辑
3. **先尝试 LLM 智能解析**，如果没通过再走基础流程
4. **与 Todos Web Manager 融合**，提供统一的用户体验

并整理出了 4 个更明确的 Todo。
