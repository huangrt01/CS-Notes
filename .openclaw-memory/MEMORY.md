# MEMORY.md - OpenClaw 长期记忆

## 📝 笔记仓库原则与最佳实践

### 1. 笔记整理核心原则
- **严格遵循 project_rules.md**：整理笔记时，必须严格遵循 `.trae/rules/project_rules.md` 中的规则
- **笔记整理流程（严格执行）**：
  1. **先提取文件目录结构**：使用 markdown-toc skill 提取多个潜在目标 Markdown 文件的目录结构
  2. 先在笔记库中广泛搜索，找到最合适的已有的笔记
  3. 找到最合适的 section，而不是随便找个地方就放
  4. 将内容整合到合适的 section 中，而不是创建新文件
  5. 附上来源链接作为引用，用 markdown 格式标注链接
  6. 尽量精简语言，精炼整理笔记
  7. 格式上：减少不必要的加粗、尽量对齐原文件的格式
- **引用原则**：所有从外部材料（文章、网页、视频等）整理的内容，必须在相关章节开头或内容旁边附上来源链接作为引用
- **限制**：不要删除原笔记中的内容，只允许进行整合，不能丢失信息
- **无幻觉原则**：仅保留真实内容，不要加入不相关的信息
- **简洁原则**：笔记整理要简洁，不要啰嗦，用 bullet point 列表，简洁明了

### 2. Markdown 笔记层级注意事项
- 注意逻辑层次关系，用正确的 Markdown 层级（#、##、###、####）来体现
- 先想清楚内容之间的逻辑关系，再确定层级

### 3. 用户发资料/链接时的两种意图
- **第一种意图：精炼整理笔记**
  - 适用于：小红书文章、论文、知乎文章、公众号文章
  - 处理方式：精炼整理内容到笔记库
- **第二种意图：资料收集**
  - 适用于：视频或课程
  - 处理方式：找到视频标题，发现属于笔记库中的某一个 section，把它加入到那个 section 中，用引用 quote 的形式扩起来

### 4. Git 操作 SOP（第一原则：公司项目文档绝对不能 push 到 git）
- **必须使用** `Notes/snippets/todo-push.sh` 和 `Notes/snippets/todo-pull.sh` 作为标准 git 操作流程，**不能直接用 git 命令**
- **todo-push.sh 白名单机制**：仅允许 `Notes/`、`.trae/`、`创作/` 三个文件夹
- **todo-push.sh 黑名单机制**：绝对禁止 `公司项目/` 文件夹
- **验证步骤**：每次 commit 前，先执行 `git status` 检查，或直接运行 `todo-push.sh`
- **.gitignore 配置**：确保 `**/公司项目/**` 在 .gitignore 中
- **公司项目/ 目录规则**：该目录下的所有内容永远不要 git add 到公开仓库
- **Git Push/Pull 工作流程**：完成任务 → 运行 todo-push.sh → 生成 commit message → 运行 todo-push-commit.sh
- **Commit 链接**：每次进行 git commit 并 push 后，必须在回复中包含对应的 GitHub commit 链接

### 5. 安全意识：敏感内容绝对不允许上传到公开 GitHub 仓库
- **任何 API key、AK/SK、token、secret、password 等敏感内容，绝对不允许上传到仓库上**
- **在写入任何文件到仓库前，必须检查是否有敏感内容**
- **在 commit 前，必须检查 git status，看看有没有不该提交的文件**
- **在写入任何包含配置的文档时，必须把敏感内容替换成占位符**（API key → "YOUR_API_KEY" 等）
- **永远不要把真实的敏感内容写入到任何会被提交到仓库的文件中**

### 6. 文件组织规范
- **设计文档、使用指南等文档应该放在 .trae/documents/ 中**
- **只有代码脚本（.py、.sh 等）才应该放到 Notes/snippets/**
- **做事前先思考文件应该放在哪里**
- **先检查目标目录中已有的文件**

### 7. Todo 管理原则
- **单一数据源**：以 `.trae/todos/todos.json` 为唯一数据源
- **Plan 机制本质化重构**（2026-02-23 更新）：
  - **核心原则**：Plan 不再是独立的状态，而是 Todo 的一个字段（property）
  - **数据模型**：一个 Todo 可以有 0 或 1 个 Plan，Plan 作为 Todo 的 `plan` 字段存在
  - **Plan 结构**：
    ```json
    {
      "content": "Plan 的完整内容（Markdown）",
      "status": "pending",  // pending / approved / rejected
      "created_at": "...",
      "review_comment": "审核评论（可选）",
      "reviewed_at": "..."
    }
    ```
  - **前端展示**：不再有独立的 Plans 标签页，Plan 直接在对应的 Todo 卡片中展示（点击展开）
  - **Todo 卡片标识**：有 Plan 的 Todo 会显示「📋 有 Plan」标记
- **任务归档前的用户确认流程**：在将 completed 任务从 todos.json 移动到 archive 之前，必须增加一次由用户进行 check/反馈的流程
- **Todos 管理的正确流程**：
  1. AI 执行任务，完成编码/实现部分
  2. 到了需要用户验证/确认的阶段时，自动归类到 "User 需要做的任务" section
  3. 明确标注 Assignee: User，并且提醒用户去做
  4. 等待用户完成后，再标记为完成
- **Review 流程**：
  - **通过**: 任务自动归档，记录 Review 历史
  - **不通过**: 任务回到 `in-progress`，附带 Review 意见，AI 可以看到并改善

### 8. 用户交互原则
- **用户消息优先原则**：无论 AI 正在执行什么任务，收到用户的新消息时必须立即回复，绝对不能让任务阻塞对话
- **长时间任务后台化**：对于耗时操作，必须使用后台模式（background: true）运行
- **进度报告要求**：主动、定期（每 10 分钟）检查任务进度，向用户报告，包含百分比、预估剩余时间
- **Intervention 定义**：只要用户和 AI 说话就算一次交互，每次交互后计数清零，在两次交互之间主动推进任务
- **自主推进原则**：只有需要用户做选择题的时候才找用户确认，否则尽量自主推进一切事项

### 9. 任务交付原则
- **E2E 落地思路**：不是写了、实现了就算 done，而是要用起来、用好
- **落地的定义**：成为了本仓库在各种交互方式下的默认 setting，并体验和效果良好，而不仅是有个设计文档和玩具代码即算落地
- **任务交付要更侧重于更 solid 的端到端验证**，不仅仅是完成任务，更要进行 thorough 的端到端验证

### 10. 错误反思与改进原则
- **失败尝试记录原则**：遇到失败时在技术文档记录跑的命令和错误
- **及时沉淀经验**：避免重复犯错
- **小步快跑**：把大任务拆成多个小阶段，每个阶段都有明确的产出
- **快速迭代**：优先实现可用的最小版本，后续再逐步完善

---

## 🔧 OpenClaw 操作经验

### 1. Token 超限错误处理
- **错误信息**：`400 Total tokens of image and text exceed max message tokens`
- **解决方法**：在 TUI 中使用 `/reset` 命令恢复会话

### 2. Session 管理优化
- **重要原则**：基于 OpenClaw 现有能力，不侵入内部代码
- **推荐做法**：定期切换 session，避免 session history 过长

### 3. 长时间任务管理
- **后台运行模式 (`background: true`)**：工具在后台执行，立即返回会话ID（sessionId），用户可随时查询进度或发送新消息
- **后台任务管理**：可查询（poll）、获取日志（log）、终止（kill）后台任务

### 4. Device Token Mismatch 问题
- **问题原因**：`paired.json` 中的 token 和 `openclaw.json` 中的 token 不一致，环境变量 `OPENCLAW_GATEWAY_TOKEN` 会覆盖 `gateway.auth.token`
- **解决方案**：`openclaw gateway install --force` + `openclaw gateway restart`

---

## 📁 CS-Notes 笔记库结构
- **`.trae/documents/`** - 文档文件夹（INBOX.md、todos管理系统.md、PROGRESS.md 等）
- **`.trae/openclaw-skills/`** - OpenClaw 技能（cs-notes-git-sync、cs-notes-todo-sync、todo-pull.sh、todo-push.sh）
- **`.trae/rules/`** - 规则文件夹（project_rules.md）
- **`.trae/skills/`** - 通用技能文件夹（code-reviewer、find-skills、fix、pr-creator）
- **`.trae/todos/`** - Todo 数据文件夹（todos.json、archive/）
- **`.trae/web-manager/`** - Todos Web Manager
- **`Notes/`** - 可复用知识笔记
- **`创作/`** - 面向输出的写作
- **`公司项目/`** - 保密项目（永远不 git add）
- **`Notes/snippets/`** - 脚本和小工具（包括 todo-pull.sh 和 todo-push.sh）

---

*最后更新：2026-02-22*
