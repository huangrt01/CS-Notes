# MEMORY.md - OpenClaw 长期记忆

## 与AI助手协作的经验

### 0. 笔记整理：先判断内容应该放在哪个 section
- **在添加新内容前，先浏览笔记的现有结构**，了解各个 section 的主题和范围
- **根据内容的主题，找到最合适的 section**，而不是随便找个地方就放
- **例子**：
  - ❌ 错误：把"大模型产品的成本结构"放在"产品" section
  - ✅ 正确：把"大模型产品的成本结构"放在"AI & LLM & Agent" section
- **原则**：在整理笔记时，先想清楚内容的主题，然后找到最合适的 section 存放

### 1. Markdown 笔记层级注意事项
- **注意逻辑层次关系**：在整理笔记时，要注意不同内容之间的逻辑层次关系
- **例子**：
  - ❌ 错误：`### 手机助手（豆包）` + `### 手机 Agent：存量博弈与场景变迁`（同一层级，没有体现包含关系）
  - ✅ 正确：`### 手机 Agent：存量博弈与场景变迁` + `#### 手机助手（豆包）`（体现包含关系，豆包是手机 Agent 的一个具体例子）
- **原则**：在整理笔记时，先想清楚内容之间的逻辑关系，然后用正确的 Markdown 层级（#、##、###、####）来体现

### 2. 用户消息优先原则
- 无论AI正在执行什么任务，收到用户的新消息时必须**立即回复**，绝对不能让任务阻塞对话
- 这是最高优先级的原则，高于任何后台任务

### 3. 长时间任务后台化
- 对于耗时操作（如下载大模型、安装依赖、执行长时间脚本），必须使用后台模式（background: true）运行
- 这样可以确保用户随时可以发送新消息并得到立即回复

### 4. 长期任务进度管理
- 当AI在执行复杂/长期任务（如下载大模型、安装依赖、执行长时间脚本）时，应该主动、定期（每10分钟）检查任务进度，并向用户报告，而不是等待用户询问
- 进度报告要包含：百分比、预估剩余时间

### 5. Git Commit 链接
- 每次进行 git commit 并 push 后，必须在回复中包含对应的 GitHub commit 链接。

### 6. Git 操作 SOP（第一原则：公司项目文档绝对不能 push 到 git）
- **公用 Skill 能力**：**必须实际调用** `Notes/snippets/todo-push.sh` 和 `Notes/snippets/todo-pull.sh` 作为标准 git 操作流程，**不能直接用 git 命令**
- **todo-push.sh 白名单机制**：仅允许 `Notes/`、`.trae/`、`创作/` 三个文件夹
- **todo-push.sh 黑名单机制**：绝对禁止 `公司项目/` 文件夹
- **todo-push.sh 排除模式**：排除 `.trae/logs/`、`*.pyc`、`__pycache__/`、`.DS_Store`
- **验证步骤**：每次 commit 前，先执行 `git status` 检查，或直接运行 `todo-push.sh`
- **.gitignore 配置**：确保 `**/公司项目/**` 在 .gitignore 中（已有配置）
- **公司项目/ 目录规则**：该目录下的所有内容永远不要 git add 到公开仓库

### 7. OpenClaw 长时间任务管理能力
- **后台运行模式 (`background: true`)**：
  - 配置粒度：每次工具调用（tool call）级别配置
  - 参数：`background: true`（布尔值）
  - 效果：工具在后台执行，立即返回会话ID（sessionId），用户可随时查询进度或发送新消息
- **进度定期报告**：
  - 配置粒度：可通过规则配置报告频率
  - 默认：每10分钟一次
  - 进度报告内容：包含百分比、预估剩余时间
- **用户消息优先**：
  - 优先级：用户消息 &gt; 后台任务进度报告 &gt; 后台任务执行
- **后台任务管理**：
  - 会话ID：每个后台任务分配唯一的 sessionId
  - 操作：可查询（poll）、获取日志（log）、终止（kill）后台任务
  - 工具：`process` 工具，支持 `list/poll/log/write/kill` 等操作

### 7. CS-Notes 笔记库结构（持续学习中）
- **核心文件夹**：
  - `.trae/documents/` - 文档文件夹（INBOX.md、todos管理系统.md、PROGRESS.md等）
  - `.trae/openclaw-skills/` - OpenClaw 技能（cs-notes-git-sync、cs-notes-todo-sync、todo-pull.sh、todo-push.sh）
  - `.trae/rules/` - 规则文件夹（project_rules.md）
  - `.trae/skills/` - 通用技能文件夹（code-reviewer、find-skills、fix、pr-creator）
  - `Notes/` - 可复用知识笔记
  - `创作/` - 面向输出的写作
  - `公司项目/` - 保密项目（永远不git add）
  - `Notes/snippets/` - 脚本和小工具（包括todo-pull.sh和todo-push.sh）
