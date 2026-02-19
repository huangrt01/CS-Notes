# MEMORY.md - OpenClaw 长期记忆

## 与AI助手协作的经验

### 0. 笔记整理：严格遵循 project_rules.md！
- **核心原则**：整理笔记时，必须严格遵循 `.trae/rules/project_rules.md` 中的规则！
- **笔记整理流程（严格执行）**：
  1. **先在笔记库中广泛搜索**，找到最合适的已有的笔记
  2. **找到最合适的 section**，而不是随便找个地方就放
  3. **将内容整合到合适的 section 中**，而不是创建新文件
  4. **附上来源链接作为引用**，用 markdown 格式标注链接
  5. **尽量精简语言，精炼整理笔记**
  6. **格式上**：减少不必要的加粗、尽量对齐原文件的格式
- **引用原则**：所有从外部材料（文章、网页、视频等）整理的内容，必须在相关章节开头或内容旁边附上来源链接作为引用，用 markdown 格式标注链接
- **限制**：请不要删除原笔记中的内容，只允许进行整合，不能丢失信息
- **重要经验**：
  - ❌ 错误：直接创建新文件，而不是找到已有的合适的笔记
  - ✅ 正确：先在笔记库中广泛搜索，找到最合适的已有的笔记，然后将内容整合到合适的 section 中

### 1. 笔记整理：先判断内容应该放在哪个 section
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

### 8. 微信公众号文章获取经验（2026-02-17 沉淀）
- **成功方法**：使用 `web_fetch` 工具可以成功获取微信公众号文章内容
- **测试结果**：
  - 测试 URL：https://mp.weixin.qq.com/s/gyEbK_UaUO3AeQvuhhRZ6g
  - 状态：200 OK
  - 内容：成功获取了完整的"年末 AI 回顾"文章内容
- **结论**：
  - 微信公众号文章的反爬机制没有那么强
  - `web_fetch` 工具可以直接使用，不需要复杂的配置
  - 可以通过这个方式来获取微信公众号文章的内容
- **经验沉淀**：
  - 遇到需要获取微信公众号文章内容时，直接使用 `web_fetch` 工具即可
  - 不需要尝试复杂的反爬绕过方法
  - 确保下次不犯错：直接用 `web_fetch`

### 9. OpenClaw Token 超限错误处理（2026-02-17 沉淀）
- **错误信息**：`400 Total tokens of image and text exceed max message tokens`
- **错误原因**：消息中的图片和文本总 token 数超过了模型允许的最大限制
- **解决方法**：
  - 在 TUI 中使用 `/reset` 命令可以恢复会话
  - 这个命令会重置会话状态，清除可能导致 token 超限的上下文
- **经验沉淀**：
  - 遇到 token 超限错误时，首先尝试 `/reset` 命令
  - 这是一个快速恢复会话的有效方法
  - 参考资料：https://www.answeroverflow.com/m/1469215145874948199

### 10. Session 管理与 Token Usage 优化实践（2026-02-17 执行）
- **问题背景**：
  - 遇到 token 超限错误，同时今天下午消耗了几千万 token
  - 根本原因：长期使用一个 session，导致 session history 过长
- **执行的工作**：
  1. **笔记重新组织**：
     - 将 "5000 万美元 ARR 的 AI 应用公司" 移到更前面的位置
     - 新建了 `## 头部 AI 产品与公司` section
     - GitHub Commit: https://github.com/huangrt01/CS-Notes/commit/0386137
  2. **设计优化方案**：
     - 创建了 `session-management-optimization.md` 设计文档
     - 设计了 3 个优化方案：Session 长度监控与自动切换、Context 智能压缩与摘要、Token 使用监控与优化
     - 规划了 4 个实施阶段：Phase 1-4
  3. **实现监控脚本**：
     - 创建了 `session-monitor.py`：Session 监控脚本，支持状态查看、重置、消息记录、自动警告
     - 创建了 `top-lean-ai-monitor.py`：Top Lean AI 榜单监控脚本框架
  4. **Todo 添加**：
     - 在 todos管理系统.md 中添加了两个高优先级 todo
- **经验总结**：
  - 用户说"按你的计划执行，没必要反复询问我了" → 领会意图，直接执行
  - 小步快跑，先实现可用的最小版本，后续再逐步完善
  - 创建 PROGRESS.md 记录执行进度

### 11. OpenClaw Device Token Mismatch 问题解决（2026-02-17 发现）
- **问题现象**：`gateway connect failed: Error: unauthorized: device token mismatch (rotate/reissue device token)`
- **问题原因**：
  1. `paired.json` 中的 token 和 `openclaw.json` 中的 token 不一致
  2. **关键发现**：环境变量 `OPENCLAW_GATEWAY_TOKEN` 会覆盖 `gateway.auth.token`！
- **发现过程**：
  - 通过 Reddit 链接找到关键线索：https://www.reddit.com/r/clawdbot/comments/1qujzgm/openclaw_gateway_token_mismatch_error/
  - 通过 `env | grep -i openclaw` 发现了环境变量 `OPENCLAW_GATEWAY_TOKEN`
  - 当前环境变量值：`59ac1f34670bb1c61a7bef9e29745b55507f0bb9170b35b1`
  - paired.json token：`fad65644b7374e54b6f512a0a3af3f87`
  - openclaw.json token：`8a952b9ebe4779d29c6bf018196e5b5d67b43ea5f77237ee`
  - 三个 token 都不一样！
- **解决方案（来自官方文档）**：
  1. `openclaw gateway install --force`
  2. `openclaw gateway restart`
- **参考文档**：
  - Reddit 讨论：https://www.reddit.com/r/clawdbot/comments/1qujzgm/openclaw_gateway_token_mismatch_error/
  - Reddit 讨论 2：https://www.reddit.com/r/openclaw/comments/1r18u6b/mismatching_tokens/
  - OpenClaw 官方文档：https://docs.openclaw.ai/gateway/troubleshooting

### 12. Session 工具时间显示问题修复（2026-02-17 执行）
- **问题现象**：session-optimizer.py 每次运行都显示仅运行 0.0h
- **问题原因**：
  1. 状态文件没有固定保存位置，每次运行都创建新文件
  2. 时间显示精度不够，短时间内用 `.1f` 格式化会显示 0.0h（例如 5 秒 = 0.001388 小时）
- **修复内容**：
  1. 固定状态文件保存位置：`/root/.openclaw/workspace/CS-Notes/.openclaw-session-optimizer.json`
  2. 改进时间显示：同时显示"小时分秒"和"小数小时"（两位小数）
     - 修复前：`Session 已运行: 0.0 小时`
     - 修复后：`Session 已运行: 0小时 1分 2秒 (0.02 小时)`
- **测试结果**：修复后可以正确显示时间
- **经验总结**：在做时间显示时，要考虑精度问题，短时间内用小数小时可能会显示 0.0

### 13. OpenClaw Browser 工具使用经验（2026-02-17 执行）
- **问题现象**：browser 工具报错 `device token mismatch`
- **问题原因**：环境变量 `OPENCLAW_GATEWAY_TOKEN` 和配置文件 `openclaw.json` 中的 token 不一致
- **解决过程**：
  1. 执行 `openclaw gateway restart` 重启 gateway
  2. 重启后 browser 工具状态恢复正常
  3. **Chrome 扩展连接步骤**：
     - 打开 Chrome 浏览器
     - 点击 OpenClaw 扩展图标（在浏览器工具栏右上角）
     - 选择一个标签页连接
- **Browser 工具状态说明**：
  - `enabled: true` - Browser 工具已启用
  - `running: false` - 浏览器未运行
  - `cdpReady: false` - Chrome DevTools Protocol 未就绪
  - 需要 Chrome 扩展连接标签页后才能使用
- **参考文档**：
  - Reddit 讨论：https://www.reddit.com/r/openclaw/comments/1r5ghfi/openclaw_docker_local_gateway_control_ui_wont/
- **Control UI 配对流程（新版本）**：
  1. Stop the gateway
  2. Delete both ~/.openclaw/devices/pending.json and paired.json
  3. Make sure your openclaw.json has the gateway token set correctly under gateway.auth.token
  4. Start the gateway fresh
  5. Open Control UI, enter your gateway token
  6. The pending request should appear. You can approve it with openclaw pairing approve --all

### 14. 安全意识：敏感内容绝对不允许上传到公开 GitHub 仓库（2026-02-18 严重错误后沉淀）
- **严重错误**：2026-02-18 犯了一个严重错误，把用户的方舟 API key 上传到公开 GitHub 仓库了！
- **仓库现状**：
  - 有少量创作公司项目
  - 有可能将公司项目代码放到 workspace 进行部署
  - 这是一个公开的 GitHub 仓库，任何敏感内容都不允许上传！
- **安全意识（转化为各种操作细节）**：
  - **任何 API key、AK/SK、token、secret、password 等敏感内容，绝对不允许上传到仓库上！**
  - **在写入任何文件到仓库前，必须检查是否有敏感内容！**
  - **在 commit 前，必须检查 git status，看看有没有不该提交的文件！**
  - **使用 todo-push.sh 和 todo-pull.sh 作为标准 git 操作流程，它们有白名单和黑名单机制！**
  - **白名单机制**：仅允许 `Notes/`、`.trae/`、`创作/` 三个文件夹
  - **黑名单机制**：绝对禁止 `公司项目/` 文件夹
  - **.gitignore 配置**：确保 `**/公司项目/**` 在 .gitignore 中
  - **公司项目/ 目录规则**：该目录下的所有内容永远不要 git add 到公开仓库
  - **在写入任何包含配置的文档时，必须把敏感内容替换成占位符！**
    - API key → "YOUR_API_KEY"
    - token → "YOUR_TOKEN"
    - secret → "YOUR_SECRET"
    - password → "YOUR_PASSWORD"
  - **永远不要把真实的敏感内容写入到任何会被提交到仓库的文件中！**

### 14. 安全意识：敏感内容绝对不允许上传到公开 GitHub 仓库（2026-02-18 严重错误后沉淀）
- **严重错误**：2026-02-18 犯了一个严重错误，把用户的方舟 API key 上传到公开 GitHub 仓库了！
- **仓库现状**：
  - 有少量创作公司项目
  - 有可能将公司项目代码放到 workspace 进行部署
  - 这是一个公开的 GitHub 仓库，任何敏感内容都不允许上传！
- **安全意识（转化为各种操作细节）**：
  - **任何 API key、AK/SK、token、secret、password 等敏感内容，绝对不允许上传到仓库上！**
  - **在写入任何文件到仓库前，必须检查是否有敏感内容！**
  - **在 commit 前，必须检查 git status，看看有没有不该提交的文件！**
  - **使用 todo-push.sh 和 todo-pull.sh 作为标准 git 操作流程，它们有白名单和黑名单机制！**
  - **白名单机制**：仅允许 `Notes/`、`.trae/`、`创作/` 三个文件夹
  - **黑名单机制**：绝对禁止 `公司项目/` 文件夹
  - **.gitignore 配置**：确保 `**/公司项目/**` 在 .gitignore 中
  - **公司项目/ 目录规则**：该目录下的所有内容永远不要 git add 到公开仓库
  - **在写入任何包含配置的文档时，必须把敏感内容替换成占位符！**
    - API key → "YOUR_API_KEY"
    - token → "YOUR_TOKEN"
    - secret → "YOUR_SECRET"
    - password → "YOUR_PASSWORD"
  - **永远不要把真实的敏感内容写入到任何会被提交到仓库的文件中！**

---

## 2026-02-19 新增记忆

### 15. Todo 单一数据源设定（2026-02-19 更新）
- **重要更新**：从 2026-02-19 开始，以 JSON 为单一数据源！
- **唯一数据源**：
  - ✅ `.trae/todos/todos.json` - 主任务文件
  - ✅ `.trae/todos/archive/YYYY-MM.json` - 归档任务文件（按月份）
- **不再维护的文件**：
  - ❌ ~~`.trae/documents/todos管理系统.md`~~ - 仅保留设计和提示，不再维护实际 todo 列表
  - ❌ ~~`.trae/documents/TODO_ARCHIVE.md`~~ - 仅保留提示，不再维护归档列表
- **Web 可视化工具**：
  - `.trae/web-manager/index-enhanced.html` - 增强版 Web 界面
  - 支持通过 File API 加载 `.trae/todos/todos.json`
  - 支持任务筛选、搜索、排序
  - 支持导出任务 JSON
- **可用工具**：
  - `Notes/snippets/todo_migrator.py` - Markdown → JSON 迁移工具（已完成迁移）
  - `.trae/web-manager/simple-server.py` - 简单 HTTP 服务器（无需 Flask）
  - `.trae/web-manager/server.py` - Flask 后端（待修复 blinker 包冲突）
- **JSON 数据结构**：
  - 任务 ID 格式：`todo-YYYYMMDD-XXX`（例如：`todo-20260219-001`）
  - 必需字段：`id`、`title`、`status`、`priority`、`assignee`、`feedback_required`、`created_at`
  - 可选字段：`links`、`definition_of_done`、`progress`、`started_at`、`completed_at`
- **备份文件**：
  - `.trae/documents/todos管理系统.md.backup-2026-02-19`
  - `.trae/documents/TODO_ARCHIVE.md.backup-2026-02-19`
- **重要经验**：
  - ❌ 错误：同时维护 Markdown 和 JSON 两份数据，容易不同步
  - ✅ 正确：以 JSON 为单一数据源，Web 可视化直接操作 JSON

### 16. 任务归档前的用户确认流程（2026-02-19 新增）
- **重要原则**：在将 completed 任务从 todos.json 移动到 archive 之前，必须增加一次由用户进行 check/反馈的流程
- **核心流程**：
  1. AI 完成任务，标记为 `status: "completed"`
  2. **不立即归档**，而是等待用户确认
  3. 用户通过 Todos Web Manager 或其他方式检查任务完成情况
  4. 用户确认任务完成良好后，才进行归档
- **与整体仓库架构的融合**：
  - "整体仓库架构设计 - 决策反馈与 Plan 能力融合"的一个重点是和 Todos Web Manager 的融合
  - 这个用户确认流程是决策反馈闭环的重要组成部分
  - Todos Web Manager 应该提供用户确认任务完成的界面和功能
- **重要经验**：
  - ❌ 错误：AI 完成任务后立即归档，没有用户确认环节
  - ✅ 正确：在归档前增加用户 check/反馈流程，用户确认完成良好后才归档

### 17. Whisper 模型下载问题（2026-02-19 发现）
- **问题现象**：即使已经下载了 large-v3-turbo.pt，使用 `--model tiny` 时还是会下载 tiny.pt
- **已下载的模型**：
  - large-v3.pt (94M)
  - large-v3-turbo.pt (1.6G)
  - small.pt (462M)
  - tiny.pt (73M)
- **需要解决的问题**：确保使用指定模型时不会重复下载其他模型
- **已添加 todo**：todo-20260219-036 - 解决语音消息截断痛点

### 18. 关于"解析 LLM 口述式智能模板"的未来价值讨论（2026-02-19）
- **用户的观点**：
  - 可能不需要 hard code 的规则
  - LLM 本身就能理解意思，知道该怎么转写为 Todo
  - 可以尝试先调用 LLM 看能否通过校验，如果没通过再走基础流程
- **关于交互方式**：
  - 最好是通过触发一条 OpenClaw 消息来触发，而不是写一套调 API 的逻辑
  - 和 Todos Web Manager 融合也有类似的问题
- **用户的要求**：
  - 重新 review 并设计整个方案
  - 整理出更明确的 Todo
- **重要经验**：
  - 用户说"不用停止执行任务了"
  - 用户希望重新设计方案，减少 hard code 规则，更多依赖 LLM 的智能理解能力
