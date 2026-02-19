# Todos 管理系统

## 进行中 (In Progress)

* [ ] 火山引擎端：创建cs-notes-git-sync Skill
  - Priority：High
  - Assignee：Ai
  - Feedback Required：否
  - Links：`.trae/documents/OpenClaw集成方案.md`、`.trae/openclaw-skills/cs-notes-git-sync/`
  - Definition of Done：
    * Skill可以接收Lark消息并解析为todo格式
    * Skill可以克隆/拉取CS-Notes仓库
    * Skill可以写入INBOX.md并自动commit & push
    * #### User 需要做的任务
  - Progress：已完成cs-notes-git-sync Skill的创建，包含完整功能
  - Started At：2026-02-16

* [ ] 解决 OpenClaw Gateway device token mismatch 问题，恢复 cron 工具等 Gateway API 的使用
  - Priority：High
  - Assignee：User
  - Feedback Required：否
  - Links：`~/.openclaw/openclaw.json`、`~/.openclaw/devices/paired.json`、MEMORY.md
  - Definition of Done：
    * 执行 `openclaw gateway install --force`
    * 执行 `openclaw gateway restart`
    * 验证 `openclaw cron list` 等命令可以正常工作
  - Progress：
  - Started At：2026-02-17
  - Completed At：2026-02-17

* [ ] 配置联网搜索能力（火山引擎 Ask-Echo 或 Brave Search）
  - Priority：High
  - Assignee：User
  - Feedback Required：否
  - Definition of Done：
    * 分析当前联网搜索能力现状
    * 确认 web_search 和 web_fetch 工具可用
    * 配置联网搜索 API key（火山引擎 Ask-Echo 或 Brave Search）
    * 测试联网搜索功能是否正常工作

* [ ] 解决火山引擎 OpenClaw Web Manager 访问问题
  - Priority：High
  - Assignee：User
  - Feedback Required：否
  - Links：`Notes/AI-Agent-Product&PE.md`、`.trae/documents/OpenClaw集成方案.md`
  - Definition of Done：
    * 成功访问 OpenClaw Web Manager
    * 方案已记录在笔记中

* [ ] 测试端到端完整流程
  - Priority：High
  - Assignee：User
  - Feedback Required：否
  - Definition of Done：
    * Lark 消息 → INBOX.md → 方舟代码模型执行 → Git commit & push → Lark 通知完成
    * ---
    * #### AI 需要做的任务（优先执行）

## 待处理 (Pending)

### AI 需要做的任务（优先执行）

* [ ] 思考并设计自主推进任务机制的优化方案（不局限于 heartbeat）
  - Priority：High
  - Assignee：Ai
  - Feedback Required：否
  - Links：HEARTBEAT.md、MEMORY.md
  - Definition of Done：
    * 从两方面思考：1) 结合当前 heartbeat 机制本身怎么优化 2) 跳出这个局限性，思考怎么去优化
    * 基于 OpenClaw 现有能力，考虑配合一些新的能力
    * 输出完整的优化方案文档
  - Progress：

* [ ] 端到端验证 Plan Generator + Hybrid Executor 完整流程
  - Priority：High
  - Assignee：Ai
  - Feedback Required：否
  - Links：`.trae/openclaw-skills/plan-generator/`、`.trae/openclaw-skills/hybrid-executor/`、`.trae/documents/Plan-Mode-混合执行实现方案.md`
  - Definition of Done：
    * 模拟用户发出一些命令或 task
    * 执行这些 task，基于新的 plan 和 hybrid executor 的能力
    * 走通端到端完整流程
    * 输出完整的验证报告
  - Progress：

* [ ] 类似RSS订阅的方式关注非技术知识.md里长期关注的博主列表，如有更新则通知我（最终目标：全部博主都能关注）
  - Priority：High
  - Assignee：Ai
  - Feedback Required：否
  - Links：`Notes/非技术知识.md`
  - Definition of Done：
    * **第一个自动化**：自动监控 md 文件（非技术知识.md）这个具体 section 中的博主名
    * **第二个自动化**：自动 rss 订阅监控博主更新
    * 解析非技术知识.md里的长期关注博主列表（**最终目标：全部博主都能关注**）
    * 实现类似RSS订阅的方式监控博主更新
    * 如有更新则通过Lark/Feishu通知用户
  - Progress：
  - 前置依赖：
    * 联网搜索能力 ready
  - 用户要求：
    * 最终目标：全部博主都能关注！
    * 博主监控脚本还是希望能自动监控，硬编码仅作为兜底项
    * 博主监控这个需求应该没做完，之后要基于联网搜索rss订阅
    * **两个自动化**：1) 自动监控 md 文件这个具体 section 中的博主名 2) 自动 rss 订阅

* [ ] 测试 Plan Generator、Plan Executor、Task Executor
  - Priority：High
  - Assignee：Ai
  - Feedback Required：否
  - Links：`.trae/openclaw-skills/plan-generator/`、`.trae/openclaw-skills/plan-executor/`、`.trae/openclaw-skills/hybrid-executor/`、`Notes/snippets/task_executor.py`
  - Definition of Done：
    * 测试 Plan Generator
    * 测试 Plan Executor
    * 测试 Task Executor
    * 验证功能正常
  - Progress：

* [ ] 实现 LLM 智能解析口述式任务模板
  - Priority：High
  - Assignee：Ai
  - Feedback Required：否
  - Links：`Notes/snippets/voice_task_parser.py`
  - Definition of Done：
    * 实现 LLM 智能解析口述式任务模板
    * 支持更自然的口述方式
    * 测试 LLM 智能解析功能
  - Progress：

* [ ] 实现自动补全上下文
  - Priority：Medium
  - Assignee：Ai
  - Feedback Required：否
  - Links：`Notes/snippets/voice_task_parser.py`
  - Definition of Done：
    * 实现自动补全上下文
    * 从最近的 git commit 历史和文件操作记录提取上下文
    * 测试自动补全上下文功能
  - Progress：

* [ ] 实现 Todos Web Manager Phase 2
  - Priority：High
  - Assignee：Ai
  - Feedback Required：否
  - Links：`.trae/web-manager/index.html`、`.trae/documents/Todos-Web-Manager-核心功能实现方案.md`
  - Definition of Done：
    * 实现任务分类和筛选
    * 实现搜索功能
    * 实现 Git 集成
    * 部署到火山引擎
  - Progress：

* [ ] 创建 Hybrid Executor Skill
  - Priority：Medium
  - Assignee：Ai
  - Feedback Required：否
  - Links：`.trae/documents/Plan-Mode-混合执行实现方案.md`
  - Definition of Done：
    * 创建 Hybrid Executor Skill
    * 调度混合执行流程
    * 测试 Hybrid Executor
  - Progress：

* [ ] 调研 Top Lean AI 榜单上所有公司并分类记录整理到 AI Agent 笔记中
  - Priority：High
  - Assignee：Ai
  - Feedback Required：否
  - Links：`.trae/documents/Top-Lean-AI-榜单-2026-02.md`、`Notes/AI-Agent-Product&PE.md`
  - Definition of Done：
    * 调研 Top Lean AI 榜单上所有 45 家公司
    * 分类记录整理到 AI Agent 笔记中
    * 包含公司信息、类型、人均年收入、商业模式等
  - 前置依赖：
    * 联网搜索能力 ready

* [ ] claude code 也作为工具，等具备联网搜索能力，能调研接入方法后再做这个 todo
  - Priority：High
  - Assignee：Ai
  - Feedback Required：否
  - Links：https://github.com/anthropics/claude-code
  - Definition of Done：
    * 调研 Claude Code 的接入方法
    * 分析 Claude Code 与现有工具的整合方案
    * 实现 Claude Code 作为 OpenClaw 的工具
  - 前置依赖：
    * 联网搜索能力 ready

* [ ] 调研 openclaw 的各种配置参数，独立分析/让我选择更适合我们场景的
  - Priority：High
  - Assignee：Ai
  - Feedback Required：是
  - Links：`/root/.openclaw/openclaw.json`、OpenClaw 官方文档、`.trae/documents/OpenClaw-配置参数分析与推荐.md`
  - Definition of Done：
    * 调研 OpenClaw 的各种配置参数
    * 独立分析各种配置参数的优缺点
    * 推荐更适合我们场景的配置参数
    * 让用户选择更适合的配置
  - Progress：

* [ ] 评估 trae-agent 的能力，对比 OpenClaw + 方舟代码模型，评估它改笔记代码的效果
  - Priority：High
  - Assignee：Ai
  - Feedback Required：否
  - Links：trae-agent/、Notes/snippets/code-reading-trae-agent.md、.trae/documents/trae-agent-评估报告.md
  - Definition of Done：
    * 跑通 trae-agent client
    * 设计一些需求让 trae-agent 来改笔记代码
    * 评估它改的效果如何
    * 对比 OpenClaw + 方舟代码模型的效果
  - Progress：

* [ ] 设计并实现 trae-agent OpenClaw Skill，整合 trae-agent 到 CS-Notes 项目
  - Priority：High
  - Assignee：Ai
  - Feedback Required：否
  - Links：.trae/documents/trae-agent-评估报告.md、trae-agent/、.trae/openclaw-skills/trae-agent/
  - Definition of Done：
    * 创建 trae-agent OpenClaw Skill
    * Skill 可以调用 trae-agent 执行复杂任务
    * Skill 可以获取 trae-agent 的轨迹记录和执行结果
    * 综合 OpenClaw 和 trae-agent 的能力，一加一大于二
  - Progress：

* [ ] 每天整合5条重要AI新闻、5条重要AI产品发布，推送给我
  - Priority：High
  - Assignee：Ai
  - Feedback Required：否
  - Definition of Done：
    * 每天整合 5 条重要 AI 新闻
    * 每天整合 5 条重要 AI 产品发布
    * 推送给我（通过 Lark/Feishu）
  - 前置依赖：
    * 联网搜索搞定

* [ ] 参考 OrbitOS GitHub 仓库，增强我自己（全能笔记管理创作系统）
  - Priority：High
  - Assignee：Ai
  - Feedback Required：否
  - Definition of Done：
    * 调研 OrbitOS GitHub 仓库
    * 分析 OrbitOS 的能力
    * 结合差异点，设计增强方案
    * 实现增强功能（如增加自己的 Web 前端）
  - 前置依赖：
    * 联网搜索权限

* [ ] 落地 Todos Web Manager（手机提交 + 执行闭环）
  - Priority：High
  - Assignee：Ai
  - Feedback Required：否
  - Links：`.trae/documents/INBOX.md`、`.trae/documents/TEMPLATES.md`、`.trae/documents/Todos-Web-Manager-设计方案.md`、`.trae/web-manager/index.html`
  - Definition of Done：
    * 提供一个手机可用的提交入口，把任务写入 `.trae/documents/INBOX.md` 或 Pending
    * 提供一个列表页可查看 Pending/进行中/已完成，并支持标记完成与筛选
    * 每个任务支持 Plan 与执行产物回写（摘要/链接/diff/复现命令）
  - Progress：

* [ ] 实现 Todos Web Manager 核心功能
  - Priority：High
  - Assignee：Ai
  - Feedback Required：否
  - Links：`.trae/documents/INBOX.md`、`.trae/documents/Todos-Web-Manager-核心功能实现方案.md`、`.trae/web-manager/index.html`
  - Definition of Done：
    * 显示任务列表
    * 添加新任务
    * 标记任务完成
    * 任务分类和筛选
    * 优先级设置
    * 手机端快速提交任务
  - Progress：

* [ ] 为任务执行引入可观测闭环
  - Priority：High
  - Assignee：Ai
  - Feedback Required：否
  - Links：`.trae/documents/任务执行可观测闭环设计方案.md`、`.trae/documents/任务执行可观测闭环-实现方案.md`
  - Definition of Done：
    * 任务执行采用结构化日志输出（如 stream-json），让调度器能识别执行阶段、失败原因、重试点与最终产物
    * 任务完成率作为核心系统指标：失败任务进入重试队列，超过阈值自动降级为"需要人工补充信息/拆分"的任务
    * 每个任务沉淀产物：执行摘要、产物链接、关键 diff、失败复现命令，回写到任务条目下，便于长期复盘
  - Progress：

* [ ] 用自然语言与语音提高任务输入吞吐
  - Priority：Medium
  - Assignee：Ai
  - Feedback Required：否
  - Links：`.trae/documents/自然语言与语音任务输入方案.md`、`Notes/snippets/voice_task_parser.py`、`Notes/snippets/speech_to_text.py`
  - Definition of Done：
    * 手机端支持语音输入，将语音转写后再结构化为 todo（content/priority/tags/links/due）
    * 支持"口述式任务模板"：例如"高优先级：把 X 文档改成 Y 风格，关联链接 Z，明天前完成"
    * 支持将口述内容自动补全上下文：如自动带上当前正在编辑的文档路径或最近打开的文件列表
  - Progress：

* [ ] 给任务系统添加 Plan Mode 与批量 Review
  - Priority：Medium
  - Assignee：Ai
  - Feedback Required：否
  - Links：`.trae/documents/Plan-Mode-与批量-Review-设计方案.md`、`.trae/documents/Plan-Mode-混合执行实现方案.md`
  - Definition of Done：
    * 新任务默认先产出 Plan（目标、假设、改动点、验收标准、风险），Plan 通过后才进入执行队列
    * 支持批量 kick off Plan 并统一 review
    * Plan 与执行拆分成两类条目：Plan 条目更强调意图与验收；执行条目更强调产物与可追溯性
    * 支持混合执行方式：当遇到复杂任务时，先生成 Plan，然后先自动执行，再用 AI 自动生成的方式
  - Progress：

* [ ] 整体仓库架构设计，考虑决策反馈方式，与 Plan 能力融合
  - Priority：High
  - Assignee：Ai
  - Feedback Required：是
  - Links：`.trae/documents/整体仓库架构设计-决策反馈与Plan能力融合.md`
  - Definition of Done：
    * 分析现有整体仓库架构
    * 设计决策反馈方式
    * 将决策反馈方式与 Plan 能力融合起来
    * 形成完整的架构设计文档
  - Progress：

### User 需要做的任务

* [ ] 我在创作文件夹中对openclaw的理解，融入到公司项目openclaw和ai推荐的结合
  - Priority：High
  - Assignee：User
  - Feedback Required：否
  - Links：`创作/OpenClaw使用心得和价值.md`、公司项目文件夹
  - Definition of Done：
    * 读取创作文件夹中对openclaw的理解（`创作/OpenClaw使用心得和价值.md`）
    * 读取公司项目文件夹中的内容
    * 将创作文件夹中对openclaw的理解融入到公司项目openclaw和ai推荐的结合中

* [ ] 测试完整的Lark → 火山引擎OpenClaw → Git → 本地Mac闭环流程
  - Priority：High
  - Assignee：User + ai
  - Feedback Required：是
  - Links：`.trae/documents/OpenClaw集成方案.md`
  - Definition of Done：
    * 从Lark发送消息，火山引擎端可以接收并写入INBOX.md
    * 火山引擎端可以自动commit & push到Git仓库
    * 本地Mac端可以pull到最新代码，todo-sync.sh扫描新任务
    * 本地Mac执行任务后commit & push，火山引擎端拉取更新并通过Lark通知用户
    * #### 其他待处理任务

## 已完成 (Completed)
