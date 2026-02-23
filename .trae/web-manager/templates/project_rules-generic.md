# 项目规则

* 请在此处定义你的项目角色和能力（例如：你是一位项目开发专家，且精通...）
* 对于问题的处理流程，请在此定义（例如：先仔细回答，再将知识整合入项目中最合适位置...）

## 项目目标

* 请在此定义本项目的整体目标。

## Todo 驱动工作流

* 当用户说"新建任务 / 加一个 todo / 记一条待办"时：
  - 默认把任务写入 `.trae/documents/todos管理系统.md` 的 Pending（或先写入 Inbox，再搬运到 Pending）
  - 任务条目尽量结构化：优先级（high/medium/low）、链接（目标文件/外部链接）、可选截止时间
* 当用户说"根据 todos 执行 / 执行 todos / 清空待办"时：
  - 先读取 `.trae/documents/todos管理系统.md`，确定当前 Pending/进行中任务
  - 选择最重要且可独立完成的一项开始执行，完成后立刻回写进度与产物链接，再继续下一项
  - 涉及复杂改动时，先给出 Plan（目标、假设、改动点、验收标准、风险），Plan 明确后再动手
  - 执行中产生的经验教训沉淀到 `.trae/documents/PROGRESS.md`

## AI 助手工作流

* **用户消息优先原则**：无论 AI 正在执行什么任务，收到用户的新消息时必须**立即回复**，不能让任务阻塞对话
* **长时间任务后台化**：对于耗时操作，必须使用后台模式（background: true）运行，避免阻塞用户交互
* 当 AI 在执行复杂/长期任务时，**必须**：
  - 主动、定期（每 10 分钟）检查任务进度
  - 向用户报告进度，不要等待用户询问
  - 进度报告要包含：百分比、预估剩余时间
* **Git 操作 SOP（第一原则：敏感内容绝对不能 push 到 git）**：
  - 使用项目提供的 git 脚本（如果有）或遵循团队约定
  - 每次 commit 前，先执行 `git status` 检查
  - **.gitignore 配置**：确保敏感目录和文件在 .gitignore 中
  - 每次进行 git commit 并 push 后，必须在回复中包含对应的 GitHub commit 链接

## 目录与落盘约定

* `.trae/`：Trae 运行所需的规则、上下文文档、技能配置，服务"自动化执行"
* 请在此定义你的项目目录结构

## 写作与质量标准

* "先回答再落盘"：先解决当前问题，再把可复用的结论沉淀到最合适的位置
* 统一可追溯：重要结论尽量附上来源链接或文件引用路径
* 验收优先：写作/改文档以"读者能一眼看懂"为验收标准，列表化、分段清晰、避免堆砌

## Todo 单一数据源设定

* **重要：以 JSON 为单一数据源！**
* **唯一数据源**：
  - ✅ `.trae/todos/todos.json` - 主任务文件
  - ✅ `.trae/todos/archive/YYYY-MM.json` - 归档任务文件（按月份）
* **Web 可视化工具**：
  - `.trae/web-manager/index-enhanced.html` - 增强版 Web 界面
  - 支持通过 File API 加载 `.trae/todos/todos.json`
  - 支持任务筛选、搜索、排序
  - 支持导出任务 JSON
* **JSON 数据结构**：
  - 任务 ID 格式：`todo-YYYYMMDD-XXX`（例如：`todo-20260219-001`）
  - 必需字段：`id`、`title`、`status`、`priority`、`assignee`、`feedback_required`、`created_at`
  - 可选字段：`links`、`definition_of_done`、`progress`、`started_at`、`completed_at`

## P0-P9 优先级体系

* **重要：使用 P0-P9 的 10 级优先级体系！**
* **优先级定义**：
  - **P0**：最高优先级 - 阻断性问题、必须立即处理
  - **P1**：非常高优先级 - 核心功能、用户体验关键
  - **P2**：高优先级 - 重要功能、用户体验优化
  - **P3**：中高优先级 - 有用的功能、体验提升
  - **P4**：中优先级 - 常规功能、一般优化
  - **P5**：中低优先级 - 次要功能、小优化
  - **P6**：低优先级 - 可有可无的功能
  - **P7**：很低优先级 - 锦上添花
  - **P8**：极低优先级 - 未来考虑
  - **P9**：最低优先级 - 几乎不做
* **向后兼容**：
  - high → 相当于 P2
  - medium → 相当于 P5
  - low → 相当于 P8
* **可用工具**：
  - `.trae/openclaw-skills/priority-task-reader/` - 按优先级读取任务的 skill
  - 使用方法：`python3 .trae/openclaw-skills/priority-task-reader/main.py`
  - 支持 `--json` 输出 JSON 格式
  - 支持 `--next` 显示下一个应该执行的任务

*这是通用化模板，请根据新项目需求进行定制*
