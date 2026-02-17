# Todos 管理系统

## 简介

笔记库的任务管理系统，用于跟踪和管理待办事项。

## 整体架构设计（火山引擎 + Git 同步方案）

### 核心思路

以 Git 为中心的同步架构，解决火山引擎（云端）与本地 Mac 之间的文件系统隔离问题。

### 三者交互流程

```
┌─────────┐     ┌─────────┐     ┌───────────────┐     ┌─────────┐     ┌───────────────┐
│  手机   │────▶│  Lark   │────▶│  火山引擎     │────▶│  Git    │────▶│  本地 Mac     │
│ (用户)  │     │ (飞书)  │     │  OpenClaw     │     │ 仓库    │     │  (Trae 执行)  │
└─────────┘     └─────────┘     └───────────────┘     └─────────┘     └───────────────┘
     ▲                                                                               │
     │                                                                               │
     └───────────────────────────────────────────────────────────────────────────────┘
                              任务完成通知 (通过 Lark)
```

### 各组件职责

1. **手机端**：
   - 用户通过 Lark（飞书）发送任务消息

2. **Lark 机器人**：
   - 接收手机消息，转发给火山引擎上的 OpenClaw

3. **火山引擎 OpenClaw**：
   - 运行在云端 ECS 上
   - 托管 CS-Notes 仓库的克隆
   - 通过自定义 Skill 接收 Lark 消息，写入 `.trae/documents/INBOX.md`
   - 自动 commit & push 到远程 Git 仓库
   - 内置方舟代码模型 `ark/doubao-seed-2-0-code-preview-260215`，可直接执行编码任务

4. **Git 仓库**：
   - 作为唯一的真相源（single source of truth）
   - 同步云端和本地的所有变更

5. **本地 Mac**：
   - Trae 定期（或手动执行 `todo-sync.sh`）从 Git 拉取最新代码
   - 执行任务，处理代码
   - 执行完成后，commit & push 回远程仓库
   - 火山引擎 OpenClaw 拉取更新，通过 Lark 通知用户任务完成

### 火山引擎部署 vs 本地 Mac 部署的核心区别

| 维度 | 本地部署 | 火山引擎部署 |
|------|---------|------------|
| 文件系统访问 | 直接访问本地文件系统 | 无法直接访问本地 Mac 文件系统，必须通过 Git 同步 |
| 网络架构 | OpenClaw Gateway 运行在 127.0.0.1:18789 | OpenClaw 运行在云端服务器 |
| 飞书集成 | 需要手动配置 | 预集成飞书，零代码配置 |
| 稳定性 | 受 Mac 休眠、关机影响 | 24/7 稳定运行 |

### OpenClaw + 方舟代码模型一体化方案

#### 核心思路

利用 OpenClaw 内置的方舟代码模型 `ark/doubao-seed-2-0-code-preview-260215` 直接调用编码能力，结合我们已有的 `cs-notes-git-sync` skill，形成完整闭环：

```
Lark 发送任务
    ↓
OpenClaw Gateway
    ↓
cs-notes-git-sync skill（写入 INBOX.md）
    ↓
方舟代码模型（执行编码任务）
    ↓
Git commit & push
    ↓
Lark 通知完成
```

#### OpenClaw 可用 Skills 清单

**本地工作区中的 Skills**：
1. image-generate - 使用内置 image_generate.py 脚本生成图片，需要准备清晰具体的 prompt
2. veadk-skills - VeADK 技能集合，包括根据需求生成 Agent、将 Enio Agent 转换为 VeADK-Go Agent
3. veadk-skills - VeADK 技能集合，包括根据需求生成 Agent、将 Langchain/Langgraph/Dify 工作流转换为 VeADK Agent
4. video-generate - 使用 video_generate.py 脚本生成视频，需要提供文件名和 prompt，可选提供首帧图片（URL 或本地路径）

**系统内置 Skills**：
1. feishu-doc - 飞书文档读写操作
2. feishu-drive - 飞书存储空间文件管理
3. feishu-perm - 飞书文档和文件权限管理
4. feishu-wiki - 飞书知识库导航
5. healthcheck - 主机安全加固和风险容忍度配置
6. skill-creator - 创建或更新 AgentSkills
7. tmux - 远程控制 tmux 会话
8. weather - 获取当前天气和预报

#### 方舟代码模型能力

**功能**：
- 直接使用 `ark/doubao-seed-2-0-code-preview-260215` 模型
- 支持自然语言指令执行编码任务
- 可以直接创建和修改代码文件
- 与 OpenClaw 其他 skill 无缝集成

**优势**：
- 开箱即用，无需单独部署 coding-agent skill
- 火山引擎方舟模型原生支持
- 结合 Lark 多渠道输入
- 已有 Git 同步 skill 可以复用

## 核心概念

* **Assignee**：明确标注任务由谁执行
  * `AI`：由 Trae 自动执行
  * `User`：需要用户亲自操作（如 OpenClaw 实操、云 API 配置等）

* **Feedback Required**：是否需要用户反馈/确认
  * `是`：任务执行前/中/后需要用户确认或补充信息
  * `否`：可以直接执行完成

* **当前任务状态**：Pending → In Progress → Completed

---

## 当前任务列表

### 进行中 (In Progress)

#### AI 需要做的任务

* [ ] 火山引擎端：创建cs-notes-git-sync Skill
  - Priority：high
  - Assignee：AI
  - Feedback Required：否
  - Links：`.trae/documents/OpenClaw集成方案.md`、`.trae/openclaw-skills/cs-notes-git-sync/`
  - Started At：2026-02-16
  - Progress：已完成cs-notes-git-sync Skill的创建，包含完整功能
  - Deliverables：
    * `.trae/openclaw-skills/cs-notes-git-sync/` - Git同步Skill
    * skill.json - Skill配置文件
    * main.py - 核心功能实现（Git操作、消息解析、INBOX写入）
    * README.md - 说明文档
  - Definition of Done：
    * Skill可以接收Lark消息并解析为todo格式
    * Skill可以克隆/拉取CS-Notes仓库
    * Skill可以写入INBOX.md并自动commit & push

#### User 需要做的任务

* [ ] 解决 OpenClaw Gateway device token mismatch 问题，恢复 cron 工具等 Gateway API 的使用
  - Priority：high
  - Assignee：User
  - Feedback Required：否
  - Links：`~/.openclaw/openclaw.json`、`~/.openclaw/devices/paired.json`、MEMORY.md
  - **【重要！需要你执行！】**
  - Definition of Done：
    * 执行 `openclaw gateway install --force`
    * 执行 `openclaw gateway restart`
    * 验证 `openclaw cron list` 等命令可以正常工作
  - **【AI 已经完成的工作】**：
    * ✅ 分析 device token mismatch 的根本原因
    * ✅ 发现环境变量 `OPENCLAW_GATEWAY_TOKEN` 会覆盖 `gateway.auth.token`
    * ✅ 找到三个 token 都不一样
    * ✅ 找到解决方案（来自官方文档）
    * ✅ 记录到 MEMORY.md
  - **【需要你执行的步骤】**：
    ```bash
    openclaw gateway install --force
    openclaw gateway restart
    ```

* [ ] 配置联网搜索能力（火山引擎 Ask-Echo 或 Brave Search）
  - Priority：high
  - Assignee：User
  - Feedback Required：否
  - Definition of Done：
    * 分析当前联网搜索能力现状
    * 确认 web_search 和 web_fetch 工具可用
    * 配置联网搜索 API key（火山引擎 Ask-Echo 或 Brave Search）
    * 测试联网搜索功能是否正常工作
  - 当前状态：
    * ✅ 已确认有 web_search 和 web_fetch 工具
    * ✅ 用户提供了火山引擎 Ask-Echo 链接：https://console.volcengine.com/ask-echo/web-search
    * ❌ 缺少 API key 配置

* [ ] 解决火山引擎 OpenClaw Web Manager 访问问题
  - Priority：high
  - Assignee：User
  - Feedback Required：否
  - Links：`Notes/AI-Agent-Product&PE.md`、`.trae/documents/OpenClaw集成方案.md`
  - 问题现状：
    * OpenClaw 部署在火山引擎 ECS 云服务器上
    * 直接访问公网 IP + 端口被拒绝
    * 无法进入 Web Manager
  - 可选方案（按推荐优先级排序）：
    * 方案一：SSH 隧道转发（推荐，最安全）
    * 方案二：配置安全组 + 修改绑定地址
    * 方案三：Nginx 反向代理 + HTTPS
  - Definition of Done：
    * 成功访问 OpenClaw Web Manager
    * 方案已记录在笔记中

* [ ] 测试端到端完整流程
  - Priority：high
  - Assignee：User
  - Feedback Required：否
  - Definition of Done：
    * Lark 消息 → INBOX.md → 方舟代码模型执行 → Git commit & push → Lark 通知完成

---

### 待处理 (Pending)

#### AI 需要做的任务（优先执行）

* [ ] 每天整合5条重要AI新闻、5条重要AI产品发布，推送给我
  - Priority：high
  - Assignee：AI
  - Feedback Required：否
  - 前置依赖：联网搜索搞定
  - Definition of Done：
    * 每天整合 5 条重要 AI 新闻
    * 每天整合 5 条重要 AI 产品发布
    * 推送给我（通过 Lark/Feishu）

* [ ] 参考 OrbitOS GitHub 仓库，增强我自己（全能笔记管理创作系统）
  - Priority：high
  - Assignee：AI
  - Feedback Required：否
  - 前置依赖：联网搜索权限
  - 差异点：
    * OrbitOS 用了 Claude Code，我们不能用
    * OrbitOS 基于 Obsidian，我们基于更简洁的 Mac 端 Typora 或 Trae IDE 浏览内容
    * 因此可能我们也有必要增加自己的 Web 前端
  - Definition of Done：
    * 调研 OrbitOS GitHub 仓库
    * 分析 OrbitOS 的能力
    * 结合差异点，设计增强方案
    * 实现增强功能（如增加自己的 Web 前端）

* [ ] 测试完整的Lark → 火山引擎OpenClaw → Git → 本地Mac闭环流程
  - Priority：high
  - Assignee：User + AI
  - Feedback Required：是
  - Links：`.trae/documents/OpenClaw集成方案.md`
  - Definition of Done：
    * 从Lark发送消息，火山引擎端可以接收并写入INBOX.md
    * 火山引擎端可以自动commit & push到Git仓库
    * 本地Mac端可以pull到最新代码，todo-sync.sh扫描新任务
    * 本地Mac执行任务后commit & push，火山引擎端拉取更新并通过Lark通知用户

#### 其他待处理任务

* [ ] 接入 Lark Bot 自动写入 Inbox
  - Priority：high
  - Links：`.trae/documents/INBOX.md`
  - Definition of Done：从手机将公网文章链接发到 Lark Bot 后，Bot 自动把内容同步进仓库 Inbox；Trae 可基于 Inbox 执行后续整理

* [ ] 落地 Todos Web Manager（手机提交 + 执行闭环）
  - Priority：high
  - Links：`.trae/documents/INBOX.md`、`.trae/documents/TEMPLATES.md`
  - Definition of Done：
    * 提供一个手机可用的提交入口，把任务写入 `.trae/documents/INBOX.md` 或 Pending
    * 提供一个列表页可查看 Pending/进行中/已完成，并支持标记完成与筛选
    * 每个任务支持 Plan 与执行产物回写（摘要/链接/diff/复现命令）

* [ ] 实现 Todos Web Manager 核心功能
  - Priority：high
  - Links：`.trae/documents/INBOX.md`
  - Definition of Done：
    * 显示任务列表
    * 添加新任务
    * 标记任务完成
    * 任务分类和筛选
    * 优先级设置
    * 手机端快速提交任务

* [ ] 为任务执行引入可观测闭环
  - Priority：high
  - Definition of Done：
    * 任务执行采用结构化日志输出（如 stream-json），让调度器能识别执行阶段、失败原因、重试点与最终产物
    * 任务完成率作为核心系统指标：失败任务进入重试队列，超过阈值自动降级为"需要人工补充信息/拆分"的任务
    * 每个任务沉淀产物：执行摘要、产物链接、关键 diff、失败复现命令，回写到任务条目下，便于长期复盘

* [ ] 用自然语言与语音提高任务输入吞吐
  - Priority：medium
  - Definition of Done：
    * 手机端支持语音输入，将语音转写后再结构化为 todo（content/priority/tags/links/due）
    * 支持"口述式任务模板"：例如"高优先级：把 X 文档改成 Y 风格，关联链接 Z，明天前完成"
    * 支持将口述内容自动补全上下文：如自动带上当前正在编辑的文档路径或最近打开的文件列表

* [ ] 给任务系统添加 Plan Mode 与批量 Review
  - Priority：medium
  - Definition of Done：
    * 新任务默认先产出 Plan（目标、假设、改动点、验收标准、风险），Plan 通过后才进入执行队列
    * 支持批量 kick off Plan 并统一 review：先把需求对齐成本前置，避免大量无效执行
    * Plan 与执行拆分成两类条目：Plan 条目更强调意图与验收；执行条目更强调产物与可追溯性

---

### 已完成 (Completed)

已完成任务已归档至：[TODO_ARCHIVE.md](TODO_ARCHIVE.md)

---

## 快速使用（已有功能）

**手机端提交任务**：

1. 用 Working Copy 打开 CS-Notes 仓库
2. 编辑 `.trae/documents/INBOX.md`，一行添加一个任务
3. Commit & Push

**电脑端同步任务**：

```bash
cd /path/to/CS-Notes
./Notes/snippets/todo-sync.sh
```

脚本会自动：

1. 拉取 git 最新代码
2. 扫描 Inbox 中的新任务
3. 输出格式化好的任务，可复制到 Pending

---

## 使用说明

### 任务执行流程（关键！避免冲突）

**重要原则**：多个执行器同时工作时，必须先标记任务为"进行中"，再开始执行。

**标准流程**：

1. **选取任务**：从 Pending 中选择一个任务
2. **原子移动**：将任务从 Pending 移动到 In Progress（这一步必须原子完成，避免多个执行器选同一个任务）
3. **开始执行**：执行任务内容
4. **完成标记**：执行完成后，将任务从 In Progress 移动到 Completed，并标记为 `- [x]`

**任务字段扩展**：

* `Started At`：任务开始执行时间（YYYY-MM-DD HH:MM）
* `Executor`：执行者标识（如 "Trae-1"、"User-Laptop"）
* `Timeout`：任务超时时间（可选，防止任务卡住）

### 添加新任务

1. 在相应的 section 中添加新任务条目
2. 使用 `- [ ]` 格式表示待办事项
3. 完成后用 `- [x]` 标记
4. 需要结构化字段与 Plan 时，参考 `.trae/documents/TEMPLATES.md`

### 执行原则

* **阶段性进展即可**：任务不追求一次完美，有阶段性进展就可以标记完成当前阶段，进入下一阶段
* **小步快跑**：把大任务拆成多个小阶段，每个阶段都有明确的产出
* **快速迭代**：优先实现可用的最小版本，后续再逐步完善

### AI 与用户协作最佳实践

* **主动催促用户执行Todo**：当发现用户是任务执行的重要block点时，必须主动提醒
  - **何时需要催促**：
    - Assignee 为 User 的任务长期处于 Pending 状态
    - 其他任务都依赖该 User 任务才能继续执行
    - 例如：OpenClaw 安装配置、飞书机器人创建等需要用户手动操作的任务
  - **如何催促**：
    - 明确告知用户："这个任务是后续工作的前置条件，需要您先完成"
    - 提供清晰的操作指引（参考 Plan 中的步骤）
    - 询问是否遇到困难，是否需要帮助
    - 不要过度打扰，但要确保用户意识到任务的重要性

### 任务优先级

* **high**: 高优先级，需要尽快完成
* **medium**: 中优先级
* **low**: 低优先级

---

## 最近关注点

用于记录用户近期关注的主题领域，帮助 AI 更准确地规划任务优先级。

### 当前关注点（2026-02-16）

* **AI Agent**
  * 关注程度：高
  * 时间：2026-02-16
  * 相关：OpenClaw 深入调研、多任务生成式与 POI 场景 paper 整理
  * 链接：`Notes/AI-Agent-Product&PE.md`、`Notes/深度学习推荐系统.md:3472`

* **自动化与效率工具**
  * 关注程度：高
  * 时间：2026-02-16
  * 相关：todo-sync.sh 脚本、手机提交 → 电脑自动执行闭环
  * 链接：`Notes/snippets/todo-sync.sh`、`Notes/snippets/todo-manager.py`

* **笔记整理与知识管理**
  * 关注程度：中
  * 时间：2026-02-16
  * 相关：大量笔记文件更新（2025.12.16）
  * 链接：`Notes/` 目录

### 优先级规划规则

1. **与最近高关注主题高度相关的任务 → 提升优先级**
2. **已标记为 high 且与关注点相关 → 最高优先级**
3. **知识整理类任务 > 实操类任务**（如笔记整理、文档更新、知识归纳等优先于需要用户手动操作的任务）
4. **Assignee 为 User 的任务 → 保持原优先级，不主动执行**
5. **阶段性进展优先，小步快跑**

---

## 📖 与 OpenClaw Bot 对话参考

详细的交互指南请参考：`.trae/openclaw-skills/OPENCLAW_BOT_GUIDE.md`

### 常用对话模板

**查询可用 Skills**：
```
你有哪些可用的 Skills？
```

**使用 cs-notes-git-sync 添加任务**：
```
帮我添加一个任务：<任务内容>
优先级：high|medium|low
```

**直接编码**：
```
帮我<编码任务描述>
```

**测试完整流程**：
```
帮我添加任务：<任务描述>，优先级 high
然后帮我执行这个任务
```

---

*最后更新：2026-02-17*
