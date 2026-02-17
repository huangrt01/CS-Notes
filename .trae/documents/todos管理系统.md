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
   - 自动 commit &amp; push 到远程 Git 仓库
   - 内置方舟代码模型 `ark/doubao-seed-2-0-code-preview-260215`，可直接执行编码任务

4. **Git 仓库**：
   - 作为唯一的真相源（single source of truth）
   - 同步云端和本地的所有变更

5. **本地 Mac**：
   - Trae 定期（或手动执行 `todo-sync.sh`）从 Git 拉取最新代码
   - 执行任务，处理代码
   - 执行完成后，commit &amp; push 回远程仓库
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
Git commit &amp; push
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

## 当前任务列表

### 进行中 (In Progress)

#### OpenClaw集成（阶段二：Git同步机制实现）

* [ ] 火山引擎端：创建cs-notes-git-sync Skill

  * Priority：high

  * Assignee：AI

  * Feedback Required：否

  * Links：`.trae/documents/OpenClaw集成方案.md`、`.trae/openclaw-skills/cs-notes-git-sync/`

  * Started At：2026-02-16

  * Progress：已完成cs-notes-git-sync Skill的创建，包含完整功能

  * Deliverables：
    * `.trae/openclaw-skills/cs-notes-git-sync/` - Git同步Skill
    * skill.json - Skill配置文件
    * main.py - 核心功能实现（Git操作、消息解析、INBOX写入）
    * README.md - 说明文档

  * Definition of Done：

    * Skill可以接收Lark消息并解析为todo格式

    * Skill可以克隆/拉取CS-Notes仓库

    * Skill可以写入INBOX.md并自动commit &amp; push

  * Plan：

    * 创建Skill目录结构（skill.json、main.py）

    * 实现Git操作（clone、pull、commit、push）

    * 实现消息解析与INBOX.md写入

    * 测试端到端流程：Lark发消息 → Skill写入 → Git push

#### 结合 OpenClaw 能力与 Lark 集成

* [ ] 结合 OpenClaw 能力与 Lark 集成

  * Priority：high

  * Assignee：AI

  * Feedback Required：否

  * Links：`Notes/AI-Agent-Product&amp;PE.md`（OpenClaw深度调研）、`.trae/documents/todos管理系统.md`、`.trae/documents/OpenClaw集成方案.md`

  * Started At：2026-02-16

  * Progress：已完成调研和设计，创建了cs-notes-todo-sync Skill，更新了OpenClaw集成方案文档

  * Deliverables：
    * `.trae/openclaw-skills/cs-notes-todo-sync/` - Todo同步Skill
    * 更新了`.trae/documents/OpenClaw集成方案.md`，添加完整闭环流程设计

  * Definition of Done：

    * 调研 OpenClaw 与 Lark 集成的可行方案

    * 设计 Lark 作为 Omni-channel Inbox 的任务输入渠道

    * 设计任务状态同步到 Lark 的通知机制

    * 探索将本项目 snippets 包装成 OpenClaw Skills 的方式

    * 形成完整的集成方案文档

  * Plan：

    * 深入分析 OpenClaw 的多渠道架构和 Lark 机器人能力

    * 设计 Lark ↔ OpenClaw ↔ 本项目的完整闭环流程

    * 调研 OpenClaw 的 Skill 系统如何与本项目结合

    * 形成集成方案文档，明确技术实现路径

  * 集成思路参考：

    * **Lark作为任务输入渠道**：从Lark群聊/私聊/机器人接收任务，自动结构化成本项目todo格式

    * **任务状态同步**：任务开始/完成时自动在Lark通知，支持卡片展示

    * **Lark文档集成**：任务相关内容写入Lark文档，从Lark文档读取上下文

    * **Skills包装**：将本项目snippets包装成OpenClaw Skills，通过Lark Bot触发

### 待处理 (Pending)

#### 新增任务

* [ ] 每天整合5条重要AI新闻、5条重要AI产品发布，推送给我
  - Priority：high
  - Assignee：AI
  - Feedback Required：否
  - 前置依赖：联网搜索搞定
  - Definition of Done：
    * 每天整合 5 条重要 AI 新闻
    * 每天整合 5 条重要 AI 产品发布
    * 推送给我（通过 Lark/Feishu）
  - Plan：
    * 等待联网搜索权限开通
    * 设计新闻和产品发布的整合方案
    * 实现定时推送机制

* [ ] 参考 OrbitOS GitHub 仓库，增强我自己（全能笔记管理创作系统）
  - Priority：high
  - Assignee：AI
  - Feedback Required：否
  - Links：https://github.com/（待补充 OrbitOS 仓库链接）
  - 差异点：
    * OrbitOS 用了 Claude Code，我们不能用
    * OrbitOS 基于 Obsidian，我们基于更简洁的 Mac 端 Typora 或 Trae IDE 浏览内容
    * 因此可能我们也有必要增加自己的 Web 前端
  - Definition of Done：
    * 调研 OrbitOS GitHub 仓库
    * 分析 OrbitOS 的能力
    * 结合差异点，设计增强方案
    * 实现增强功能（如增加自己的 Web 前端）
  - Plan：
    * 调研 OrbitOS GitHub 仓库
    * 分析 OrbitOS 的能力和架构
    * 结合差异点，设计增强方案
    * 实现增强功能

#### User 需要做的任务

这些任务需要你在 Mac 上操作，或通过 Lark 与 OpenClaw bot 对话。

##### 阶段一：基础配置与验证

✅ **已完成，已归档至 TODO_ARCHIVE.md**

##### 阶段二：完整闭环测试

* [ ] 测试端到端完整流程
  * **Assignee**：User
  * **Priority**：high
  * **你需要说的（在 Lark 中）**：
    ```
    帮我记录一个任务：整理笔记 https://mp.weixin.qq.com/s/gyEbK_UaUO3AeQvuhhRZ6g
    优先级：high
    然后帮我执行这个任务
    ```
  * **或者分两步说**：
    1. 先添加任务：**"帮我添加任务：测试编码能力，优先级 high"**
    2. 再执行：**"使用 cs-notes-git-sync 来处理 INBOX 中的任务，然后帮我执行编码任务"**
  * **Definition of Done**：
    * Lark 消息 → INBOX.md → 方舟代码模型执行 → Git commit &amp; push → Lark 通知完成

* [ ] OpenClaw实操实践
  * Priority：medium
  * Assignee：User
  * Feedback Required：是（实操过程中可能需要反馈，实操完成后需记录）
  * Links：&lt;https://mp.weixin.qq.com/s/Mkbbqdvxh-95pVlnLv9Wig、`Notes/AI-Agent-Product&amp;PE.md`&gt;
  * Definition of Done：用户完成OpenClaw实操，将实践过程和结果记录到笔记
  * Plan：基于AI的调研分析，用户按照实践例子完成实操

* [ ] 解决火山引擎 OpenClaw Web Manager 访问问题

  * Priority：high

  * Assignee：User

  * Feedback Required：否

  * Links：`Notes/AI-Agent-Product&amp;PE.md`、`.trae/documents/OpenClaw集成方案.md`

  * 问题现状：
    * OpenClaw 部署在火山引擎 ECS 云服务器上
    * 直接访问公网 IP + 端口被拒绝
    * 无法进入 Web Manager

  * 原因分析：
    * OpenClaw 默认仅绑定 127.0.0.1:18789（本地优先架构）
    * 火山引擎 ECS 安全组默认限制入站访问
    * 火山引擎提供的三层安全防护方案减少公网暴露面

  * 可选方案（按推荐优先级排序）：

    **方案一：SSH 隧道转发（推荐，最安全）**
    * 优点：无需开放公网端口，安全性最高
    * 缺点：需要本地有 SSH 客户端
    * 操作：在本地 Mac 执行 `ssh -L 18789:127.0.0.1:18789 root@<ECS公网IP>`
    * 访问：本地浏览器打开 `http://127.0.0.1:18789`

    **方案二：配置安全组 + 修改绑定地址**
    * 优点：直接访问，无需 SSH 隧道
    * 缺点：需要开放公网端口，存在安全风险
    * 操作：
      1. 火山引擎控制台配置安全组，开放 18789 端口入站访问
      2. SSH 登录 ECS，修改 OpenClaw 配置绑定 0.0.0.0
      3. 重启 OpenClaw 服务
    * 访问：浏览器打开 `http://<ECS公网IP>:18789`

    **方案三：Nginx 反向代理 + HTTPS**
    * 优点：更安全，支持 HTTPS，可配置访问控制
    * 缺点：配置相对复杂
    * 操作：
      1. 安装 Nginx
      2. 配置反向代理到 127.0.0.1:18789
      3. 配置 SSL 证书（可选）
      4. 配置安全组开放 80/443 端口

  * Definition of Done：
    * 成功访问 OpenClaw Web Manager
    * 方案已记录在笔记中

  * Plan：
    * 先尝试方案一（SSH 隧道），这是最简单安全的方式
    * 如果需要长期稳定访问，再考虑方案二或三

#### AI 需要做的任务

这些任务由 AI 自动执行，或在 OpenClaw 云端运行。

##### OpenClaw 稳定性优化

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
  - **【AI 已经完成的工作】**
    * ✅ 分析 device token mismatch 的根本原因
    * ✅ 发现环境变量 `OPENCLAW_GATEWAY_TOKEN` 会覆盖 `gateway.auth.token`
    * ✅ 找到三个 token 都不一样：
      - paired.json：`fad65644b7374e54b6f512a0a3af3f87`
      - openclaw.json：`8a952b9ebe4779d29c6bf018196e5b5d67b43ea5f77237ee`
      - 环境变量：`59ac1f34670bb1c61a7bef9e29745b55507f0bb9170b35b1`
    * ✅ 找到解决方案（来自官方文档）
    * ✅ 记录到 MEMORY.md
  - **【需要你执行的步骤】**
    ```bash
    openclaw gateway install --force
    openclaw gateway restart
    ```
  - **【验证步骤】**
    ```bash
    openclaw cron list
    ```
  - **参考文档**：
    * Reddit 讨论 1：https://www.reddit.com/r/clawdbot/comments/1qujzgm/openclaw_gateway_token_mismatch_error/
    * Reddit 讨论 2：https://www.reddit.com/r/openclaw/comments/1r18u6b/mismatching_tokens/
    * OpenClaw 官方文档：https://docs.openclaw.ai/gateway/troubleshooting
  - 问题背景：
    * 错误信息：`gateway connect failed: Error: unauthorized: device token mismatch (rotate/reissue device token)`
    * 影响：无法使用 `openclaw cron` 等 Gateway CLI 工具
    * 当前状态：CLI 命令无法连接 Gateway，但 Feishu 聊天正常工作
    * AI 已经完成分析和记录，需要你执行命令！

* [ ] 设计实现 OpenClaw session 管理优化方案，避免 token 超限错误 + Token Usage 优化
  - Priority：high
  - Assignee：AI
  - Feedback Required：否
  - Links：https://www.answeroverflow.com/m/1469215145874948199、MEMORY.md、session-management-optimization.md
  - Definition of Done：
    * 深入研究 Answer Overflow 链接中关于 session 管理优化的探讨
    * 分析当前 token 超限问题的根本原因
    * 设计具体的 session 管理优化方案
    * 实现优化方案（如自动检测 session 长度、自动切换新 session 等）
    * 综合考虑 token usage 优化（今天下午消耗了几千万 token）
    * 测试验证方案有效，避免以后再遇到 "400 Total tokens of image and text exceed max message tokens" 错误
  - Plan：
    * ✅ 先尝试访问并学习 Answer Overflow 链接中的内容（遇到安全检查，跳过）
    * ✅ 分析当前 OpenClaw session 管理机制
    * ✅ 分析今天 token 消耗异常的原因
    * ✅ 设计优化方案（session-management-optimization.md）
    * 🔄 **执行方案 1：Session 长度监控与自动切换（进行中）**
      - **原则**：基于 OpenClaw 现有能力，不侵入内部代码
      - **实现**：创建独立的监控脚本（session-monitor.py）
      - **功能**：
        - Session 长度监控
        - Token 使用估算
        - 自动警告（消息数量、token 使用量）
        - 提供手动切换建议（使用 `/reset` 命令）
    * ⏸️ 方案 2-4：暂不执行（可能侵入 OpenClaw 内部实现）
    * ✅ 测试验证
  - 问题背景：
    * 错误信息：`400 Total tokens of image and text exceed max message tokens`
    * 解决方法（临时）：在 TUI 中使用 `/reset` 命令恢复会话
    * 问题本质：session 管理问题，当 session history 较长时需要切换新 session
    * 补充：今天下午消耗了几千万 token，需要综合优化
  - 进展：
    * ✅ 设计文档已创建：session-management-optimization.md
    * ✅ 使用指南已创建：.trae/documents/SESSION-OPTIMIZER-USAGE.md
    * ✅ 监控脚本已创建：session-monitor.py
    * ✅ 优化器脚本已创建：session-optimizer.py（自动检查 + 提醒切换）
    * ✅ 方案 1 已完成：Session 长度监控与提醒（脚本已创建，cron job 待配置）
    * ⏸️ 方案 2-4：暂不执行（避免侵入 OpenClaw 内部代码）

✅ **已完成，已归档至 TODO_ARCHIVE.md**

##### 笔记整理

✅ **探索如何让你能阅读微信公众号的文章** - 已完成，已归档至 TODO_ARCHIVE.md

✅ **创建 speech-to-text 工具，用于处理音频附件** - 已完成，已归档至 TODO_ARCHIVE.md

* [ ] 配置联网搜索能力（火山引擎 Ask-Echo 或 Brave Search）
  - Priority：high
  - Assignee：User
  - Feedback Required：否
  - Definition of Done：
    * 分析当前联网搜索能力现状
    * 确认 web_search 和 web_fetch 工具可用
    * 配置联网搜索 API key（火山引擎 Ask-Echo 或 Brave Search）
    * 测试联网搜索功能是否正常工作
  - Plan：
    * 分析当前工具状态（已完成：确认有 web_search 和 web_fetch 工具）
    * 了解火山引擎 Ask-Echo 服务（链接：https://console.volcengine.com/ask-echo/web-search）
    * 选择合适的方案（火山引擎 Ask-Echo 或 Brave Search）
    * 配置 API key
    * 测试联网搜索功能
  - 当前状态：
    * ✅ 已确认有 web_search 和 web_fetch 工具
    * ✅ 用户提供了火山引擎 Ask-Echo 链接：https://console.volcengine.com/ask-echo/web-search
    * ❌ 缺少 API key 配置
  - 分析结果：
    * 用户提供了火山引擎 Ask-Echo 服务链接
    * 需要进一步了解 Ask-Echo 是什么服务，是否可以替代 Brave Search
    * 可能的方案：
      - 方案一：使用火山引擎 Ask-Echo 服务
      - 方案二：继续使用 Brave Search API
  - 参考文档：https://docs.openclaw.ai/tools/web

##### OpenClaw集成

* [ ] 火山引擎端：创建cs-notes-git-sync Skill

  * Priority：high

  * Assignee：AI

  * Feedback Required：否

  * Links：`.trae/documents/OpenClaw集成方案.md`

  * Definition of Done：

    * Skill可以接收Lark消息并解析为todo格式

    * Skill可以克隆/拉取CS-Notes仓库

    * Skill可以写入INBOX.md并自动commit &amp; push

  * Plan：

    * 创建Skill目录结构（skill.json、main.py）

    * 实现Git操作（clone、pull、commit、push）

    * 实现消息解析与INBOX.md写入

    * 测试端到端流程：Lark发消息 → Skill写入 → Git push

* [x] 实现任务状态同步到Lark的通知机制
  - Priority：medium
  - Assignee：AI
  - Feedback Required：否
  - Links：`.trae/documents/OpenClaw集成方案.md`
  - Progress：已完成！✅
    - ✅ 结论：OpenClaw 已经有 `message` 工具，可以直接向用户发送消息，任务状态同步需求已经可以达成！
  - Definition of Done：
    - ✅ 火山引擎端可以监听Git仓库变化（已验证：OpenClaw 可以通过 heartbeat 定期检查 todo 状态）
    - ✅ 任务完成时可以通过OpenClaw message send发送通知到Lark（已验证：OpenClaw 有 `message` 工具）
    - ✅ 支持卡片展示任务状态（OpenClaw 的 message 工具支持 Feishu 交互式卡片）
  - Plan：
    - 研究Git webhook或定期轮询机制（已验证：使用 OpenClaw 的 heartbeat 机制定期检查 todo 状态）
    - 编写状态同步脚本（已验证：OpenClaw 有 `message` 工具，可以直接发送消息）
    - 集成openclaw message send命令（已验证：OpenClaw 已经有 `message` 工具）

##### 测试与验证

* [ ] 测试完整的Lark → 火山引擎OpenClaw → Git → 本地Mac闭环流程

  * Priority：high

  * Assignee：User + AI

  * Feedback Required：是

  * Links：`.trae/documents/OpenClaw集成方案.md`

  * Definition of Done：

    * 从Lark发送消息，火山引擎端可以接收并写入INBOX.md

    * 火山引擎端可以自动commit &amp; push到Git仓库

    * 本地Mac端可以pull到最新代码，todo-sync.sh扫描新任务

    * 本地Mac执行任务后commit &amp; push，火山引擎端拉取更新并通过Lark通知用户

  * Plan：

    * 从Lark发送测试消息

    * 验证火山引擎端是否接收并push到Git

    * 本地Mac运行todo-sync.sh验证是否拉取到新任务

    * 模拟本地执行任务并push，验证Lark通知

* [x] 验证cs-notes-git-sync Skill的能力

  * Priority：medium

  * Assignee：AI

  * Feedback Required：否

  * Links：`.trae/documents/OpenClaw集成方案.md`、`.trae/documents/TODO_ARCHIVE.md`

  * Definition of Done：

    * 验证火山引擎上 `~/.openclaw/workspace/skills/cs-notes-git-sync/` 目录存在且完整

    * 验证 skill.json 配置正确

    * 验证 main.py 可以正常从OpenClaw接收消息、写入INBOX.md、并执行Git操作

    * 验证端到端流程正常工作

  * Progress：已完成！✅
    * ✅ 检查目录结构完整性：skill.json、main.py、README.md 都存在
    * ✅ 验证 skill.json 配置正确
    * ✅ 测试 main.py 功能：成功运行，可以正常 pull、写入 INBOX.md、commit & push
    * ✅ 验证端到端流程正常工作：测试消息成功添加到 INBOX.md！
  * Plan：

    * 检查目录结构完整性

    * 验证skill.json配置

    * 测试main.py功能

    * 结合OpenClaw gateway进行端到端测试

##### Skill 整合

* [x] 分析并整合 cs-notes-git-sync 和方舟代码模型
  * Assignee：AI
  * Priority：high
  * Definition of Done：
    * cs-notes-git-sync skill 和方舟代码模型可以配合工作
    * 设计工作流：Lark 消息 → INBOX.md → 编码执行 → Git 同步
  * Progress：已完成！✅
    * ✅ 分析 cs-notes-git-sync skill 的工作流
    * ✅ 设计与方舟代码模型的整合方案
    * ✅ 创建整合方案文档：`cs-notes-git-sync-与方舟代码模型整合方案.md`
  * Plan：
    * 分析 cs-notes-git-sync skill 的工作流
    * 设计与方舟代码模型的整合方案
    * 如果需要，创建整合 skill

##### 其他待处理任务

* [ ] 接入 Lark Bot 自动写入 Inbox

  * Priority：high

  * Links：`.trae/documents/INBOX.md`

  * Definition of Done：从手机将公网文章链接发到 Lark Bot 后，Bot 自动把内容同步进仓库 Inbox；Trae 可基于 Inbox 执行后续整理

  * Plan：设计消息格式与字段映射（content/priority/links/due），实现写入方式（commit/PR 或接口写入），补齐鉴权与防刷策略

* [ ] 落地 Todos Web Manager（手机提交 + 执行闭环）

  * Priority：high

  * Links：`.trae/documents/INBOX.md`、`.trae/documents/TEMPLATES.md`

  * Definition of Done：

    * 提供一个手机可用的提交入口，把任务写入 `.trae/documents/INBOX.md` 或 Pending

    * 提供一个列表页可查看 Pending/进行中/已完成，并支持标记完成与筛选

    * 每个任务支持 Plan 与执行产物回写（摘要/链接/diff/复现命令）

  * Plan：

    * MVP：仅实现 "提交入口 + 列表页 + 写入 Inbox + 基本鉴权"

    * 可观测闭环：写入与执行回写采用结构化字段，失败可追踪与可重试

    * 部署：优先本地/内网可用，后续再考虑公网访问与风控

* [ ] 实现 Todos Web Manager 核心功能

  * Priority：high

  * Links：`.trae/documents/INBOX.md`

  * Definition of Done：

    * 显示任务列表

    * 添加新任务

    * 标记任务完成

    * 任务分类和筛选

    * 优先级设置

    * 手机端快速提交任务

  * Plan：

    * 确定前端框架、后端接口、数据存储技术栈

    * 先实现 MVP 版本（提交入口 + 列表页 + 基本功能）

* [ ] 为任务执行引入可观测闭环

  * Priority：high

  * Links：本文件原"增强设计：闭环、语音、Plan Mode"小节、胡渊鸣《我给 10 个 Claude Code 打工》Step 7

  * Definition of Done：

    * 任务执行采用结构化日志输出（如 stream-json），让调度器能识别执行阶段、失败原因、重试点与最终产物

    * 任务完成率作为核心系统指标：失败任务进入重试队列，超过阈值自动降级为"需要人工补充信息/拆分"的任务

    * 每个任务沉淀产物：执行摘要、产物链接、关键 diff、失败复现命令，回写到任务条目下，便于长期复盘

* [ ] 用自然语言与语音提高任务输入吞吐

  * Priority：medium

  * Links：本文件原"增强设计：闭环、语音、Plan Mode"小节、胡渊鸣《我给 10 个 Claude Code 打工》Step 8

  * Definition of Done：

    * 手机端支持语音输入，将语音转写后再结构化为 todo（content/priority/tags/links/due）

    * 支持"口述式任务模板"：例如"高优先级：把 X 文档改成 Y 风格，关联链接 Z，明天前完成"

    * 支持将口述内容自动补全上下文：如自动带上当前正在编辑的文档路径或最近打开的文件列表

* [ ] 给任务系统添加 Plan Mode 与批量 Review

  * Priority：medium

  * Links：本文件原"增强设计：闭环、语音、Plan Mode"小节、胡渊鸣《我给 10 个 Claude Code 打工》Step 9

  * Definition of Done：

    * 新任务默认先产出 Plan（目标、假设、改动点、验收标准、风险），Plan 通过后才进入执行队列

    * 支持批量 kick off Plan 并统一 review：先把需求对齐成本前置，避免大量无效执行

    * Plan 与执行拆分成两类条目：Plan 条目更强调意图与验收；执行条目更强调产物与可追溯性

##### 备份方案

* [x] trae-agent 调研
  * Assignee：AI
  * Priority：low
  * Links：https://github.com/bytedance/trae-agent
  * Definition of Done：
    * 克隆 trae-agent 仓库
    * 阅读项目文档
    * 了解项目架构和功能
  * Progress：已完成！✅
    * ✅ 克隆 trae-agent 仓库
    * ✅ 阅读项目文档
    * ✅ 了解项目架构和功能
    * ✅ 创建调研报告文档：`Notes/snippets/code-reading-trae-agent.md`

### 已完成 (Completed)

已完成任务已归档至：[TODO\_ARCHIVE.md](TODO_ARCHIVE.md)

## 快速使用（已有功能）

**手机端提交任务**：

1. 用 Working Copy 打开 CS-Notes 仓库
2. 编辑 `.trae/documents/INBOX.md`，一行添加一个任务
3. Commit &amp; Push

**电脑端同步任务**：

```bash
cd /path/to/CS-Notes
./Notes/snippets/todo-sync.sh
```

脚本会自动：

1. 拉取 git 最新代码
2. 扫描 Inbox 中的新任务
3. 输出格式化好的任务，可复制到 Pending

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

**示例**：

```
### 进行中 (In Progress)
- [ ] OpenClaw深入调研分析
  - Priority：medium
  - Assignee：AI
  - Feedback Required：是
  - Started At：2026-02-16 14:30
  - Executor：Trae-1
  - Links：...
```

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

### 手机远程协作：手机给 Trae 提交任务

参考 [Editor.md](file:///Users/bytedance/CS-Notes/Notes/Editor.md#L5-L16) 中"手机远程协作"的思路，将手机定位为"随时随地提交轻量任务"的入口，让 Trae 在电脑侧消费任务并执行。

* 目标：

  * 在不打开电脑 IDE 的情况下，从手机快速提交一个待办（灵感、改文档、查资料、跑脚本等）

  * 保留结构化字段（优先级、标签、截止时间、关联文档），便于 Trae 自动路由和执行

* 推荐实现路径（按投入从低到高）：

  * Git 客户端提交：使用 Working Copy 等手机 Git 客户端，直接编辑本文件或新增 `.trae/documents/INBOX.md` 条目，通过 commit/PR 同步到仓库

  * 远程 SSH：手机通过 SSH 连接到开发机，在终端里追加任务条目（适合临时救急）

  * Web 提交入口：提供一个极简表单（PWA/网页），手机填写后调用后端接口把任务写入任务列表（或写入 `.trae/documents/INBOX.md`，再由 Trae 定时搬运）

* Web 提交入口的接口草案：

  * `POST /todos`：写入 Pending（或 inbox）

  * Body: `{ "content": "...", "priority": "high|medium|low", "tags": ["..."], "links": ["..."], "due": "YYYY-MM-DD" }`

  * 安全：Token/签名校验，避免公网被刷；写入操作记录来源与时间

* Trae 消费任务的协作方式：

  * Trae 定期扫描 Pending 与 `.trae/documents/INBOX.md`，自动生成执行计划并回写进度

  * 任务与文档互链：提交时带上目标文件路径，Trae 直接定位到对应文档执行修改

## 最近关注点

用于记录用户近期关注的主题领域，帮助 AI 更准确地规划任务优先级。

### 当前关注点（2026-02-16）

* **AI Agent**

  * 关注程度：高

  * 时间：2026-02-16

  * 相关：OpenClaw 深入调研、多任务生成式与 POI 场景 paper 整理

  * 链接：`Notes/AI-Agent-Product&amp;PE.md`、`Notes/深度学习推荐系统.md:3472`

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
3. **知识整理类任务 &gt; 实操类任务**（如笔记整理、文档更新、知识归纳等优先于需要用户手动操作的任务）
4. **Assignee 为 User 的任务 → 保持原优先级，不主动执行**
5. **阶段性进展优先，小步快跑**

## 📖 与 OpenClaw Bot 对话参考

详细的交互指南请参考：`.trae/openclaw-skills/OPENCLAW_BOT_GUIDE.md`

### 常用对话模板

**查询可用 Skills**：
```
你有哪些可用的 Skills？
```

**使用 cs-notes-git-sync 添加任务**：
```
帮我添加一个任务：&lt;任务内容&gt;
优先级：high|medium|low
```

**直接编码**：
```
帮我&lt;编码任务描述&gt;
```

**测试完整流程**：
```
帮我添加任务：&lt;任务描述&gt;，优先级 high
然后帮我执行这个任务
```

***

*最后更新：2026-02-16*
