# TODO 归档

用于存放已完成的待办事项，便于长期回顾和复盘。

---

## 2026-02-16

### 已完成

- [x] 整理多任务生成式与POI场景的paper笔记
  - Priority：medium
  - Links：http://xhslink.com/o/8rKN7celDGn、https://arxiv.org/html/2602.11664v1
  - Definition of Done：将paper核心内容整理到笔记库，重点关注多任务生成式实现、POI场景insight
  - Progress：已完成 ✅
    - 成果：将 IntTravel paper 整理到 `Notes/深度学习推荐系统.md:3472`，包含数据集介绍、三大核心模块（TIP/TSG/TSF）、实验结果与关键 Insight

- [x] OpenClaw深入调研分析
  - Priority：medium
  - Assignee：AI
  - Started At：2026-02-16
  - Executor：Trae
  - Links：https://mp.weixin.qq.com/s/Mkbbqdvxh-95pVlnLv9Wig、`Notes/AI-Agent-Product&amp;PE.md`
  - Definition of Done：完成OpenClaw架构与实现的深入调研分析，整理成笔记
  - Progress：已完成 ✅
    - 成果：将 OpenClaw 深入调研分析整理到 `Notes/AI-Agent-Product&amp;PE.md:123`，包含架构设计、核心子系统、安装使用、技能系统、安全模型、实际部署案例、对本项目的启发等

- [x] 将胡渊鸣文章核心内容整合到 AI-Agent-Product&amp;PE.md
- [x] 将 9 个 Topic 分别整理到笔记库不同位置
- [x] 结合 Trae 机制配置让 AI 长记性
- [x] 公司项目脱敏与创作文件夹管理方案设计
  - Priority：medium
  - Assignee：AI
  - Started At：2026-02-16
  - Completed At：2026-02-16
  - Executor：Trae-1
  - Links：`/Users/bytedance/CS-Notes/创作/公司项目/`、`.trae/documents/todos管理系统.md`
  - Definition of Done：
    - 确定创作文件夹的git管理方案（依靠手动操作确保不被git add，但trae仍可访问）
    - 设计公司项目相关todos/inbox在创作文件夹中的存储结构
    - 优化todos manager以支持创作文件夹的本地任务管理
    - 提供完整的使用指南和操作检查清单
  - Progress：已完成 ✅
    - 成果：
      - 创建了 `创作/公司项目/` 目录结构
      - 设计了todo在md文件开头的格式规范（frontmatter + Todo列表）
      - 创建了示例项目 `创作/公司项目/示例项目/` 及示例任务文件
      - 撰写了操作检查清单 `创作/公司项目/操作检查清单.md`
      - 验证了git状态，确保 `创作/` 在 Untracked files 中，不被git add但trae可访问

- [x] 设计并实现手机提交 → 电脑自动执行的完整闭环
  - Priority：high
  - Started At：2026-02-16
  - Completed At：2026-02-16
  - Executor：Trae-1
  - Links：`Notes/snippets/todo-manager.py`、`Notes/snippets/todo-sync.sh`、`Notes/snippets/todo-prompt.py`、`.trae/documents/INBOX.md`、`.trae/documents/todos管理系统.md`、`Notes/AI-Agent-Product&amp;PE.md`（OpenClaw）、`.trae/documents/OpenClaw集成方案.md`
  - Progress：阶段 1-5 已完成 ✅，阶段 6 为长期目标
    - ✓ 阶段 1+2：创建了 `todo-sync.sh` 脚本，一键拉取 git + 扫描新任务
    - ✓ 阶段 3：添加了日志记录功能（日志目录：`.trae/logs/`）
    - ✓ 阶段 4：调研 Trae 自动执行 todos 的方式 - 结论：Trae 暂无命令行/API 自动触发方式，采用替代方案
    - ✓ 阶段 5：创建 `todo-prompt.py` 脚本，生成待执行任务清单供人工确认后执行
    - ✓ 阶段 5+：创建 OpenClaw 集成方案文档
    - ⏳ 阶段 6：（长期）研究 OpenClaw 架构，探索多渠道消息接入 Lark Bot/Telegram Bot
  - 阶段 4-5 产出：
    - 创建了 `Notes/snippets/todo-prompt.py`：扫描 Pending 任务，生成格式化的执行清单，保存到 `.trae/logs/pending-tasks-*.md`
    - 更新了 `Notes/snippets/todo-sync.sh`：集成 todo-prompt.py，同步流程升级为 4 步骤
    - 创建了 `.trae/documents/OpenClaw集成方案.md`：完整的 OpenClaw 集成设计文档
    - 将 OpenClaw 集成方案拆分成具体 todo 项，明确了需要 User 亲自操作的任务
  - Definition of Done：
    - 电脑端能定期自动拉取 git 仓库最新代码 ✅
    - 自动运行 todo-manager.py 扫描新任务 ✅
    - 探索 Trae 自动执行 todos 的方式（技术难点）✅ - 结论：采用提示清单方案
    - 整体流程可观测，失败有日志记录 ✅
    - 长期：参考 OpenClaw 的 Omni-channel Inbox 思路，接入 Lark/Telegram 等多渠道 ⏳
  - Plan：
    - 1. ✓ 先设计一个定期拉取仓库的脚本（已完成：todo-sync.sh）
    - 2. ✓ 集成 todo-manager.py 到自动化流程中（已完成）
    - 3. ✓ 调研 Trae 是否有命令行/API 可以触发执行 - 结论：暂无，采用替代方案
    - 4. ✓ 如无法自动触发 Trae，则改为生成提示清单，人工确认后执行 - 已创建 todo-prompt.py
    - 5. ✓ 添加日志记录和错误处理
    - 6. ⏳ （长期）研究 OpenClaw 架构，探索多渠道消息接入 Lark Bot/Telegram Bot

- [x] 建立公司项目创作标准化流程
  - Priority：high
  - Assignee：AI
  - Feedback Required：否
  - Links：`公司项目/01-公司项目创作pipeline.md`
  - Definition of Done：
    - 创建通用的公司项目创作pipeline文档
    - 包含从想法到完整文档的完整流程
    - 提供可复用的模板和方法论
  - Completed At：2026-02-16
  - Result：已完成公司项目创作标准化流程建立，具体内容：
    - 创建了 `公司项目/01-公司项目创作pipeline.md`，包含5个阶段：
      - 阶段一：想法捕获与结构化
      - 阶段二：调研与信息收集
      - 阶段三：文档架构设计
      - 阶段四：内容填充与写作
      - 阶段五：迭代与完善
    - 提供了标准章节架构模板（7个章节）
    - 提供了项目文件组织建议
    - 更新了 `/.trae/rules/project_rules.md`，让 Trae 在创作时必须参考此 pipeline

- [x] 公司项目文档架构设计方法论总结
  - Priority：medium
  - Assignee：AI
  - Feedback Required：否
  - Links：`公司项目/01-公司项目创作pipeline.md`
  - Definition of Done：
    - 总结出通用的公司项目文档架构设计方法
    - 包含标准章节结构和设计原则
    - 提供可复用的架构模板
  - Completed At：2026-02-16
  - Result：已包含在 `公司项目/01-公司项目创作pipeline.md` 的"阶段三：文档架构设计"中：
    - 设计原则：倒金字塔结构、模块化、渐进式展开、可追溯性
    - 标准章节架构：背景与动机、核心洞察、架构设计、产品思考、实施路径、风险与挑战、总结与展望
    - 已在 03.01 项目中成功验证应用

- [x] OpenViking笔记整理
  - Priority：medium
  - Assignee：AI
  - Feedback Required：否
  - Links：[https://zhuanlan.zhihu.com/p/2000634266720161814、https://openviking.ai/、联网搜索 OpenViking](https://zhuanlan.zhihu.com/p/2000634266720161814、https://openviking.ai/、`公司项目/03.01)
  - Definition of Done：
    - 从提供的链接中提取 OpenViking 相关信息
    - 将知识整合到笔记库中最合适位置
    - 确保格式规范、引用正确
  - Completed At：2026-02-16
  - Result：已将 OpenViking 知识整合到 `Notes/AI-Agent-Product&amp;PE.md` 中，包含：
    * 背景：传统 RAG 的局限性
    * 核心定位：字节跳动火山引擎 2026 年 1 月开源的全球首个专门面向 AI Agent 设计的上下文数据库
    * 设计理念：文件系统式上下文、三层分层上下文（L0/L1/L2）、目录递归检索策略、可观测自演进上下文
    * 虚拟文件系统结构：resources/、user/、agent/ 目录
    * 双存储架构：VikingFS（URI 抽象层）、AGFS（内容存储）、VectorDB（索引存储）
    * API 接口、与 LangChain/DeerFlow 等框架的集成
    * 对本项目的启发：上下文管理、分层加载、可追溯性、与 OpenClaw 结合、双存储架构借鉴

- [x] 创建OpenClaw Workspace Skill目录结构和基础文件
  - Priority：high
  - Assignee：AI
  - Feedback Required：否
  - Links：`.trae/documents/OpenClaw集成方案.md`
  - Started At：2026-02-16
  - Completed At：2026-02-16
  - Executor：Trae-1
  - Definition of Done：
    * \~/.openclaw/workspace/skills/cs-notes-todo/ 目录存在
    * skill.json 配置文件创建完成
    * main.py 脚本框架创建完成
  - Result：已成功完成OpenClaw Workspace Skill目录结构和基础文件创建：
    * 创建了目录：`~/.openclaw/workspace/skills/cs-notes-todo/`
    * 创建了 skill.json：包含名称、版本、描述、入口文件和权限配置
    * 创建了 main.py：包含消息解析、写入INBOX.md的核心逻辑，已添加执行权限
    * 文件结构完整，可以用于后续OpenClaw与Lark机器人集成

- [x] 实现cs-notes-todo Skill的main.py核心逻辑
  - Priority：high
  - Assignee：AI
  - Feedback Required：否
  - Links：`.trae/documents/OpenClaw集成方案.md`、`.trae/documents/INBOX.md`
  - Started At：2026-02-16
  - Completed At：2026-02-16
  - Executor：Trae-1
  - Definition of Done：
    * 可以从命令行参数接收消息
    * 可以将消息解析为todo格式
    * 可以写入 .trae/documents/INBOX.md
  - Result：已成功实现main.py核心逻辑，功能验证通过：
    * 实现了 `parse_message_to_todo()` 函数：将消息解析为结构化todo格式
    * 实现了 `write_to_inbox()` 函数：将todo写入 `.trae/documents/INBOX.md`
    * 实现了 `handle_message()` 主逻辑：协调消息处理流程
    * 验证测试通过：成功从命令行接收消息并写入 INBOX.md
    * 已在 `~/.openclaw/workspace/skills/cs-notes-todo/main.py` 中完成实现

- [x] 明确火山引擎部署架构（Git作为同步层）
  - Priority：high
  - Assignee：AI
  - Feedback Required：否
  - Links：`.trae/documents/OpenClaw集成方案.md`
  - Definition of Done：
    * 明确火山引擎部署与本地Mac部署的核心区别
    * 设计手机→Lark→火山引擎→Git→本地Mac的完整交互架构
    * 写入文档
  - Plan：
    * 分析文件系统访问差异（火山引擎无法直接访问本地Mac文件系统）
    * 设计以Git为中心的同步架构
    * 定义各组件职责：火山引擎OpenClaw（接收消息、写入Git）、本地Mac（拉取任务、执行、推送结果）
  - Result：已在 `todos管理系统.md:3-58` 中完成完整架构设计

- [x] 在火山引擎部署OpenClaw
  - Priority：high
  - Assignee：User
  - Feedback Required：否
  - Started At：2026-02-16 17:14
  - Completed At：2026-02-16
  - Links：`.trae/documents/OpenClaw集成方案.md`、<https://github.com/openclaw/openclaw>、<https://www.volcengine.com/docs/6396/2189942?lang=zh>、`Notes/AI-Agent-Product&amp;PE.md:390-466`
  - Definition of Done：
    * 火山引擎ECS（2核2G）创建成功 ✅
    * OpenClaw预集成模板部署完成（约15-25分钟）✅
    * OpenClaw gateway 可以正常访问 ✅
    * 飞书机器人配对成功，可以正常对话 ✅
  - Plan：
    * 登录火山引擎控制台
    * 选择ECS云服务器（2核2G内存、40G+系统盘）
    * 选择OpenClaw预集成部署模板
    * 按照向导完成配置
    * 执行 `openclaw pairing approve feishu GXVBLJZ4` 完成飞书用户配对
  - Result：部署成功！飞书机器人已正常回复，教程已整理到笔记

- [x] 本地Mac端：完善Git拉取/推送机制
  - Priority：high
  - Assignee：AI
  - Feedback Required：否
  - Started At：2026-02-16
  - Completed At：2026-02-16
  - Links：`Notes/snippets/todo-sync.sh`
  - Definition of Done：
    * todo-sync.sh可以自动拉取Git最新代码 ✅
    * 任务执行完成后可以自动commit & push结果 ✅
    * 支持冲突检测与处理提示 ✅
  - Plan：
    * 增强todo-sync.sh的Git拉取逻辑
    * 添加任务完成后的自动commit & push功能
    * 增加冲突检测与用户提示
  - Result：已成功完善todo-sync.sh，新增功能：
    - 增强的git_pull_enhanced()：分支检查、自动stash、冲突检测
    - git_commit_push()：自动检测更改、交互式提交、push功能
    - check_conflicts()：合并冲突检测
    - check_git_status()：仓库状态检查
    - 流程从4步升级到5步

- [x] 重构Git同步脚本，实现自然语言交互
  - Priority：high
  - Assignee：AI
  - Feedback Required：否
  - Started At：2026-02-16
  - Completed At：2026-02-16
  - Links：`Notes/snippets/todo-pull.sh`、`Notes/snippets/todo-push.sh`、`Notes/snippets/todo-push-commit.sh`、`.trae/documents/PROJECT_CONTEXT.md`
  - Definition of Done：
    * 脚本职责分离：todo-pull.sh、todo-push.sh、todo-push-commit.sh
    * 支持自然语言交互：说"pull"就拉取，说"push"就推送
    * 隐私保护机制：仅允许Notes/、.trae/、创作/，禁止公司项目/
    * 流程验证成功 ✅
  - Plan：
    * 删除旧的todo-sync.sh
    * 创建todo-pull.sh（仅拉取）
    * 创建todo-push.sh（生成变更摘要）
    * 创建todo-push-commit.sh（执行提交）
    * 沉淀使用方法到PROJECT_CONTEXT.md
    * 实际验证完整Push流程
  - Result：已成功完成！验证流程：
    - 用户说"push" → 运行todo-push.sh → 生成变更摘要
    - AI分析摘要 → 生成commit message → 展示给用户确认
    - 用户确认"y" → 运行todo-push-commit.sh → 提交成功！
    - 隐私保护正常工作：自动跳过日志文件和公司项目文件夹

- [x] 零代码配置飞书机器人（火山引擎方式）
  - Priority：high
  - Assignee：User
  - Feedback Required：否
  - Started At：2026-02-16
  - Completed At：2026-02-16
  - Links：`.trae/documents/OpenClaw集成方案.md`、`Notes/AI-Agent-Product&amp;PE.md:390-466`
  - Definition of Done：
    * 飞书企业自建应用创建成功 ✅
    * OpenClaw与Lark机器人连接成功 ✅
    * 可以从Lark接收消息到OpenClaw ✅
  - Plan：
    * 访问OpenClaw WebChat页面
    * 输入：`帮我接飞书`
    * 按照AI指引完成配置（创建应用、配置权限、事件订阅等）
  - Result：飞书机器人已成功配置并正常回复！

- [x] 配置OpenClaw（SOUL/AGENTS人设配置）
  - Priority：high
  - Assignee：User
  - Feedback Required：否
  - Started At：2026-02-16
  - Completed At：2026-02-16
  - Links：飞书机器人对话
  - Definition of Done：
    * 配置好OpenClaw的名字 ✅
    * 配置好OpenClaw的人设（AI助手/其他）✅
    * 配置好OpenClaw的风格（正式/轻松/活泼/温暖）✅
    * 配置好OpenClaw的签名emoji ✅
  - Plan：
    * 和飞书机器人对话，回答初始化问题
    * 配置SOUL.md/AGENTS.md
  - Result：OpenClaw配置成功！
    - 名字：小C
    - 存在方式：AI助手，专注于帮助管理笔记、执行todos、整理知识、创作、写项目，全能专家
    - 风格：理性、精确
    - 签名emoji：📝

---

*最后更新：2026-02-16*
