# Lark Bot vs Codex Mobile 手机入口方案调研

## 结论

更推荐分阶段做，而不是一上来重造一个完整 Lark bot：

1. **短期：优先用 Codex mobile/web 直接开任务**。适合“让 Codex 读仓库、改代码、开 PR、解释代码”的标准工程任务，维护成本最低。
2. **同时做一个很薄的 Lark inbox bot**。只接收 `素材：xxx`、`整理笔记：xxx`、链接、语音转文字后的文本，写入本仓库 inbox/todos，并回一个确认消息。
3. **中期再接本机常驻执行器**。如果确实需要“手机一句话，MacBook 本地 Codex 直接执行并持续回报进度”，优先复用 OpenClaw / GolemBot / VibeAround 这一类现成桥接器，不建议从 Lark WebSocket、会话管理、权限审批、日志、恢复全部自研。

核心判断：**Codex mobile/web 是最好的“工程任务入口”，Lark 是最好的“生活流捕获入口”。** 两者不应该互相替代。

## 调研依据

- OpenAI 官方 Codex cloud 文档说明，Codex cloud 可以在后台并行执行任务，并通过 GitHub 连接仓库、创建 PR；这适合手机上发起标准仓库任务：[Codex web](https://developers.openai.com/codex/cloud)。
- OpenAI Codex app 说明强调本地 app/CLI/IDE/cloud 能力贯通、支持多 agent、worktree、skills 和 automations；但它更像 agent 控制台，不是 IM 机器人：[Introducing the Codex app](https://openai.com/index/introducing-the-codex-app/)。
- OpenAI Help Center 区分了 Codex Local 与 Codex Cloud 的管理控制：Local 覆盖 CLI/IDE/app 本地工作流，Cloud 覆盖云端委托任务。这意味着“手机触发云任务”和“手机遥控本机 Mac session”是两类问题：[Using Codex with your ChatGPT plan](https://help.openai.com/en/articles/11369540)。
- Feishu CLI 是官方 Lark/飞书命令行工具，已面向 AI Agent 设计，并提供 messaging/doc/sheet/calendar/task 等技能；本机已经安装 `lark-cli 1.0.23`，不需要从 OpenAPI 底层造轮子：[Feishu CLI](https://feishu-cli.com/)。
- 飞书/Lark bot 的主流实现已经收敛到 WebSocket 长连接，不需要公网 webhook；这适合 MacBook 常驻运行，但需要处理 allowlist、重连、去重、权限和回执：[OpenClaw Feishu docs](https://github.com/openclaw/openclaw/blob/main/docs/channels/feishu.md)、[OpenPRX Lark docs](https://docs.openprx.dev/en/prx/channels/lark)、[Hermes Feishu/Lark docs](https://hermes-agent.nousresearch.com/docs/user-guide/messaging/feishu)。
- 现成桥接器已经覆盖“IM -> 本地 coding agent”这类需求，例如 GolemBot 支持 Codex + Feishu/Slack/Telegram 等，VibeAround 支持 Codex CLI + Feishu/Lark + session handover，HeyAgent 则是 Telegram + Codex/Claude 的轻量方案：[GolemBot](https://github.com/0xranx/golembot)、[VibeAround](https://github.com/jazzenchen/VibeAround)、[HeyAgent](https://www.heyagent.dev/)。
- 小红书整理了“手机控制电脑/服务器上的 Codex”的 4 类方案：Happy、HAPI、Remodex、Claude-to-IM Skill。其中 Happy 被描述为现成配套 App，但依赖自有客户端生态；用户补充观察是 **Happy 连接不稳定**，因此不宜作为核心链路：[Codex接入手机4种方案整理](http://xhslink.com/o/5SXCYtuL0pS)。
- HAPI 是 Happy 的开源/本地优先替代，支持本地 Claude Code / Codex / Gemini / OpenCode，通过 Web / PWA / Telegram Mini App 远程控制；这更适合希望减少第三方官方 server 依赖的路径：[HAPI](https://github.com/tiann/hapi/blob/main/README.md)。
- Remodex 现在已经有 iPhone App 与 Mac bridge，定位是 local-first 的 Codex 手机遥控器，支持 QR 配对、端到端加密、Git 操作和 active run steering。小红书 2026-03-24 提到“还在内测没有 app”的信息已过期：[Remodex](https://www.phodex.app/)。
- Claude-to-IM Skill 已支持 Claude Code / Codex 桥接到 Telegram、Discord、Feishu/Lark、QQ、WeChat，并内置后台 daemon、权限确认、日志脱敏、Feishu 长连接配置指引；它和“用 Lark 像 OpenClaw 一样控制本地 Codex”的目标最贴近：[Claude-to-IM Skill](https://github.com/op7418/Claude-to-IM-skill)。

## 方案对比

| 方案 | 适合做什么 | 不适合做什么 | 维护成本 |
|---|---|---|---|
| Codex mobile/web | 从手机发起标准工程任务、云端跑、开 PR、查看结果 | 本地私密文件、未 push 内容、需要操作当前 Mac 会话的任务 | 低 |
| 极薄 Lark inbox bot | 捕获素材、创建 todo、状态通知、把语音/链接沉淀到仓库 | 长时间自主执行、复杂审批、多人权限 | 低-中 |
| Lark bot + 本地 Codex exec 自研 | 手机遥控 Mac 本地执行、访问本地未提交上下文 | 会话恢复、权限、安全、日志、异常恢复都要自己维护 | 高 |
| OpenClaw / GolemBot / VibeAround | IM 控制本地 agent、会话管理、回执、跨工具扩展 | 需要接受第三方框架约束，需评估安全与可控性 | 中 |
| Happy | 最快试用手机继续本机会话 | 依赖官方 server；国内网络下可能连接不稳定 | 低 |
| HAPI | 自托管/本地优先的 Web/PWA 手机遥控 | 需要自己部署和维护 hub/relay | 中 |
| Remodex | iPhone 原生体验，local-first 控制 Mac 上 Codex | 主要面向 iPhone + Mac，生态更窄 | 低-中 |
| Claude-to-IM Skill | 直接把当前 Codex/Claude 会话桥到 Lark/Telegram/QQ/微信 | 仍需配置 IM bot 权限与后台 daemon | 中 |

## 推荐落地路径

### Phase 1：不要先做执行器，只做手机捕获

先支持这些指令：

```text
素材：<任意文本/链接>
整理笔记：<材料或目标>
新建任务：<任务>
状态
```

行为：

- `素材：` 写入 `.trae/documents/INBOX.md` 或 `.local/lark-material-cache/`，回传材料 ID。
- `整理笔记：` 写入 `.trae/todos/todos.json`，默认 P0/P1，等待 Codex/Trae 执行。
- `状态` 返回最近 todo、in-progress 任务、本机 runner 是否在线。

这一层只需要 `lark-cli event consume im.message.receive_v1 --as bot` 和 `lark-cli im +messages-reply --as bot`，不直接让任意消息改仓库，安全边界清楚。

### Phase 2：把高频整理任务接 Codex，但必须加闸

可以支持：

```text
执行：整理笔记：<xxx>
```

但默认要求：

- 只允许白名单 open_id / chat_id。
- 对会改文件的任务，先生成 plan 或 todo，不直接 push。
- 本地运行 `codex exec -C /Users/bytedance/CS-Notes ...`，输出摘要回 Lark。
- Git 提交仍走现有 `todo-push.sh` / `todo-push-commit.sh` SOP，尤其不能碰 `公司项目/`。

### Phase 3：如果想要完整手机遥控，再选现成桥

优先顺序：

1. **Claude-to-IM Skill**：和“Lark 里直接远程当前 Codex/Claude 会话”最贴近，且明确支持 Feishu/Lark；适合先做一条轻量 IM 桥。
2. **OpenClaw Feishu channel**：和仓库已有 `.openclaw-memory/`、`.trae/openclaw-skills/` 思路最一致，适合继续做个性化 agent。
3. **HAPI**：如果 Happy 连接不稳定，HAPI 的本地优先/PWA 路线更可控，适合作为“非 IM 的手机遥控层”。
4. **Remodex**：如果主要是 iPhone + Mac，且想要原生 App 体验，可以单独试；它不替代 Lark inbox，但能补上“继续本机会话”。
5. **VibeAround / GolemBot**：适合想统一多 IM、多 agent、多入口时再评估。
6. **Happy**：可以快速试用，但不建议作为核心依赖，原因是你已观察到连接不稳定，且小红书也提到其依赖官方 server。
7. **HeyAgent**：如果愿意用 Telegram，最轻；但不解决 Lark 入口。

## 当前建议

你的场景里，MacBook 常开不熄屏是可行执行侧，但入口不要只选一个：

- **日常工程任务**：直接用 Codex mobile/web。
- **碎片素材与笔记指令**：用 Lark bot，因为飞书在手机上发文本、链接、语音很顺手。
- **真正后台执行**：先让 Lark bot 入队，不直接开跑；等 Phase 1 用顺了，再接 Claude-to-IM Skill / OpenClaw / HAPI / Remodex / VibeAround，或本地 `codex exec`。

这样做的好处是：先获得 80% 的手机入口价值，同时不把系统复杂度一次性拉满。
