# OpenClaw Bot 交互指南

本指南说明如何与 OpenClaw bot 对话，让它使用我们创建的 Skills。

## 前置条件

1. 已成功部署 Skills 到火山引擎 OpenClaw
2. 已配置好飞书机器人（如果用飞书）
3. 能访问 OpenClaw WebChat 或通过飞书与 bot 对话

## 与 OpenClaw Bot 对话的几种方式

### 方式 1: 直接描述任务（自然语言）

你可以直接说你要做什么，让 OpenClaw 自动理解并使用合适的 Skill：

```
帮我把这个任务记录下来：整理一下最近关于 AI Agent 的笔记
```

```
添加一个待办事项：明天之前完成 OpenClaw 集成的测试
```

```
我有一个想法：研究一下如何把 todo-push.sh 也包装成 Skill
```

### 方式 2: 明确指定使用我们的 Skill

你可以明确提到我们的 Skill 名称：

```
使用 cs-notes-git-sync Skill 来添加任务：测试一下完整流程
```

```
调用 cs-notes-git-sync，帮我记录：验证部署是否成功
```

### 方式 3: 结构化任务（推荐）

为了让任务更清晰，可以使用结构化的方式描述：

```
帮我添加一个任务：
- 内容：验证 OpenClaw 部署
- 优先级：high
- 截止时间：今天
- 关联链接：.trae/openclaw-skills/DEPLOYMENT_GUIDE.md
```

### 方式 4: 测试 Skill

直接让 bot 测试我们的 Skill：

```
测试一下 cs-notes-git-sync Skill 是否正常工作
```

```
运行一个简单的测试：添加一个测试任务到 INBOX
```

## 验证 Skill 是否被识别

你可以问 OpenClaw：

```
你有哪些可用的 Skills？
```

```
列出所有已安装的 Skills
```

```
cs-notes-git-sync Skill 是做什么的？
```

## 完整测试流程示例

### 第一步：添加任务（Lark → OpenClaw → Git）

在 Lark 中对 OpenClaw bot 说：

```
帮我添加一个任务：测试完整的端到端流程
优先级：high
```

### 第二步：验证（检查 Git 仓库）

在本地 Mac 上执行：

```bash
cd /Users/bytedance/CS-Notes
git pull
cat .trae/documents/INBOX.md
```

### 第三步：本地执行任务

在本地 Trae 中执行 todo-sync 并处理任务。

### 第四步：任务完成通知（Git → OpenClaw → Lark）

本地完成任务并 push 后，OpenClaw 应该能检测到更新并通过 Lark 通知你。

## 常见问题

### Q: OpenClaw 不使用我的 Skill 怎么办？

A: 
1. 确认 Skill 已正确安装到 `~/.openclaw/workspace/skills/`
2. 检查 `skill.json` 配置是否正确
3. 尝试明确提到 Skill 名称

### Q: 如何调试 Skill？

A:
1. 直接在服务器上运行：`python ~/.openclaw/workspace/skills/cs-notes-git-sync/main.py "测试任务"`
2. 查看 OpenClaw 的日志文件
3. 在 WebChat 中开启 verbose 模式

### Q: Git 权限问题？

A:
1. 确认服务器上的 Git 凭据已配置
2. 检查仓库 URL 是否正确
3. 尝试用 Personal Access Token 替代密码

