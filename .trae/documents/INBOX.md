# Inbox

用于随手记录想法与任务（尤其来自手机语音/远程）。Trae 可将 Inbox 条目搬运到 `.trae/documents/todos管理系统.md` 的 Pending，并补齐结构化字段。

## 记录格式（多种选择）

### 格式 1：极简版（手机快速输入推荐）
只需一行，自由发挥：
```
- [一句话描述你的想法/任务]
```
示例：
```
- 把这篇 AI Agent 文章整理到笔记
- 研究 Triton 矩阵乘法优化
- 修正 GPU.md 里的错误
```

### 格式 2：带链接版
直接粘贴链接，Trae 会智能解析：
```
- https://example.com/article - 这篇文章关于 AI Agent，很有价值
```

### 格式 3：结构化版（完整信息）
需要精确控制时使用：
```
- 内容：[任务描述]
  优先级：high | medium | low
  关联文件：[相关文件路径，如 Notes/GPU.md]
  参考链接：[外部链接]
  截止时间：[YYYY-MM-DD，可选]
```

## 使用场景示例

这些是示例，不要直接使用：

### 场景 1：手机上看到一篇好文章（极简）
```
- https://example.com/article - AI Agent 应用案例
```

### 场景 2：突然想到一个想法（极简）
```
- 研究一下如何用 Triton 优化矩阵乘法
```

### 场景 3：需要修复一个笔记中的错误（完整）
```
- 内容：修正 GPU.md 中关于 shared memory 的描述
  优先级：low
  关联文件：Notes/GPU.md
```

## 手机提交方法

### 方法 1：Git 客户端（推荐，Working Copy）
**步骤**：
1. 打开 Working Copy，进入 CS-Notes 仓库
2. 找到本文件，直接在"条目"下方追加内容
3. 输入越简单越好，一行即可
4. Commit 并 Push

**Commit 信息建议**：`Add inbox: [简短描述]`

### 方法 2：远程 SSH
手机通过 SSH 连接到开发机，在终端追加任务条目。

### 方法 3：OpenClaw + Lark/Telegram（未来）
参考 OpenClaw 的思路，通过多渠道（Lark/Telegram 等）接收消息，自动写入 Inbox。

## 条目

（在此下方添加你的任务条目，一行一个最简单）


- 内容：沉淀经验：如何成功获取微信公众号文章内容
  - 优先级：high
  - 关联文件：MEMORY.md
  - 参考链接：
  - 截止时间：

- 内容：测试任务：验证Skill功能
  - 优先级：medium
  - 关联文件：
  - 参考链接：
  - 截止时间：

- [2026-02-16 19:56:46] 测试任务：验证 cs-notes-git-sync 功能
  - Priority：medium
  - Links：
  - Due：

- 内容：解决 OpenClaw Gateway device token mismatch 问题
  - 优先级：high
  - 关联文件：`~/.openclaw/openclaw.json`、`~/.openclaw/devices/paired.json`
  - 参考链接：
  - 截止时间：

