# Session Optimizer 使用指南

## 自动化运行 `session-optimizer.py check` 的方案

## 方案一：HEARTBEAT.md 心跳机制（推荐！）

### 已配置！

已在 `HEARTBEAT.md` 中添加了 session 状态检查任务。

### 工作原理：
- OpenClaw 心跳时自动检查 session 状态
- 基于 OpenClaw 现有能力，不侵入内部代码

### 使用方法：
无需额外配置，OpenClaw 心跳时自动执行

---

## 方案二：手动定时运行（备选方案）

### 使用 cron（需要配置 cron job

如果需要更频繁检查，可以手动配置 cron job。

### 步骤：

```bash
# 编辑 crontab
crontab -e

# 添加每小时检查一次
0 * * * * cd /root/.openclaw/workspace/CS-Notes && python3 session-optimizer.py check >> /var/log/session-optimizer.log 2>&1
```

---

## 方案三：对话前手动检查（最简单）

### 使用方法：
每次对话前手动运行：

```bash
cd /root/.openclaw/workspace/CS-Notes
python3 session-optimizer.py check
```

### 如果看到警告：
在 OpenClaw TUI 中执行 `/reset` 命令切换 session

---

## session-optimizer.py 命令参考

```bash
# 检查 session 状态
python3 session-optimizer.py check

# 记录一条消息并检查
python3 session-optimizer.py log

# 准备重置 session
python3 session-optimizer.py reset

# 查看历史记录
python3 session-optimizer.py history
```

---

## 重要原则

⚠️ 基于 OpenClaw 现有能力，不侵入内部代码！

- ✅ 不修改 OpenClaw 源代码
- ✅ 不修改 OpenClaw 配置文件
- ✅ 仅使用 OpenClaw 已提供的功能
- ✅ 推荐使用 OpenClaw 内置的 `/reset` 命令

---

## 检查项

- 消息数量：
  - ≥30条：提醒
  - ≥50条：强烈建议切换

- 运行时间：
  - ≥12小时：提醒
  - ≥24小时：强烈建议切换

---

## GitHub Commit

- https://github.com/huangrt01/CS-Notes/commit/c619828
