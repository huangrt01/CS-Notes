# OpenClaw Cron 配置与使用

## 📋 概述

本文档记录 OpenClaw Cron 的配置和使用方法，沉淀在 CS-Notes 库中。

---

## 🎯 当前 OpenClaw Cron 配置

### 查看当前 Cron Jobs

```bash
openclaw cron list
```

### 当前配置（2026-02-18）

| ID                                   | Name                     | Schedule                         | Next       | Last       | Status    | Target    | Agent     |
|--------------------------------------|--------------------------|----------------------------------|------------|------------|-----------|-----------|-----------|
| 7e09e575-b292-4269-b1b1-19f8d985df44 | Session Manager Check    | cron */30 * * * *                | &lt;1m ago   | 30m ago    | running   | main      | default   |
| c1c6d840-4c7f-4f0e-9716-f3fdb8be2c09 | Top Lean AI Monitor      | cron 0 */6 * * *                 | in 4h      | 1h ago     | ok        | main      | default   |

---

## 📝 Cron Job 说明

### 1. Session Manager Check

**Schedule:** `cron */30 * * * *`（每 30 分钟运行一次）

**功能：**
- 检查 session 状态
- 如果发现需要切换 session，提醒用户
- 记录检查结果到状态文件

**检查项：**
- 消息数量（≥30条提醒，≥50条强烈建议切换）
- 运行时间（≥12小时提醒，≥24小时强烈建议切换）

---

### 2. Top Lean AI Monitor

**Schedule:** `cron 0 */6 * * *`（每 6 小时运行一次）

**功能：**
- 监控 Top Lean AI 榜单的变化
- 记录更新历史
- 提供分析报告

---

## 🛠️ OpenClaw Cron 常用命令

### 查看 Cron Jobs

```bash
openclaw cron list
```

### 添加 Cron Job

```bash
openclaw cron add
```

### 更新 Cron Job

```bash
openclaw cron update <job-id>
```

### 删除 Cron Job

```bash
openclaw cron remove <job-id>
```

### 立即运行 Cron Job

```bash
openclaw cron run <job-id>
```

### 查看 Cron Job 运行历史

```bash
openclaw cron runs <job-id>
```

---

## 📊 Cron 表达式说明

### Cron 表达式格式

```
* * * * *
│ │ │ │ │
│ │ │ │ └─ 星期 (0-6, 0=周日)
│ │ │ └─── 月份 (1-12)
│ │ └───── 日期 (1-31)
│ └─────── 小时 (0-23)
└───────── 分钟 (0-59)
```

### 常用 Cron 表达式示例

| 表达式 | 说明 |
|--------|------|
| `*/30 * * * *` | 每 30 分钟运行一次 |
| `0 * * * *` | 每小时运行一次（整点） |
| `0 */6 * * *` | 每 6 小时运行一次 |
| `0 9 * * *` | 每天早上 9 点运行一次 |
| `0 4 * * *` | 每天凌晨 4 点运行一次 |
| `0 0 * * 0` | 每周日凌晨 0 点运行一次 |

---

## 🔗 与 HEARTBEAT.md 的关系

### OpenClaw Cron vs HEARTBEAT.md

**OpenClaw Cron:**
- 精确到分钟级的定时任务
- 适用于精确时间要求的任务（如"每天早上 9 点"）
- 通过 `openclaw cron` 命令管理
- 独立的 Cron Job 配置

**HEARTBEAT.md:**
- 基于 OpenClaw 内置的心跳机制
- 适用于周期性检查（如"每小时检查一次"）
- 通过 HEARTBEAT.md 配置
- 与 OpenClaw 运行时集成

### 何时使用 OpenClaw Cron，何时使用 HEARTBEAT.md？

**使用 OpenClaw Cron：**
- 需要精确时间（如"每天早上 9 点"）
- 需要独立的 Cron Job 管理
- 需要查看运行历史

**使用 HEARTBEAT.md：**
- 周期性检查（如"每小时检查一次"）
- 与 OpenClaw 运行时集成
- 不需要精确时间

---

## 📝 配置 Cron Job 的步骤

### 1. 确定任务需求

- 任务需要什么时候运行？
- 需要精确时间还是周期性检查？
- 是否需要查看运行历史？

### 2. 选择配置方式

- 如果需要精确时间 → 使用 OpenClaw Cron
- 如果需要周期性检查 → 使用 HEARTBEAT.md

### 3. 配置 Cron Job

**方式一：OpenClaw Cron**
```bash
openclaw cron add
```

**方式二：HEARTBEAT.md**
编辑 `HEARTBEAT.md`，添加心跳任务

---

## 🎯 最佳实践

### 1. 混合使用 OpenClaw Cron 和 HEARTBEAT.md

- 精确时间要求的任务 → OpenClaw Cron
- 周期性检查的任务 → HEARTBEAT.md

### 2. 定期检查 Cron Job 状态

```bash
openclaw cron list
```

### 3. 记录 Cron Job 配置

- 在文档中记录 Cron Job 的配置
- 记录 Cron Job 的功能和 schedule
- 便于后续维护和调试

---

## 📚 参考文档

- OpenClaw 官方文档：https://docs.openclaw.ai
- Cron 表达式参考：https://crontab.guru/

---

## 📝 历史记录

- 2026-02-18：创建本文档，记录当前 OpenClaw Cron 配置

