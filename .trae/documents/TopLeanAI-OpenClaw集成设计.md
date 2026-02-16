# Top Lean AI 榜单 - OpenClaw 集成设计

## 概述

本设计文档说明如何将 `top-lean-ai-monitor.py` 与 OpenClaw 深度集成，利用 OpenClaw 的原生能力（定时任务、飞书通知等）。

## 核心脚本

**文件**: `/Users/bytedance/CS-Notes/top-lean-ai-monitor.py`

### 主要 API

```python
from top_lean_ai_monitor import TopLeanAIMonitor

monitor = TopLeanAIMonitor()

# 1. 检查更新
result = monitor.check_updates()
# 返回: {
#   "success": True,
#   "timestamp": "2026-02-17T...",
#   "new_companies": [...],
#   "total_companies": 45,
#   "known_companies_count": 45
# }

# 2. 获取状态
status = monitor.get_status()

# 3. 获取所有公司
companies = monitor.get_all_companies()

# 4. 格式化公司消息
msg = monitor.format_company_message(company)
```

### 命令行接口

```bash
# 检查更新（输出 JSON）
python3 top-lean-ai-monitor.py check

# 获取状态（输出 JSON）
python3 top-lean-ai-monitor.py status

# 列出所有公司（输出 JSON）
python3 top-lean-ai-monitor.py list
```

## OpenClaw 集成方案

### 方案一：OpenClaw Skill 包装（推荐）

创建一个 OpenClaw Skill，封装监控功能。

#### Skill 结构

```
.trae/openclaw-skills/top-lean-ai-monitor/
├── skill.json          # Skill 配置
├── main.py             # Skill 入口
└── README.md           # 说明文档
```

#### skill.json

```json
{
  "name": "top-lean-ai-monitor",
  "version": "1.0.0",
  "description": "Top Lean AI 榜单监控",
  "author": "AI",
  "commands": [
    {
      "name": "check",
      "description": "检查榜单更新",
      "handler": "handle_check"
    },
    {
      "name": "status",
      "description": "查看监控状态",
      "handler": "handle_status"
    }
  ]
}
```

#### main.py

```python
import sys
import json
from pathlib import Path

# 添加脚本路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from top_lean_ai_monitor import TopLeanAIMonitor

def handle_check():
    """检查更新并发送飞书通知"""
    monitor = TopLeanAIMonitor()
    result = monitor.check_updates()
    
    if result["success"] and result["new_companies"]:
        # 有新公司，构建消息
        message = "🔔 Top Lean AI 榜单更新!\n\n"
        message += f"发现 {len(result['new_companies'])} 家新公司:\n\n"
        
        for company in result["new_companies"]:
            message += monitor.format_company_message(company)
            message += "\n"
        
        # 发送飞书通知（使用 OpenClaw 原生能力）
        print(f"[OPENCLAW_MESSAGE_SEND]{message}")
    
    return json.dumps(result, ensure_ascii=False)

def handle_status():
    """查看状态"""
    monitor = TopLeanAIMonitor()
    status = monitor.get_status()
    return json.dumps(status, ensure_ascii=False)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        command = sys.argv[1]
        if command == "check":
            print(handle_check())
        elif command == "status":
            print(handle_status())
```

### 方案二：OpenClaw 定时任务

利用 OpenClaw 的定时任务能力，每日运行监控。

#### 定时任务配置

在 OpenClaw 中配置 cron job：

```
0 9 * * *  cd /Users/bytedance/CS-Notes && python3 top-lean-ai-monitor.py check
```

#### 结果处理

OpenClaw 可以：
1. 捕获脚本的 JSON 输出
2. 检测 `new_companies` 字段
3. 如果有新公司，使用 OpenClaw 原生飞书通知能力发送

### 方案三：OpenClaw + Git 同步（最推荐）

结合 `cs-notes-git-sync` Skill，形成完整闭环：

```
OpenClaw 定时任务
    ↓
运行 top-lean-ai-monitor.py check
    ↓
检测到新公司
    ↓
更新 .top-lean-ai-state.json
    ↓
Git commit & push
    ↓
OpenClaw 检测到 Git 变化
    ↓
通过 Lark 发送通知
```

## 使用示例

### 在 OpenClaw 中调用

```python
# Python 方式
from top_lean_ai_monitor import TopLeanAIMonitor

monitor = TopLeanAIMonitor()
result = monitor.check_updates()

if result["new_companies"]:
    # 发送通知
    pass
```

```bash
# 命令行方式
RESULT=$(python3 top-lean-ai-monitor.py check)
HAS_NEW=$(echo "$RESULT" | python3 -c "import sys, json; d=json.load(sys.stdin); print(len(d.get('new_companies', [])) > 0)")

if [ "$HAS_NEW" = "True" ]; then
    # 发送通知
fi
```

## 数据结构

### 公司信息

```json
{
  "rank": "1",
  "name": "Telegram",
  "description": "Messaging",
  "location": "Dubai",
  "annual_revenue": "$1,000,000,000",
  "num_employees": "30",
  "revenue_per_employee": "$33,333,333",
  "profitable": "Yes",
  "total_funding": "$3,200,000,000",
  "valuation": "$30,000,000,000",
  "valuation_per_employee": "$1,000,000,000",
  "founded": "2013",
  "last_updated": "Dec 2024",
  "source": "https://x.com/durov/status/..."
}
```

### check_updates() 返回

```json
{
  "success": true,
  "timestamp": "2026-02-17T02:45:33.817081",
  "new_companies": [
    {
      "name": "NewCompany",
      "info": {...},
      "discovered_at": "2026-02-17T..."
    }
  ],
  "total_companies": 45,
  "known_companies_count": 45
}
```

## 下一步

1. **用户**：将此设计文档转发给 OpenClaw
2. **OpenClaw**：根据设计实现 Skill 或定时任务
3. **OpenClaw**：利用原生飞书通知能力发送更新
4. **用户**：享受每日榜单更新通知！
