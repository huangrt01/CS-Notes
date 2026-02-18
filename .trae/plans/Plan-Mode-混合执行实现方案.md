# Plan Mode 混合执行实现方案

## 概述

基于用户的语音需求，实现 Plan Mode 的混合执行方式：
- 当遇到复杂任务的时候，先生成 Plan
- 然后先自动执行
- 再用 AI 自动生成的方式

---

## 我的决策

**Plan Mode 配置**：
- Plan Mode 默认开启：混合模式（高优先级任务默认开启）
- Plan 生成方式：AI 自动生成 + 用户编辑
- Review 流程：低优先级任务自动执行，高优先级任务需要 Review
- 批量 Review 粒度：可以逐个查看详情，再批量操作
- Plan 存储位置：单独的 Plan 文件（`.trae/plans/` 目录）

---

## 混合执行流程

```
1. 新任务创建
   ↓
2. 判断任务优先级
   ├─ 高优先级 → 进入 Plan Mode
   └─ 低优先级 → 直接执行
   ↓
3. AI 自动生成 Plan
   ↓
4. 先自动执行 Plan
   ↓
5. 用 AI 自动生成的方式继续
   ↓
6. 任务完成
```

---

## Plan 结构

### Plan 文件存储方案

**目录结构**:
```
.trae/
└── plans/
    ├── 2026-02-17-任务标题-1.md
    ├── 2026-02-17-任务标题-2.md
    └── ...
```

**文件命名规则**: `{YYYY-MM-DD}-{slugified-title}-{random-id}.md`

### Plan 文件格式

```markdown
---
id: plan-20260217-abc123
title: "任务标题"
priority: high
status: pending
created_at: 2026-02-17T10:30:00
updated_at: 2026-02-17T10:30:00
tags: []
---

## 目标
- 明确任务的目标是什么
- 要解决什么问题
- 期望达成什么效果

## 假设
- 任务执行的前提假设
- 哪些条件是已知的
- 哪些条件是需要验证的

## 改动点
- 需要修改哪些文件
- 需要新增哪些文件
- 需要删除哪些文件
- 关键的代码变更点

## 验收标准
- 如何判断任务完成
- 具体的验收条件
- 可量化的指标

## 风险
- 可能遇到的风险
- 风险的应对措施
- 备用方案

## 执行步骤
- 步骤 1：...
- 步骤 2：...
- 步骤 3：...

## 时间估算
- 预计需要多长时间
- 拆分成哪些阶段
- 每个阶段的时间估算

---

## 执行记录
### 2026-02-17 10:35
- 开始执行步骤 1
- ...
```

---

## 实现方案

### 1. Plan Generator Skill

创建 OpenClaw Skill：`.trae/openclaw-skills/plan-generator/`，自动生成 Plan 并写入 `.trae/plans/`。

### 2. Plan Executor Skill

创建 OpenClaw Skill：`.trae/openclaw-skills/plan-executor/`，读取 Plan 并执行。

### 3. 混合执行调度器

创建 OpenClaw Skill：`.trae/openclaw-skills/hybrid-executor/`，调度混合执行流程。

---

## 下一步

- [x] 创建 `.trae/plans/` 目录
- [ ] 创建 Plan Generator Skill（`.trae/openclaw-skills/plan-generator/`）
- [ ] 创建 Plan Executor Skill（`.trae/openclaw-skills/plan-executor/`）
- [ ] 创建 Hybrid Executor Skill（`.trae/openclaw-skills/hybrid-executor/`）
- [ ] 测试混合执行流程

---

## 决策记录

所有决策已记录，等待用户确认后再继续实现。

## OpenClaw 交互技术设计

### OpenClaw 核心架构回顾

OpenClaw 是一个开源 AI Agent 框架，采用 "配置优先" 理念，通过 Markdown 定义智能体人格与行为。核心特性包括：
- **Skill 系统**：内置、托管和工作区三种技能扩展机制
- **本地优先架构**：控制平面运行在 `ws://127.0.0.1:18789`
- **多渠道接入**：Lark（飞书）、Telegram、Slack 等
- **心跳系统**：主动行为调度

### Plan Generator/Executor 与 OpenClaw 的交互方式

#### 方式一：OpenClaw Skill 包装（推荐）

将 Plan Generator 和 Plan Executor 包装成 OpenClaw Workspace Skills。

##### 架构图

```
┌─────────────────────────────────────────────────────────────┐
│                      OpenClaw Runtime                        │
│  (ws://127.0.0.1:18789)                                     │
│                                                               │
│  ┌─────────────────────────────────────────────────────┐   │
│  │         Plan Generator Skill                         │   │
│  │  - 接收用户任务需求                                  │   │
│  │  - 调用 AI 生成 Plan                                 │   │
│  │  - 写入 todos管理系统.md                             │   │
│  └───────────────────┬─────────────────────────────────┘   │
│                      │                                        │
│                      ▼                                        │
│  ┌─────────────────────────────────────────────────────┐   │
│  │         Plan Executor Skill                          │   │
│  │  - 读取 Plan                                         │   │
│  │  - 调用 Trae Agent / 方舟模型执行任务                │   │
│  │  - 更新任务状态                                      │   │
│  └───────────────────┬─────────────────────────────────┘   │
│                      │                                        │
└──────────────────────┼────────────────────────────────────────┘
                       │
                       ▼
              ┌──────────────────┐
              │  本地文件系统     │
              │  (CS-Notes 仓库) │
              └──────────────────┘
```

##### Plan Generator Skill 设计

**目录结构**:
```
.trae/openclaw-skills/plan-generator/
├── skill.json
├── main.py
└── README.md
```

**skill.json**:
```json
{
  "name": "plan-generator",
  "version": "0.1.0",
  "description": "Plan Generator - 自动生成任务执行计划",
  "author": "AI",
  "entry": "main.py",
  "capabilities": ["filesystem"],
  "commands": [
    {
      "name": "generate",
      "description": "生成执行计划",
      "handler": "handle_generate"
    }
  ]
}
```

**main.py**:
```python
#!/usr/bin/env python3
"""
Plan Generator Skill - 自动生成任务执行计划
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

REPO_ROOT = Path("/Users/bytedance/CS-Notes")
PLANS_DIR = REPO_ROOT / ".trae/plans"

class PlanGenerator:
    def __init__(self):
        self.repo_root = REPO_ROOT
        self.plans_dir = PLANS_DIR
        self.plans_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_plan(self, task_description: str, priority: str = "medium") -> dict:
        """
        生成 Plan（此处为示例，实际应调用 AI）
        """
        import uuid
        plan_id = f"plan-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:8]}"
        
        plan = {
            "id": plan_id,
            "title": task_description[:50],
            "goal": task_description,
            "priority": priority,
            "status": "pending",
            "assumptions": [
                "文件系统可访问",
                "AI 模型可用",
                "Git 仓库状态正常"
            ],
            "changes": {
                "modify": [],
                "add": [],
                "delete": []
            },
            "acceptance_criteria": [
                "任务目标达成",
                "代码通过验证"
            ],
            "risks": [
                {"risk": "AI 生成错误", "mitigation": "人工审核"}
            ],
            "steps": [
                "步骤 1: 分析任务",
                "步骤 2: 执行操作",
                "步骤 3: 验证结果"
            ],
            "time_estimate": "30分钟",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "tags": []
        }
        return plan
    
    def write_plan_to_file(self, plan: dict) -> Path:
        """将 Plan 写入单独的文件"""
        import re
        slugified_title = re.sub(r'[^\w\-]+', '-', plan['title'].lower()).strip('-')
        filename = f"{datetime.now().strftime('%Y-%m-%d')}-{slugified_title}-{plan['id'].split('-')[-1]}.md"
        file_path = self.plans_dir / filename
        
        plan_markdown = self._format_plan_as_markdown(plan)
        
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(plan_markdown)
        
        return file_path
    
    def _format_plan_as_markdown(self, plan: dict) -> str:
        """将 Plan 格式化为 Markdown（带 YAML frontmatter）"""
        lines = []
        
        # YAML frontmatter
        lines.append("---")
        lines.append(f"id: {plan['id']}")
        lines.append(f'title: "{plan["title"]}"')
        lines.append(f"priority: {plan['priority']}")
        lines.append(f"status: {plan['status']}")
        lines.append(f"created_at: {plan['created_at']}")
        lines.append(f"updated_at: {plan['updated_at']}")
        lines.append(f"tags: {plan['tags']}")
        lines.append("---")
        lines.append("")
        
        # Plan 内容
        lines.append("## 目标")
        lines.append(f"- {plan['goal']}")
        lines.append("")
        lines.append("## 假设")
        for assumption in plan['assumptions']:
            lines.append(f"- {assumption}")
        lines.append("")
        lines.append("## 改动点")
        lines.append(f"- 修改: {', '.join(plan['changes']['modify']) or '无'}")
        lines.append(f"- 新增: {', '.join(plan['changes']['add']) or '无'}")
        lines.append(f"- 删除: {', '.join(plan['changes']['delete']) or '无'}")
        lines.append("")
        lines.append("## 验收标准")
        for criteria in plan['acceptance_criteria']:
            lines.append(f"- {criteria}")
        lines.append("")
        lines.append("## 风险")
        for risk in plan['risks']:
            lines.append(f"- {risk['risk']}: {risk['mitigation']}")
        lines.append("")
        lines.append("## 执行步骤")
        for step in plan['steps']:
            lines.append(f"- {step}")
        lines.append("")
        lines.append("## 时间估算")
        lines.append(f"- 预计: {plan['time_estimate']}")
        lines.append("")
        lines.append("---")
        lines.append("")
        lines.append("## 执行记录")
        
        return "\n".join(lines)

def handle_generate():
    """处理生成 Plan 命令"""
    if len(sys.argv) < 3:
        print("Usage: python main.py generate <task_description> [priority]")
        return json.dumps({"success": False, "error": "Missing task description"})
    
    task_description = sys.argv[2]
    priority = sys.argv[3] if len(sys.argv) > 3 else "medium"
    
    generator = PlanGenerator()
    plan = generator.generate_plan(task_description, priority)
    file_path = generator.write_plan_to_file(plan)
    
    return json.dumps({
        "success": True,
        "plan": plan,
        "file_path": str(file_path),
        "message": f"Plan 已生成并写入 {file_path}"
    })

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "generate":
        print(handle_generate())
```

##### Plan Executor Skill 设计

**目录结构**:
```
.trae/openclaw-skills/plan-executor/
├── skill.json
├── main.py
└── README.md
```

**skill.json**:
```json
{
  "name": "plan-executor",
  "version": "0.1.0",
  "description": "Plan Executor - 执行任务计划",
  "author": "AI",
  "entry": "main.py",
  "capabilities": ["filesystem", "shell"],
  "commands": [
    {
      "name": "execute",
      "description": "执行 Plan",
      "handler": "handle_execute"
    }
  ]
}
```

**main.py**:
```python
#!/usr/bin/env python3
"""
Plan Executor Skill - 执行任务计划
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from datetime import datetime

REPO_ROOT = Path("/Users/bytedance/CS-Notes")
PLANS_DIR = REPO_ROOT / ".trae/plans"

class PlanExecutor:
    def __init__(self):
        self.repo_root = REPO_ROOT
        self.plans_dir = PLANS_DIR
    
    def find_plan_file(self, plan_id_or_title: str) -> Path:
        """根据 Plan ID 或标题查找 Plan 文件"""
        if not self.plans_dir.exists():
            return None
        
        # 先尝试按 ID 查找
        for plan_file in self.plans_dir.glob("*.md"):
            content = plan_file.read_text(encoding="utf-8")
            if f"id: {plan_id_or_title}" in content:
                return plan_file
        
        # 再尝试按标题查找
        for plan_file in self.plans_dir.glob("*.md"):
            content = plan_file.read_text(encoding="utf-8")
            if plan_id_or_title.lower() in content.lower():
                return plan_file
        
        return None
    
    def read_plan(self, plan_file: Path) -> dict:
        """读取 Plan 文件"""
        import yaml
        content = plan_file.read_text(encoding="utf-8")
        
        # 解析 YAML frontmatter
        if content.startswith("---"):
            _, frontmatter, body = content.split("---", 2)
            plan_meta = yaml.safe_load(frontmatter)
            return {**plan_meta, "body": body}
        
        return {"body": content}
    
    def update_plan_status(self, plan_file: Path, status: str):
        """更新 Plan 状态"""
        import yaml
        content = plan_file.read_text(encoding="utf-8")
        
        if content.startswith("---"):
            _, frontmatter, body = content.split("---", 2)
            plan_meta = yaml.safe_load(frontmatter)
            plan_meta["status"] = status
            plan_meta["updated_at"] = datetime.now().isoformat()
            
            # 重新写入
            new_content = "---\n"
            new_content += yaml.dump(plan_meta, allow_unicode=True)
            new_content += "---\n"
            new_content += body
            
            plan_file.write_text(new_content, encoding="utf-8")
    
    def execute_plan(self, plan_id_or_title: str) -> dict:
        """
        执行 Plan（调用 Trae Agent 或方舟模型）
        """
        plan_file = self.find_plan_file(plan_id_or_title)
        if not plan_file:
            return {
                "success": False,
                "error": f"Plan not found: {plan_id_or_title}"
            }
        
        plan = self.read_plan(plan_file)
        
        # 更新状态为 in_progress
        self.update_plan_status(plan_file, "in_progress")
        
        # 方案 1: 调用 Trae Agent
        result = self._execute_with_trae_agent(plan)
        
        # 方案 2: 调用方舟模型（当前 Trae IDE 内置能力）
        # result = self._execute_with_fangzhou(plan)
        
        # 更新状态
        if result["success"]:
            self.update_plan_status(plan_file, "completed")
        else:
            self.update_plan_status(plan_file, "failed")
        
        result["plan_file"] = str(plan_file)
        return result
    
    def _execute_with_trae_agent(self, plan: dict) -> dict:
        """通过 Trae Agent 执行"""
        plan_title = plan.get("title", "Untitled Plan")
        cmd = [
            "bash", "-c",
            f"cd {self.repo_root} && "
            f"python3 -c \"from trae_agent import TraeAgent; agent = TraeAgent(); result = agent.run('{plan_title}'); print(result)\""
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600
            )
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "executor": "trae-agent"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "executor": "trae-agent"
            }
    
    def _execute_with_fangzhou(self, plan: dict) -> dict:
        """通过方舟模型执行（当前 Trae IDE）"""
        # 注：方舟模型是当前 Trae IDE 的内置能力
        # Plan Executor Skill 可以通过 OpenClaw 触发，
        # 然后通知用户在 Trae IDE 中继续执行
        plan_title = plan.get("title", "Untitled Plan")
        return {
            "success": True,
            "message": f"Plan '{plan_title}' 已准备好，请在 Trae IDE 中继续执行",
            "executor": "fangzhou",
            "action_required": "user_continue_in_trae_ide",
            "plan": plan
        }

def handle_execute():
    """处理执行 Plan 命令"""
    if len(sys.argv) < 3:
        print("Usage: python main.py execute <plan_id_or_title>")
        return json.dumps({"success": False, "error": "Missing plan id or title"})
    
    plan_id_or_title = sys.argv[2]
    
    executor = PlanExecutor()
    result = executor.execute_plan(plan_id_or_title)
    
    return json.dumps(result, ensure_ascii=False)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "execute":
        print(handle_execute())
```

#### 方式二：混合执行调度器（Hybrid Executor）

通过一个统一的调度器协调 OpenClaw 和 Trae IDE：

```
┌─────────────────────────────────────────────────────────────┐
│                    Hybrid Executor                            │
│  (位于 .trae/openclaw-skills/hybrid-executor/)              │
└────────────────────┬────────────────────────────────────────┘
                     │
         ┌───────────┴───────────┐
         │                       │
         ▼                       ▼
┌──────────────────┐    ┌──────────────────┐
│  OpenClaw Skill  │    │   Trae IDE       │
│  (Plan Generator)│    │   (任务执行)      │
│  (Plan Executor) │    │                  │
└──────────────────┘    └──────────────────┘
```

### 完整交互流程

```
1. 用户通过 Lark / WebChat 发送任务
   ↓
2. OpenClaw 接收消息
   ↓
3. 调用 Plan Generator Skill
   - 解析任务需求
   - 调用 AI 生成 Plan
   - 写入 todos管理系统.md
   ↓
4. （可选）用户在 Lark 中 Review Plan
   ↓
5. 调用 Plan Executor Skill
   - 读取 Plan
   - 触发 Trae IDE 执行（方式 1：Trae Agent；方式 2：通知用户）
   ↓
6. Trae IDE 执行任务
   ↓
7. 任务完成，更新状态
   ↓
8. OpenClaw 通过 Lark 通知用户
```

### 关键技术点

1. **Plan 单独文件存储方案**
   - Plan 存储在 `.trae/plans/` 目录下，每个 Plan 一个文件
   - 文件命名规则：`{YYYY-MM-DD}-{slugified-title}-{random-id}.md`
   - 使用 YAML frontmatter 存储元数据（id、title、priority、status 等）
   - Git 作为同步机制

2. **Plan 状态管理**
   - `pending`: 待执行
   - `in_progress`: 执行中
   - `completed`: 已完成
   - `failed`: 失败

3. **OpenClaw Skill 的能力**
   - `filesystem`: 读写文件
   - `shell`: 执行命令
   - 通过标准输出返回 JSON 结果
   - 使用 `[OPENCLAW_MESSAGE_SEND]` 标记发送消息

4. **与 Trae IDE 的衔接**
   - 方式 A: 通过 Trae Agent（独立进程）
   - 方式 B: 通过文件共享 + 用户手动在 Trae IDE 中继续
   - 方式 C: 未来可以通过 WebSocket/API 直接通信
