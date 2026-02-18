#!/usr/bin/env python3
"""
Plan Generator Skill - 自动生成任务执行计划
"""

import os
import sys
import json
import uuid
import re
from pathlib import Path
from datetime import datetime

# 配置路径
REPO_ROOT = Path("/root/.openclaw/workspace/CS-Notes")
PLANS_DIR = REPO_ROOT / ".trae/plans"

class PlanGenerator:
    def __init__(self):
        self.repo_root = REPO_ROOT
        self.plans_dir = PLANS_DIR
        self.plans_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_plan(self, task_description: str, priority: str = "medium") -> dict:
        """
        生成 Plan
        """
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
