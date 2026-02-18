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

# 配置路径
REPO_ROOT = Path("/root/.openclaw/workspace/CS-Notes")
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
        try:
            import yaml
            content = plan_file.read_text(encoding="utf-8")
            
            # 解析 YAML frontmatter
            if content.startswith("---"):
                _, frontmatter, body = content.split("---", 2)
                plan_meta = yaml.safe_load(frontmatter)
                return {**plan_meta, "body": body}
            
            return {"body": content}
        except Exception as e:
            return {"error": str(e), "body": content}
    
    def update_plan_status(self, plan_file: Path, status: str):
        """更新 Plan 状态"""
        try:
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
        except Exception as e:
            print(f"Error updating plan status: {e}")
    
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
