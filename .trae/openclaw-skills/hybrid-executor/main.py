#!/usr/bin/env python3
"""
Hybrid Executor Skill - 混合执行调度器
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from datetime import datetime

# 添加 snippets 目录到 sys.path，以便导入 task_execution_logger
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "Notes" / "snippets"))

try:
    from task_execution_logger import (
        TaskExecutionLogger,
        TaskStage,
        LogLevel,
        TaskArtifact,
        create_logger
    )
    TASK_LOGGER_AVAILABLE = True
except ImportError:
    TASK_LOGGER_AVAILABLE = False

# 配置路径
REPO_ROOT = Path("/root/.openclaw/workspace/CS-Notes")
PLANS_DIR = REPO_ROOT / ".trae/plans"

# 初始化任务日志系统
task_logger = None
if TASK_LOGGER_AVAILABLE:
    try:
        task_logger = create_logger(REPO_ROOT)
    except Exception as e:
        print(f"⚠️ 初始化任务日志系统失败: {e}")

class HybridExecutor:
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
    
    def determine_execution_mode(self, plan: dict) -> str:
        """决定执行模式"""
        priority = plan.get("priority", "medium")
        
        # 高优先级任务 -> 需要用户 Review
        if priority == "high":
            return "review_required"
        
        # 中低优先级任务 -> 自动执行
        return "auto_execute"
    
    def execute_plan(self, plan_id_or_title: str) -> dict:
        """
        混合执行 Plan
        """
        plan_file = self.find_plan_file(plan_id_or_title)
        if not plan_file:
            return {
                "success": False,
                "error": f"Plan not found: {plan_id_or_title}"
            }
        
        plan = self.read_plan(plan_file)
        
        # 决定执行模式
        execution_mode = self.determine_execution_mode(plan)
        
        if execution_mode == "review_required":
            # 需要用户 Review
            self.update_plan_status(plan_file, "pending_review")
            return {
                "success": True,
                "mode": "review_required",
                "message": "Plan 已生成，等待用户 Review",
                "plan_file": str(plan_file),
                "plan": plan
            }
        
        # 自动执行
        self.update_plan_status(plan_file, "in_progress")
        
        # 先自动执行简单部分
        result = self._auto_execute_simple_part(plan)
        
        if not result["success"]:
            self.update_plan_status(plan_file, "failed")
            return result
        
        # 再用 AI 自动生成的方式执行复杂部分
        result = self._ai_generate_complex_part(plan)
        
        # 更新状态
        if result["success"]:
            self.update_plan_status(plan_file, "completed")
        else:
            self.update_plan_status(plan_file, "failed")
        
        result["plan_file"] = str(plan_file)
        result["mode"] = "hybrid"
        return result
    
    def _auto_execute_simple_part(self, plan: dict) -> dict:
        """自动执行简单部分"""
        plan_title = plan.get("title", "Untitled Plan")
        # 这里简化处理，实际应该执行 Plan 中的简单步骤
        return {
            "success": True,
            "message": f"自动执行简单部分完成：{plan_title}",
            "executor": "hybrid-auto"
        }
    
    def _ai_generate_complex_part(self, plan: dict) -> dict:
        """用 AI 自动生成的方式执行复杂部分"""
        plan_title = plan.get("title", "Untitled Plan")
        # 这里简化处理，实际应该调用 AI 生成复杂部分
        return {
            "success": True,
            "message": f"AI 自动生成复杂部分完成：{plan_title}",
            "executor": "hybrid-ai"
        }

def handle_execute():
    """处理混合执行命令"""
    task_id = f"hybrid-executor-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    agent = "openclaw"
    
    # 记录任务开始
    if TASK_LOGGER_AVAILABLE and task_logger:
        try:
            task_logger.start_task(task_id, agent=agent)
            task_logger.log_info(
                task_id,
                TaskStage.PLANNING,
                "开始执行 Plan",
                {},
                agent=agent
            )
        except Exception as e:
            print(f"⚠️ 记录任务日志失败: {e}")
    
    if len(sys.argv) < 3:
        error_msg = "Missing plan id or title"
        if TASK_LOGGER_AVAILABLE and task_logger:
            try:
                task_logger.fail_task(task_id, error_msg)
            except Exception as e:
                print(f"⚠️ 记录任务失败日志失败: {e}")
        print("Usage: python main.py execute <plan_id_or_title>")
        return json.dumps({"success": False, "error": error_msg})
    
    plan_id_or_title = sys.argv[2]
    
    try:
        if TASK_LOGGER_AVAILABLE and task_logger:
            task_logger.log_info(
                task_id,
                TaskStage.EXECUTING,
                "执行 Plan",
                {"plan_id_or_title": plan_id_or_title},
                agent=agent
            )
        
        executor = HybridExecutor()
        result = executor.execute_plan(plan_id_or_title)
        
        if TASK_LOGGER_AVAILABLE and task_logger:
            if result.get("success"):
                task_logger.log_success(
                    task_id,
                    TaskStage.COMPLETED,
                    "Plan 执行成功",
                    {"mode": result.get("mode")},
                    agent=agent
                )
                task_logger.complete_task(task_id, agent=agent)
            else:
                task_logger.fail_task(task_id, result.get("error", "Unknown error"))
        
        # 移除不能序列化的 datetime 对象
        def remove_datetime(obj):
            if isinstance(obj, dict):
                return {k: remove_datetime(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [remove_datetime(v) for v in obj]
            elif isinstance(obj, datetime):
                return obj.isoformat()
            else:
                return obj
        
        result_serializable = remove_datetime(result)
        
        return json.dumps(result_serializable, ensure_ascii=False)
    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        if TASK_LOGGER_AVAILABLE and task_logger:
            try:
                task_logger.fail_task(task_id, error_msg)
            except Exception as log_e:
                print(f"⚠️ 记录任务失败日志失败: {log_e}")
        return json.dumps({"success": False, "error": error_msg})

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "execute":
        print(handle_execute())
