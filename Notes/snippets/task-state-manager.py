#!/usr/bin/env python3
"""
任务状态管理脚本 - 实现任务恢复机制
"""

import json
from pathlib import Path
from datetime import datetime


class TaskStateManager:
    """任务状态管理器"""
    
    def __init__(self, state_file_path: str = None):
        """初始化"""
        if state_file_path is None:
            state_file_path = "/root/.openclaw/workspace/CS-Notes/.trae/logs/task-state.json"
        
        self.state_file = Path(state_file_path)
        self.state = self.load_state()
    
    def load_state(self):
        """加载任务状态"""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"警告: 无法加载任务状态文件: {e}")
                return {}
        return {}
    
    def save_state(self):
        """保存任务状态"""
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.state_file, 'w', encoding='utf-8') as f:
                json.dump(self.state, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"警告: 无法保存任务状态文件: {e}")
    
    def start_task(self, task_id: str, progress: str = "", next_step: str = ""):
        """开始执行任务"""
        self.state = {
            "current_task": task_id,
            "progress": progress,
            "next_step": next_step,
            "started_at": datetime.now().isoformat(),
            "interrupted": False,
            "completed": False
        }
        self.save_state()
        print(f"✅ 任务开始: {task_id}")
    
    def update_progress(self, progress: str, next_step: str = ""):
        """更新任务进度"""
        if self.state:
            self.state["progress"] = progress
            if next_step:
                self.state["next_step"] = next_step
            self.save_state()
            print(f"📝 任务进度更新: {progress}")
    
    def mark_interrupted(self):
        """标记任务被打断"""
        if self.state and not self.state.get("completed", False):
            self.state["interrupted"] = True
            self.state["interrupted_at"] = datetime.now().isoformat()
            self.save_state()
            print(f"⚠️ 任务被打断")
    
    def mark_completed(self):
        """标记任务完成"""
        if self.state:
            self.state["completed"] = True
            self.state["completed_at"] = datetime.now().isoformat()
            self.save_state()
            print(f"✅ 任务完成")
    
    def clear_state(self):
        """清除任务状态"""
        self.state = {}
        if self.state_file.exists():
            self.state_file.unlink()
        print(f"🗑️ 任务状态已清除")
    
    def has_interrupted_task(self):
        """检查是否有被打断的任务"""
        return self.state and self.state.get("interrupted", False) and not self.state.get("completed", False)
    
    def get_interrupted_task(self):
        """获取被打断的任务"""
        if self.has_interrupted_task():
            return {
                "task_id": self.state.get("current_task"),
                "progress": self.state.get("progress"),
                "next_step": self.state.get("next_step"),
                "started_at": self.state.get("started_at"),
                "interrupted_at": self.state.get("interrupted_at")
            }
        return None


def main():
    """主函数 - 测试用"""
    import sys
    
    if len(sys.argv) < 2:
        print("用法:")
        print("  python task-state-manager.py start <task_id> [progress] [next_step]")
        print("  python task-state-manager.py update <progress> [next_step]")
        print("  python task-state-manager.py interrupt")
        print("  python task-state-manager.py complete")
        print("  python task-state-manager.py clear")
        print("  python task-state-manager.py check")
        return
    
    manager = TaskStateManager()
    command = sys.argv[1]
    
    if command == "start":
        if len(sys.argv) < 3:
            print("错误: 缺少 task_id 参数")
            return
        task_id = sys.argv[2]
        progress = sys.argv[3] if len(sys.argv) > 3 else ""
        next_step = sys.argv[4] if len(sys.argv) > 4 else ""
        manager.start_task(task_id, progress, next_step)
    
    elif command == "update":
        if len(sys.argv) < 3:
            print("错误: 缺少 progress 参数")
            return
        progress = sys.argv[2]
        next_step = sys.argv[3] if len(sys.argv) > 3 else ""
        manager.update_progress(progress, next_step)
    
    elif command == "interrupt":
        manager.mark_interrupted()
    
    elif command == "complete":
        manager.mark_completed()
    
    elif command == "clear":
        manager.clear_state()
    
    elif command == "check":
        if manager.has_interrupted_task():
            task = manager.get_interrupted_task()
            print(f"⚠️ 发现被打断的任务:")
            print(f"  Task ID: {task['task_id']}")
            print(f"  进度: {task['progress']}")
            print(f"  下一步: {task['next_step']}")
            print(f"  开始时间: {task['started_at']}")
            print(f"  打断时间: {task['interrupted_at']}")
        else:
            print("✅ 没有被打断的任务")


if __name__ == "__main__":
    main()
