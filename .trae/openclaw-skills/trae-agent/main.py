#!/usr/bin/env python3
"""
Trae Agent Skill - 调用 trae-agent 执行复杂任务
利用其强可观测性和完整轨迹记录
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from datetime import datetime


class TraeAgentSkill:
    def __init__(self):
        # 配置路径
        self.trae_agent_path = Path("/root/.openclaw/workspace/trae-agent")
        self.workspace = Path("/root/.openclaw/workspace/CS-Notes")
        self.trajectory_dir = self.trae_agent_path / "trajectories"
        
        # 确保轨迹目录存在
        self.trajectory_dir.mkdir(exist_ok=True)
    
    def run_task(self, task_description, working_dir=None):
        """
        调用 trae-agent 执行任务
        
        Args:
            task_description: 任务描述
            working_dir: 工作目录（默认是 CS-Notes）
        
        Returns:
            dict: 执行结果，包含成功状态、输出、轨迹文件路径等
        """
        if working_dir is None:
            working_dir = str(self.workspace)
        
        # 构建命令
        cmd = [
            "bash", "-c",
            f"export PATH='$HOME/.local/bin:$PATH' && "
            f"cd {self.trae_agent_path} && "
            f"source .venv/bin/activate && "
            f"trae-cli run \"{task_description}\" --working-dir {working_dir}"
        ]
        
        print(f"🚀 调用 trae-agent 执行任务...")
        print(f"📝 任务描述: {task_description}")
        print(f"📂 工作目录: {working_dir}")
        print()
        
        try:
            # 执行命令
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600  # 10 分钟超时
            )
            
            # 收集输出
            stdout = result.stdout
            stderr = result.stderr
            success = result.returncode == 0
            
            # 查找轨迹文件
            trajectory_file = self._find_latest_trajectory()
            
            print()
            print("=" * 60)
            if success:
                print("✅ trae-agent 任务执行成功！")
            else:
                print("❌ trae-agent 任务执行失败！")
            print("=" * 60)
            
            if stdout:
                print()
                print("📤 标准输出:")
                print("-" * 60)
                print(stdout)
                print("-" * 60)
            
            if stderr:
                print()
                print("📥 标准错误:")
                print("-" * 60)
                print(stderr)
                print("-" * 60)
            
            if trajectory_file:
                print()
                print(f"📂 轨迹文件: {trajectory_file}")
            
            return {
                "success": success,
                "stdout": stdout,
                "stderr": stderr,
                "returncode": result.returncode,
                "trajectory_file": str(trajectory_file) if trajectory_file else None,
                "task_description": task_description,
                "working_dir": working_dir,
                "timestamp": datetime.now().isoformat()
            }
            
        except subprocess.TimeoutExpired:
            print()
            print("⏰ 任务超时！（超过 10 分钟）")
            return {
                "success": False,
                "error": "Timeout",
                "task_description": task_description,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            print()
            print(f"❌ 执行出错: {e}")
            return {
                "success": False,
                "error": str(e),
                "task_description": task_description,
                "timestamp": datetime.now().isoformat()
            }
    
    def _find_latest_trajectory(self):
        """查找最新的轨迹文件"""
        if not self.trajectory_dir.exists():
            return None
        
        trajectory_files = list(self.trajectory_dir.glob("trajectory_*.json"))
        if not trajectory_files:
            return None
        
        # 按修改时间排序，取最新的
        trajectory_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return trajectory_files[0]
    
    def get_trajectory(self, trajectory_file):
        """读取轨迹文件"""
        trajectory_path = Path(trajectory_file)
        if not trajectory_path.exists():
            return None
        
        try:
            with open(trajectory_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"[警告] 读取轨迹文件失败: {e}")
            return None


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Trae Agent Skill")
    parser.add_argument("task", help="任务描述")
    parser.add_argument("--working-dir", help="工作目录")
    
    args = parser.parse_args()
    
    skill = TraeAgentSkill()
    result = skill.run_task(args.task, args.working_dir)
    
    # 返回 JSON 格式的结果
    print()
    print("📋 结果摘要:")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
