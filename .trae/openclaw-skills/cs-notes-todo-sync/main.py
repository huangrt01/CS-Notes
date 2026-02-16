#!/usr/bin/env python3
"""
CS-Notes Todo Sync Skill - 拉取 Git、扫描任务、生成执行提示
"""

import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime


REPO_ROOT = Path.home() / ".openclaw" / "workspace" / "CS-Notes"
TODO_PULL_SCRIPT = REPO_ROOT / "Notes" / "snippets" / "todo-pull.sh"


def run_command(cmd: list, cwd: Path = None) -> tuple:
    """运行命令并返回 (返回码, stdout, stderr)"""
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=300
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "Command timed out"
    except Exception as e:
        return -1, "", str(e)


def ensure_repo_exists():
    """确保仓库存在"""
    if not REPO_ROOT.exists():
        print(f"Error: Repository not found at {REPO_ROOT}")
        return False
    return True


def run_todo_pull():
    """运行 todo-pull.sh"""
    if not TODO_PULL_SCRIPT.exists():
        print(f"Error: todo-pull.sh not found at {TODO_PULL_SCRIPT}")
        return False
    
    print("Running todo-pull.sh...")
    code, stdout, stderr = run_command(
        ["bash", str(TODO_PULL_SCRIPT)],
        cwd=REPO_ROOT
    )
    
    if stdout:
        print(stdout)
    if stderr:
        print(stderr)
    
    return code == 0


def main():
    """主函数"""
    print("=" * 80)
    print("CS-Notes Todo Sync Skill")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    if not ensure_repo_exists():
        sys.exit(1)
    
    success = run_todo_pull()
    
    print("=" * 80)
    if success:
        print("Todo sync completed successfully!")
        sys.exit(0)
    else:
        print("Todo sync failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()

