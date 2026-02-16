#!/usr/bin/env python3
"""
CS-Notes Git Sync Skill - 接收 Lark 消息并同步到 Git 仓库
"""

import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime
import json


REPO_NAME = "CS-Notes"
REPO_URL = "https://github.com/huangrt01/CS-Notes.git"
WORKSPACE_ROOT = Path.home() / ".openclaw" / "workspace"
REPO_PATH = WORKSPACE_ROOT / REPO_NAME
INBOX_PATH = REPO_PATH / ".trae" / "documents" / "INBOX.md"


def run_command(cmd: list, cwd: Path = None) -> tuple:
    """运行命令并返回 (返回码, stdout, stderr)"""
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=60
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "Command timed out"
    except Exception as e:
        return -1, "", str(e)


def ensure_repo_exists():
    """确保仓库存在，如果不存在则克隆"""
    WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)
    
    if not REPO_PATH.exists():
        print(f"Cloning repository from {REPO_URL}...")
        code, stdout, stderr = run_command(
            ["git", "clone", REPO_URL, str(REPO_PATH)],
            cwd=WORKSPACE_ROOT
        )
        if code != 0:
            print(f"Clone failed: {stderr}")
            return False
        print("Clone successful")
    return True


def git_pull():
    """拉取最新代码"""
    print("Pulling latest changes...")
    code, stdout, stderr = run_command(["git", "pull"], cwd=REPO_PATH)
    if code != 0:
        print(f"Pull failed: {stderr}")
        return False
    print("Pull successful")
    return True


def git_commit_and_push(commit_message: str):
    """提交并推送更改"""
    print("Adding changes...")
    code, stdout, stderr = run_command(["git", "add", "."], cwd=REPO_PATH)
    if code != 0:
        print(f"Add failed: {stderr}")
        return False
    
    print(f"Committing with message: {commit_message}")
    code, stdout, stderr = run_command(["git", "commit", "-m", commit_message], cwd=REPO_PATH)
    if code != 0:
        print(f"Commit failed: {stderr}")
        return False
    
    print("Pushing changes...")
    code, stdout, stderr = run_command(["git", "push"], cwd=REPO_PATH)
    if code != 0:
        print(f"Push failed: {stderr}")
        return False
    
    print("Push successful")
    return True


def parse_message_to_todo(message: str) -> str:
    """将消息解析为 todo 格式"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = []
    lines.append(f"- [{timestamp}] {message.strip()}")
    lines.append("  - Priority：medium")
    lines.append("  - Links：")
    lines.append("  - Due：")
    return "\n".join(lines)


def write_to_inbox(todo_content: str):
    """写入 INBOX.md"""
    INBOX_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    if not INBOX_PATH.exists():
        INBOX_PATH.write_text("# INBOX\n\n", encoding="utf-8")
    
    with open(INBOX_PATH, "a", encoding="utf-8") as f:
        f.write("\n")
        f.write(todo_content)
        f.write("\n")


def handle_message(message: str) -> str:
    """处理消息"""
    if not ensure_repo_exists():
        return "Error: Failed to ensure repository exists"
    
    if not git_pull():
        return "Error: Failed to pull latest changes"
    
    todo = parse_message_to_todo(message)
    write_to_inbox(todo)
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    commit_message = f"Add task from Lark: {timestamp}"
    
    if not git_commit_and_push(commit_message):
        return "Error: Failed to commit and push changes"
    
    return f"Success: Added task to Inbox and synced to Git: {message[:50]}..."


def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("Usage: python main.py <message>")
        sys.exit(1)
    
    message = " ".join(sys.argv[1:])
    result = handle_message(message)
    print(result)


if __name__ == "__main__":
    main()

