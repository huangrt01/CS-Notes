#!/usr/bin/env python3
"""
Todo Finder - 从 todos.json 和归档中以多种匹配方式查找 todo
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

# 配置路径
REPO_ROOT_CANDIDATES = [
    Path("/root/.openclaw/workspace/CS-Notes"),
    Path("/Users/bytedance/CS-Notes"),
    Path(__file__).parent.parent.parent.parent
]

REPO_ROOT = None
for candidate in REPO_ROOT_CANDIDATES:
    if candidate.exists():
        REPO_ROOT = candidate
        break

if REPO_ROOT is None:
    REPO_ROOT = Path.cwd()

TODOS_JSON_PATH = REPO_ROOT / ".trae/todos/todos.json"
ARCHIVE_DIR = REPO_ROOT / ".trae/todos/archive"


def load_todos_json():
    """加载 todos.json 文件"""
    if not TODOS_JSON_PATH.exists():
        print(f"Error: {TODOS_JSON_PATH} does not exist")
        return None
    
    with open(TODOS_JSON_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_archive_todos():
    """加载归档中的 todos"""
    archive_todos = []
    
    if not ARCHIVE_DIR.exists():
        return archive_todos
    
    for archive_file in ARCHIVE_DIR.glob("*.json"):
        try:
            with open(archive_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    archive_todos.extend(data)
                elif isinstance(data, dict) and "todos" in data:
                    archive_todos.extend(data["todos"])
        except Exception as e:
            print(f"Warning: Could not load {archive_file}: {e}")
    
    return archive_todos


def match_todo(todo, args):
    """检查 todo 是否匹配查找条件"""
    # 按 ID 匹配
    if args.id:
        if todo.get("id") != args.id:
            return False
    
    # 按标题匹配
    if args.title:
        title = todo.get("title", "").lower()
        if args.title.lower() not in title:
            return False
    
    # 按关键词匹配
    if args.keyword:
        keyword = args.keyword.lower()
        title = todo.get("title", "").lower()
        progress = todo.get("progress", "").lower()
        links = " ".join(todo.get("links", [])).lower()
        if keyword not in title and keyword not in progress and keyword not in links:
            return False
    
    # 按状态匹配
    if args.status:
        if todo.get("status") != args.status:
            return False
    
    # 按优先级匹配
    if args.priority:
        if todo.get("priority") != args.priority:
            return False
    
    # 按负责人匹配
    if args.assignee:
        if todo.get("assignee") != args.assignee:
            return False
    
    return True


def print_todo_list(todos):
    """打印 todo 列表"""
    if not todos:
        print("No todos found")
        return
    
    print(f"Found {len(todos)} todos:\n")
    for i, todo in enumerate(todos, 1):
        todo_id = todo.get("id", "N/A")
        title = todo.get("title", "N/A")
        status = todo.get("status", "N/A")
        priority = todo.get("priority", "N/A")
        assignee = todo.get("assignee", "N/A")
        print(f"{i}. [{todo_id}] {title}")
        print(f"   Status: {status} | Priority: {priority} | Assignee: {assignee}")
        print()


def print_todo_detail(todo):
    """打印 todo 详情"""
    print(f"{'='*80}")
    print(f"Todo ID: {todo.get('id', 'N/A')}")
    print(f"Title: {todo.get('title', 'N/A')}")
    print(f"Status: {todo.get('status', 'N/A')}")
    print(f"Priority: {todo.get('priority', 'N/A')}")
    print(f"Assignee: {todo.get('assignee', 'N/A')}")
    print(f"Created at: {todo.get('created_at', 'N/A')}")
    print(f"Started at: {todo.get('started_at', 'N/A')}")
    print(f"Completed at: {todo.get('completed_at', 'N/A')}")
    print(f"Archived at: {todo.get('archived_at', 'N/A')}")
    print(f"{'='*80}")
    print("\nProgress:")
    print(todo.get("progress", "N/A"))
    print(f"\n{'='*80}")
    
    if todo.get("links"):
        print("\nLinks:")
        for link in todo.get("links", []):
            print(f"  - {link}")
    
    if todo.get("definition_of_done"):
        print("\nDefinition of Done:")
        for item in todo.get("definition_of_done", []):
            print(f"  - {item}")
    
    if todo.get("user_requirements"):
        print("\nUser Requirements:")
        for req in todo.get("user_requirements", []):
            print(f"  - {req}")
    
    print(f"\n{'='*80}")


def main():
    parser = argparse.ArgumentParser(description="Todo Finder - 从 todos.json 和归档中以多种匹配方式查找 todo")
    
    # 匹配方式
    parser.add_argument("--id", help="按 todo ID 精确匹配")
    parser.add_argument("--title", help="按标题关键词匹配")
    parser.add_argument("--keyword", help="按关键词匹配（标题、内容、链接等）")
    parser.add_argument("--status", help="按状态匹配（pending/in-progress/completed）")
    parser.add_argument("--priority", help="按优先级匹配（P0-P9）")
    parser.add_argument("--assignee", help="按负责人匹配")
    
    # 范围选项
    parser.add_argument("--archived", action="store_true", help="只查找归档中的 todo")
    parser.add_argument("--all", action="store_true", help="查找所有 todo（包括归档）")
    
    # 输出选项
    parser.add_argument("--detail", action="store_true", help="显示 todo 详情")
    
    args = parser.parse_args()
    
    # 加载 todos
    all_todos = []
    
    if not args.archived:
        todos_data = load_todos_json()
        if todos_data and "todos" in todos_data:
            all_todos.extend(todos_data["todos"])
    
    if args.archived or args.all:
        archive_todos = load_archive_todos()
        all_todos.extend(archive_todos)
    
    # 匹配 todos
    matched_todos = []
    for todo in all_todos:
        if match_todo(todo, args):
            matched_todos.append(todo)
    
    # 输出结果
    if args.detail and len(matched_todos) == 1:
        print_todo_detail(matched_todos[0])
    else:
        print_todo_list(matched_todos)


if __name__ == "__main__":
    main()
