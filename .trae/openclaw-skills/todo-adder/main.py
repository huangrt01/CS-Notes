
#!/usr/bin/env python3
"""
Todo Adder Skill - 使用Todos Web Manager的API来添加todo
避免直接编辑todos.json导致的语法错误
"""

import sys
import json
import requests
from pathlib import Path
from datetime import datetime

# 配置
WEB_MANAGER_URL = "http://localhost:5000"
REPO_ROOT = Path(__file__).parent.parent.parent.parent


def add_todo_via_api(todo_data):
    """通过Todos Web Manager的API添加todo"""
    try:
        response = requests.post(
            f"{WEB_MANAGER_URL}/api/tasks",
            json=todo_data,
            timeout=10
        )
        response.raise_for_status()
        result = response.json()
        
        if result.get("success"):
            print(f"✅ Todo添加成功！")
            print(f"   ID: {result['task']['id']}")
            print(f"   标题: {result['task']['title']}")
            return True
        else:
            print(f"❌ Todo添加失败: {result.get('message', '未知错误')}")
            return False
            
    except requests.exceptions.ConnectionError:
        print(f"❌ 无法连接到Todos Web Manager ({WEB_MANAGER_URL})")
        print(f"   请确保server.py正在运行！")
        print(f"   启动命令: cd {REPO_ROOT}/.trae/web-manager && python3 server.py")
        return False
    except Exception as e:
        print(f"❌ Todo添加失败: {e}")
        return False


def main():
    print("=" * 80)
    print("🤖 Todo Adder Skill")
    print("=" * 80)
    
    # 检查参数
    if len(sys.argv) < 2:
        print("\n使用方法:")
        print("  python3 main.py <title> [priority] [assignee]")
        print("\n示例:")
        print("  python3 main.py '测试todo' P2 ai")
        print("\n或者从stdin读取JSON:")
        print("  cat todo.json | python3 main.py")
        return
    
    # 尝试从stdin读取JSON
    if not sys.stdin.isatty():
        try:
            todo_data = json.load(sys.stdin)
            print(f"\n📋 从stdin读取到todo数据")
            add_todo_via_api(todo_data)
            return
        except Exception as e:
            print(f"⚠️ 从stdin读取JSON失败: {e}")
    
    # 从命令行参数创建todo
    title = sys.argv[1]
    priority = sys.argv[2] if len(sys.argv) > 2 else "P2"
    assignee = sys.argv[3] if len(sys.argv) > 3 else "ai"
    
    # 收集links（第4个参数及之后的都是links）
    links = sys.argv[4:] if len(sys.argv) > 4 else []
    
    todo_data = {
        "title": title,
        "status": "pending",
        "priority": priority,
        "assignee": assignee,
        "feedback_required": False,
        "created_at": datetime.now().isoformat(),
        "links": links,
        "definition_of_done": [],
        "user_requirements": [title],
        "progress": "",
        "started_at": "",
        "completed_at": "",
        "commit_hash": ""
    }
    
    print(f"\n📋 创建todo:")
    print(f"   标题: {title}")
    print(f"   优先级: {priority}")
    print(f"   负责人: {assignee}")
    
    add_todo_via_api(todo_data)


if __name__ == "__main__":
    main()

