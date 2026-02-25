#!/usr/bin/env python3
"""
Todo Updater Skill - 使用Todos Web Manager的API来更新todo状态
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


def update_task_via_api(task_id, update_data):
    """通过Todos Web Manager的API更新任务"""
    try:
        response = requests.put(
            f"{WEB_MANAGER_URL}/api/tasks/{task_id}",
            json=update_data,
            timeout=10
        )
        response.raise_for_status()
        result = response.json()
        
        if result.get("success"):
            print(f"✅ Todo更新成功！")
            print(f"   ID: {task_id}")
            return True
        else:
            print(f"❌ Todo更新失败: {result.get('message', '未知错误')}")
            return False
            
    except requests.exceptions.ConnectionError:
        print(f"❌ 无法连接到Todos Web Manager ({WEB_MANAGER_URL})")
        print(f"   请确保server.py正在运行！")
        print(f"   启动命令: cd {REPO_ROOT}/.trae/web-manager && python3 server.py")
        return False
    except Exception as e:
        print(f"❌ Todo更新失败: {e}")
        return False


def update_task_status_via_api(task_id, new_status, progress=None, commit_hash_before=None, commit_hash_after=None):
    """通过Todos Web Manager的API更新任务状态"""
    try:
        data = {"status": new_status}
        if progress:
            data["progress"] = progress
        if commit_hash_before:
            data["commit_hash_before"] = commit_hash_before
        if commit_hash_after:
            data["commit_hash_after"] = commit_hash_after
        
        response = requests.put(
            f"{WEB_MANAGER_URL}/api/tasks/{task_id}/status",
            json=data,
            timeout=10
        )
        response.raise_for_status()
        result = response.json()
        
        if result.get("success"):
            print(f"✅ Todo状态更新成功！")
            print(f"   ID: {task_id}")
            print(f"   新状态: {new_status}")
            return True
        else:
            print(f"❌ Todo状态更新失败: {result.get('message', '未知错误')}")
            return False
            
    except requests.exceptions.ConnectionError:
        print(f"❌ 无法连接到Todos Web Manager ({WEB_MANAGER_URL})")
        print(f"   请确保server.py正在运行！")
        print(f"   启动命令: cd {REPO_ROOT}/.trae/web-manager && python3 server.py")
        return False
    except Exception as e:
        print(f"❌ Todo状态更新失败: {e}")
        return False


def main():
    print("=" * 80)
    print("🤖 Todo Updater Skill")
    print("=" * 80)
    
    # 检查参数
    if len(sys.argv) < 3:
        print("\n使用方法:")
        print("  python3 main.py <task_id> <new_status> [progress] [commit_hash_before] [commit_hash_after]")
        print("\n示例:")
        print("  python3 main.py todo-20260225-008 completed \"✅ 已完成！\"")
        print("  python3 main.py todo-20260225-008 in-progress")
        print("  python3 main.py todo-20260225-008 completed \"✅ 已完成！\" <commit_hash_before> <commit_hash_after>")
        print("\n或者从stdin读取JSON:")
        print("  cat update.json | python3 main.py <task_id>")
        return
    
    # 尝试从stdin读取JSON
    if not sys.stdin.isatty():
        try:
            update_data = json.load(sys.stdin)
            print(f"\n📋 从stdin读取到更新数据")
            task_id = sys.argv[1]
            update_task_via_api(task_id, update_data)
            return
        except Exception as e:
            print(f"⚠️ 从stdin读取JSON失败: {e}")
    
    # 从命令行参数更新状态
    task_id = sys.argv[1]
    new_status = sys.argv[2]
    progress = sys.argv[3] if len(sys.argv) > 3 else None
    commit_hash_before = sys.argv[4] if len(sys.argv) > 4 else None
    commit_hash_after = sys.argv[5] if len(sys.argv) > 5 else None
    
    print(f"\n📋 更新todo状态:")
    print(f"   ID: {task_id}")
    print(f"   新状态: {new_status}")
    if progress:
        print(f"   进度: {progress[:100]}...")
    if commit_hash_before:
        print(f"   Commit Hash Before: {commit_hash_before}")
    if commit_hash_after:
        print(f"   Commit Hash After: {commit_hash_after}")
    
    update_task_status_via_api(task_id, new_status, progress, commit_hash_before, commit_hash_after)


if __name__ == "__main__":
    main()
