#!/usr/bin/env python3
"""
Todos Web Manager - 后端服务（开发验证专用）
支持 Git 集成、文件读写、任务解析等功能
"""

import os
import sys
import json
import re
import subprocess
from pathlib import Path
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# 配置路径
REPO_ROOT = Path(__file__).parent.parent.parent
TODOS_FILE = REPO_ROOT / ".trae/todos/todos.json"
TODO_ARCHIVE_DIR = REPO_ROOT / ".trae/todos/archive"
INBOX_FILE = REPO_ROOT / ".trae/documents/INBOX.md"
WEB_MANAGER_DIR = Path(__file__).parent

app = Flask(__name__, static_folder='.')
CORS(app)

# ============================================
# Git 集成功能
# ============================================

def run_git_command(cmd, cwd=None):
    """执行 Git 命令"""
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd or REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=30
        )
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@app.route('/api/git/status', methods=['GET'])
def git_status():
    """获取 Git 状态"""
    result = run_git_command(['git', 'status'])
    return jsonify(result)

@app.route('/api/git/add', methods=['POST'])
def git_add():
    """添加文件到 Git"""
    data = request.json
    files = data.get('files', ['.'])
    result = run_git_command(['git', 'add'] + files)
    return jsonify(result)

@app.route('/api/git/commit', methods=['POST'])
def git_commit():
    """提交 Git 更改"""
    data = request.json
    message = data.get('message', 'Update todos')
    result = run_git_command(['git', 'commit', '-m', message])
    return jsonify(result)

@app.route('/api/git/push', methods=['POST'])
def git_push():
    """推送到远程仓库"""
    result = run_git_command(['git', 'push'])
    return jsonify(result)

@app.route('/api/git/pull', methods=['POST'])
def git_pull():
    """从远程仓库拉取"""
    result = run_git_command(['git', 'pull'])
    return jsonify(result)

@app.route('/api/git/log', methods=['GET'])
def git_log():
    """获取 Git 日志"""
    limit = request.args.get('limit', 10)
    result = run_git_command(['git', 'log', f'-{limit}', '--oneline'])
    return jsonify(result)

# ============================================
# 任务解析功能
# ============================================

def load_todos_from_json(file_path):
    """从 JSON 文件加载任务"""
    if not file_path.exists():
        return {
            "version": "1.0.0",
            "updated_at": datetime.now().isoformat(),
            "todos": []
        }
    
    try:
        content = file_path.read_text(encoding='utf-8')
        return json.loads(content)
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return {
            "version": "1.0.0",
            "updated_at": datetime.now().isoformat(),
            "todos": []
        }

def save_todos_to_json(data, file_path):
    """保存任务到 JSON 文件"""
    try:
        data["updated_at"] = datetime.now().isoformat()
        content = json.dumps(data, ensure_ascii=False, indent=2)
        file_path.write_text(content, encoding='utf-8')
        return True
    except Exception as e:
        print(f"Error saving JSON file: {e}")
        return False

@app.route('/api/tasks', methods=['GET'])
def get_tasks():
    """获取任务列表"""
    data = load_todos_from_json(TODOS_FILE)
    return jsonify({
        "success": True,
        "data": data,
        "tasks": data.get("todos", []),
        "total": len(data.get("todos", []))
    })

@app.route('/api/tasks/archive', methods=['GET'])
def get_archive_tasks():
    """获取归档任务"""
    # 读取所有归档文件
    archive_tasks = []
    if TODO_ARCHIVE_DIR.exists():
        for archive_file in TODO_ARCHIVE_DIR.glob("*.json"):
            data = load_todos_from_json(archive_file)
            archive_tasks.extend(data.get("todos", []))
    
    return jsonify({
        "success": True,
        "tasks": archive_tasks,
        "total": len(archive_tasks)
    })

# ============================================
# 任务管理功能
# ============================================

def generate_task_id():
    """生成任务 ID"""
    today = datetime.now().strftime('%Y%m%d')
    data = load_todos_from_json(TODOS_FILE)
    existing_ids = [t.get('id', '') for t in data.get('todos', [])]
    
    # 找到今天最大的序号
    max_seq = 0
    for task_id in existing_ids:
        if task_id.startswith(f'todo-{today}-'):
            try:
                seq = int(task_id.split('-')[-1])
                max_seq = max(max_seq, seq)
            except ValueError:
                pass
    
    return f'todo-{today}-{max_seq + 1:03d}'

@app.route('/api/tasks', methods=['POST'])
def add_task():
    """添加新任务"""
    data = request.json
    
    # 创建新任务
    new_task = {
        'id': data.get('id', generate_task_id()),
        'title': data.get('title', ''),
        'status': data.get('status', 'pending'),
        'priority': data.get('priority', 'medium'),
        'assignee': data.get('assignee', 'user'),
        'feedback_required': data.get('feedback_required', False),
        'created_at': data.get('created_at', datetime.now().isoformat()),
        'links': data.get('links', []),
        'definition_of_done': data.get('definition_of_done', []),
        'progress': data.get('progress', ''),
        'started_at': data.get('started_at', ''),
        'completed_at': data.get('completed_at', '')
    }
    
    # 加载现有数据
    todos_data = load_todos_from_json(TODOS_FILE)
    todos_data['todos'].append(new_task)
    
    # 保存
    if save_todos_to_json(todos_data, TODOS_FILE):
        return jsonify({
            "success": True,
            "message": "任务已添加",
            "task": new_task
        })
    else:
        return jsonify({
            "success": False,
            "message": "保存任务失败"
        }), 500

@app.route('/api/tasks/<task_id>', methods=['PUT'])
def update_task(task_id):
    """更新任务"""
    data = request.json
    
    # 加载现有数据
    todos_data = load_todos_from_json(TODOS_FILE)
    tasks = todos_data.get('todos', [])
    
    # 找到任务
    task_found = False
    for i, task in enumerate(tasks):
        if task.get('id') == task_id:
            # 更新任务
            tasks[i].update(data)
            task_found = True
            break
    
    if not task_found:
        return jsonify({
            "success": False,
            "message": f"任务 {task_id} 不存在"
        }), 404
    
    # 保存
    if save_todos_to_json(todos_data, TODOS_FILE):
        return jsonify({
            "success": True,
            "message": f"任务 {task_id} 已更新"
        })
    else:
        return jsonify({
            "success": False,
            "message": "保存任务失败"
        }), 500

@app.route('/api/tasks/<task_id>/status', methods=['PUT'])
def update_task_status(task_id):
    """更新任务状态"""
    data = request.json
    new_status = data.get('status', 'pending')
    
    # 加载现有数据
    todos_data = load_todos_from_json(TODOS_FILE)
    tasks = todos_data.get('todos', [])
    
    # 找到任务
    task_found = False
    for i, task in enumerate(tasks):
        if task.get('id') == task_id:
            tasks[i]['status'] = new_status
            
            # 如果完成，设置完成时间
            if new_status == 'completed' and not tasks[i].get('completed_at'):
                tasks[i]['completed_at'] = datetime.now().isoformat()
            
            # 如果开始，设置开始时间
            if new_status == 'in-progress' and not tasks[i].get('started_at'):
                tasks[i]['started_at'] = datetime.now().isoformat()
            
            task_found = True
            break
    
    if not task_found:
        return jsonify({
            "success": False,
            "message": f"任务 {task_id} 不存在"
        }), 404
    
    # 保存
    if save_todos_to_json(todos_data, TODOS_FILE):
        return jsonify({
            "success": True,
            "message": f"任务 {task_id} 状态已更新为 {new_status}"
        })
    else:
        return jsonify({
            "success": False,
            "message": "保存任务失败"
        }), 500

@app.route('/api/tasks/<task_id>', methods=['DELETE'])
def delete_task(task_id):
    """删除任务"""
    # 加载现有数据
    todos_data = load_todos_from_json(TODOS_FILE)
    tasks = todos_data.get('todos', [])
    
    # 找到并删除任务
    original_len = len(tasks)
    todos_data['todos'] = [t for t in tasks if t.get('id') != task_id]
    
    if len(todos_data['todos']) == original_len:
        return jsonify({
            "success": False,
            "message": f"任务 {task_id} 不存在"
        }), 404
    
    # 保存
    if save_todos_to_json(todos_data, TODOS_FILE):
        return jsonify({
            "success": True,
            "message": f"任务 {task_id} 已删除"
        })
    else:
        return jsonify({
            "success": False,
            "message": "保存任务失败"
        }), 500

# ============================================
# 开发验证功能
# ============================================

@app.route('/api/dev/validate', methods=['POST'])
def dev_validate():
    """验证任务数据"""
    data = request.json
    tasks = data.get('tasks', [])
    
    errors = []
    warnings = []
    
    for i, task in enumerate(tasks):
        if not task.get('id'):
            errors.append(f"任务 {i+1}: 缺少 id 字段")
        
        if not task.get('title'):
            errors.append(f"任务 {i+1}: 缺少 title 字段")
        
        if not task.get('status'):
            errors.append(f"任务 {i+1}: 缺少 status 字段")
        
        priority = task.get('priority')
        if priority and priority not in ['high', 'medium', 'low']:
            errors.append(f"任务 {i+1}: 无效的 priority 值: {priority}")
        
        status = task.get('status')
        if status and status not in ['pending', 'in-progress', 'completed']:
            errors.append(f"任务 {i+1}: 无效的 status 值: {status}")
    
    return jsonify({
        "success": True,
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "total": len(tasks)
    })

# ============================================
# 静态文件服务
# ============================================

@app.route('/')
def index():
    """主页 - 重定向到增强版"""
    return send_from_directory('.', 'index-enhanced.html')

@app.route('/<path:path>')
def static_files(path):
    """静态文件服务"""
    return send_from_directory('.', path)

# ============================================
# 主函数
# ============================================

if __name__ == '__main__':
    print("=" * 60)
    print("Todos Web Manager - 后端服务（开发验证专用）")
    print("=" * 60)
    print(f"仓库根目录: {REPO_ROOT}")
    print(f"任务文件: {TODOS_FILE}")
    print(f"归档目录: {TODO_ARCHIVE_DIR}")
    print(f"INBOX 文件: {INBOX_FILE}")
    print("=" * 60)
    print("可用的 API:")
    print("  - GET    /api/tasks              获取任务列表")
    print("  - POST   /api/tasks              添加新任务")
    print("  - PUT    /api/tasks/<id>         更新任务")
    print("  - DELETE /api/tasks/<id>         删除任务")
    print("  - PUT    /api/tasks/<id>/status  更新任务状态")
    print("  - GET    /api/git/status          获取 Git 状态")
    print("  - POST   /api/git/commit          提交 Git 更改")
    print("  - POST   /api/git/push            推送到远程仓库")
    print("=" * 60)
    print("启动服务器: http://localhost:5000")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=5000, debug=True)
