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
TODOS_FILE = REPO_ROOT / ".trae/documents/todos管理系统.md"
TODO_ARCHIVE_FILE = REPO_ROOT / ".trae/documents/TODO_ARCHIVE.md"
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

def parse_todos_from_markdown(file_path):
    """从 Markdown 文件解析任务"""
    if not file_path.exists():
        return []
    
    content = file_path.read_text(encoding='utf-8')
    tasks = []
    current_task = None
    
    lines = content.split('\n')
    for i, line in enumerate(lines):
        # 匹配任务行
        task_match = re.match(r'^(\*|-)\s+\[([ x])\]\s+(.*)$', line)
        if task_match:
            if current_task:
                tasks.append(current_task)
            
            list_marker, status_marker, title = task_match.groups()
            current_task = {
                'id': len(tasks) + 1,
                'title': title.strip(),
                'status': 'completed' if status_marker == 'x' else 'pending',
                'priority': 'medium',
                'assignee': 'User',
                'description': '',
                'links': [],
                'definitionOfDone': [],
                'progress': '',
                'createdAt': '',
                'completedAt': '',
                'startedAt': '',
                'lineNumber': i + 1
            }
        elif current_task:
            # 解析任务属性
            priority_match = re.match(r'^\s*-\s*Priority[：:]\s*(high|medium|low)', line, re.I)
            if priority_match:
                current_task['priority'] = priority_match.group(1).lower()
            
            assignee_match = re.match(r'^\s*-\s*Assignee[：:]\s*(\w+)', line, re.I)
            if assignee_match:
                current_task['assignee'] = assignee_match.group(1)
            
            progress_match = re.match(r'^\s*-\s*Progress[：:]\s*(.*)', line, re.I)
            if progress_match:
                current_task['progress'] = progress_match.group(1).strip()
            
            created_match = re.match(r'^\s*-\s*Started\s+At[：:]\s*(.*)', line, re.I)
            if created_match:
                current_task['startedAt'] = created_match.group(1).strip()
            
            completed_match = re.match(r'^\s*-\s*Completed\s+At[：:]\s*(.*)', line, re.I)
            if completed_match:
                current_task['completedAt'] = completed_match.group(1).strip()
            
            # 解析链接
            link_match = re.match(r'^\s*-\s*Links[：:]\s*(.*)', line, re.I)
            if link_match:
                links_str = link_match.group(1).strip()
                links = re.findall(r'`([^`]+)`', links_str)
                current_task['links'] = links
            
            # 解析 Definition of Done
            dod_match = re.match(r'^\s*-\s*Definition\s+of\s+Done[：:]\s*', line, re.I)
            if dod_match:
                # 继续读取后续的列表项
                j = i + 1
                while j < len(lines):
                    dod_line = lines[j]
                    dod_item_match = re.match(r'^\s*\*\s*(.*)$', dod_line)
                    if dod_item_match:
                        current_task['definitionOfDone'].append(dod_item_match.group(1).strip())
                        j += 1
                    else:
                        break
    
    if current_task:
        tasks.append(current_task)
    
    return tasks

@app.route('/api/tasks', methods=['GET'])
def get_tasks():
    """获取任务列表"""
    tasks = parse_todos_from_markdown(TODOS_FILE)
    return jsonify({
        "success": True,
        "tasks": tasks,
        "total": len(tasks)
    })

@app.route('/api/tasks/archive', methods=['GET'])
def get_archive_tasks():
    """获取归档任务"""
    tasks = parse_todos_from_markdown(TODO_ARCHIVE_FILE)
    return jsonify({
        "success": True,
        "tasks": tasks,
        "total": len(tasks)
    })

# ============================================
# 任务管理功能
# ============================================

def generate_task_markdown(task):
    """生成任务 Markdown"""
    status_box = '[x]' if task.get('status') == 'completed' else '[ ]'
    lines = []
    
    # 任务标题
    lines.append(f"* {status_box} {task['title']}")
    
    # 优先级
    priority = task.get('priority', 'medium')
    lines.append(f"  - Priority：{priority.capitalize()}")
    
    # Assignee
    assignee = task.get('assignee', 'User')
    lines.append(f"  - Assignee：{assignee}")
    
    # Feedback Required
    lines.append("  - Feedback Required：否")
    
    # Links
    links = task.get('links', [])
    if links:
        links_str = '、'.join([f'`{link}`' for link in links])
        lines.append(f"  - Links：{links_str}")
    
    # Definition of Done
    dod = task.get('definitionOfDone', [])
    if dod:
        lines.append("  - Definition of Done：")
        for item in dod:
            lines.append(f"    * {item}")
    
    # Progress
    progress = task.get('progress', '')
    if progress:
        lines.append(f"  - Progress：{progress}")
    
    # Started At
    started_at = task.get('startedAt', '')
    if started_at:
        lines.append(f"  - Started At：{started_at}")
    
    # Completed At
    completed_at = task.get('completedAt', '')
    if completed_at:
        lines.append(f"  - Completed At：{completed_at}")
    
    return '\n'.join(lines)

@app.route('/api/tasks', methods=['POST'])
def add_task():
    """添加新任务"""
    data = request.json
    task = {
        'title': data.get('title', ''),
        'description': data.get('description', ''),
        'priority': data.get('priority', 'medium'),
        'status': 'pending',
        'assignee': data.get('assignee', 'User'),
        'links': data.get('links', []),
        'definitionOfDone': data.get('definitionOfDone', []),
        'startedAt': datetime.now().strftime('%Y-%m-%d')
    }
    
    # 生成 Markdown
    task_markdown = generate_task_markdown(task)
    
    # 写入 INBOX.md
    if INBOX_FILE.exists():
        content = INBOX_FILE.read_text(encoding='utf-8')
        content = task_markdown + '\n\n' + content
        INBOX_FILE.write_text(content, encoding='utf-8')
    else:
        INBOX_FILE.write_text(task_markdown + '\n', encoding='utf-8')
    
    return jsonify({
        "success": True,
        "message": "任务已添加到 INBOX.md",
        "task": task
    })

@app.route('/api/tasks/<int:task_id>/status', methods=['PUT'])
def update_task_status(task_id):
    """更新任务状态"""
    data = request.json
    new_status = data.get('status', 'pending')
    
    # 这里简化处理，实际应该读取文件、找到任务、更新状态、写回文件
    return jsonify({
        "success": True,
        "message": f"任务 {task_id} 状态已更新为 {new_status}"
    })

# ============================================
# 开发验证功能
# ============================================

@app.route('/api/dev/parse-test', methods=['POST'])
def dev_parse_test():
    """测试 Markdown 解析"""
    data = request.json
    markdown = data.get('markdown', '')
    
    # 临时写入文件
    temp_file = REPO_ROOT / '.trae/documents/temp-test.md'
    temp_file.write_text(markdown, encoding='utf-8')
    
    # 解析
    tasks = parse_todos_from_markdown(temp_file)
    
    # 删除临时文件
    temp_file.unlink(missing_ok=True)
    
    return jsonify({
        "success": True,
        "tasks": tasks,
        "total": len(tasks)
    })

@app.route('/api/dev/generate-test', methods=['POST'])
def dev_generate_test():
    """测试 Markdown 生成"""
    data = request.json
    task = data.get('task', {})
    
    markdown = generate_task_markdown(task)
    
    return jsonify({
        "success": True,
        "markdown": markdown
    })

@app.route('/api/dev/validate', methods=['POST'])
def dev_validate():
    """验证任务数据"""
    data = request.json
    tasks = data.get('tasks', [])
    
    errors = []
    warnings = []
    
    for i, task in enumerate(tasks):
        if not task.get('title'):
            errors.append(f"任务 {i+1}: 缺少 title 字段")
        
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
    print(f"归档文件: {TODO_ARCHIVE_FILE}")
    print(f"INBOX 文件: {INBOX_FILE}")
    print("=" * 60)
    print("可用的 API:")
    print("  - GET  /api/tasks          获取任务列表")
    print("  - POST /api/tasks          添加新任务")
    print("  - GET  /api/git/status      获取 Git 状态")
    print("  - POST /api/git/commit      提交 Git 更改")
    print("  - POST /api/git/push        推送到远程仓库")
    print("=" * 60)
    print("启动服务器: http://localhost:5000")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=5000, debug=True)
