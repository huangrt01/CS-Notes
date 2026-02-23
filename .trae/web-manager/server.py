#!/usr/bin/env python3
"""
Todos Web Manager - 后端服务
支持 Git 集成、文件读写、任务解析等功能

=======================================================================
使用说明
=======================================================================

1. 安装依赖（首次使用）：
   cd /Users/bytedance/CS-Notes/.trae/web-manager
   pip3 install flask flask-cors

2. 启动后端服务器：
   cd /Users/bytedance/CS-Notes/.trae/web-manager
   python3 server.py

3. 在浏览器中访问：
   http://localhost:5000

=======================================================================
本次运行的有效指令记录：
=======================================================================

安装依赖：
pip3 install flask flask-cors

启动服务器：
python3 server.py

=======================================================================
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

# ============================================
# 配置文件加载
# ============================================

CONFIG_FILE = Path(__file__).parent / "config.json"

def load_config():
    """加载配置文件"""
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading config file: {e}")
            return get_default_config()
    return get_default_config()

def get_default_config():
    """获取默认配置"""
    return {
        "project": {
            "name": "Project",
            "title": "Todos Web Manager"
        },
        "paths": {
            "repo_root": "../..",
            "todos_file": ".trae/todos/todos.json",
            "todo_archive_dir": ".trae/todos/archive",
            "plans_dir": ".trae/plans",
            "inbox_file": ".trae/documents/INBOX.md"
        }
    }

config = load_config()

# 配置路径
REPO_ROOT = Path(__file__).parent.parent.parent
TODOS_FILE = REPO_ROOT / config['paths']['todos_file']
TODO_ARCHIVE_DIR = REPO_ROOT / config['paths']['todo_archive_dir']
PLANS_DIR = REPO_ROOT / config['paths']['plans_dir']
INBOX_FILE = REPO_ROOT / config['paths']['inbox_file']
WEB_MANAGER_DIR = Path(__file__).parent

app = Flask(__name__, static_folder='.')
CORS(app)

# 初始化任务执行日志系统
task_logger = None
if TASK_LOGGER_AVAILABLE:
    try:
        task_logger = create_logger(REPO_ROOT)
        print("✅ 任务执行日志系统已初始化")
    except Exception as e:
        print(f"⚠️ 任务执行日志系统初始化失败: {e}")
        task_logger = None

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

@app.route('/api/git/diff', methods=['GET'])
def git_diff():
    """获取 Git diff"""
    commit = request.args.get('commit', 'HEAD~1')
    commit2 = request.args.get('commit2', 'HEAD')
    file = request.args.get('file', None)
    
    cmd = ['git', 'diff', commit, commit2]
    if file:
        cmd.append(file)
    
    result = run_git_command(cmd)
    return jsonify(result)

@app.route('/api/git/diff/<commit>', methods=['GET'])
def git_diff_commit(commit):
    """获取特定 commit 的 diff"""
    result = run_git_command(['git', 'show', commit])
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
    todos = data.get("todos", [])
    
    # 加载所有独立 Plan 文件
    independent_plans = load_all_plans()
    
    # 建立关联：将独立 Plan 文件与 Todo 条目关联
    for todo in todos:
        todo_id = todo.get("id")
        # 查找对应的独立 Plan 文件
        for plan in independent_plans:
            # 通过标题匹配或 todo_id 匹配
            plan_todo_id = plan.get("todo_id")
            if plan_todo_id == todo_id:
                todo["plan_file"] = plan["file_path"]
                if "plan" not in todo:
                    todo["plan"] = {
                        "content": plan["content"],
                        "status": plan["status"],
                        "created_at": plan["created_at"],
                        "updated_at": plan["updated_at"]
                    }
                break
            # 如果没有 todo_id，尝试通过标题匹配
            if not plan_todo_id and plan.get("title") and todo.get("title"):
                if plan["title"] in todo["title"] or todo["title"] in plan["title"]:
                    todo["plan_file"] = plan["file_path"]
                    if "plan" not in todo:
                        todo["plan"] = {
                            "content": plan["content"],
                            "status": plan["status"],
                            "created_at": plan["created_at"],
                            "updated_at": plan["updated_at"]
                        }
                    break
    
    return jsonify({
        "success": True,
        "data": data,
        "tasks": todos,
        "independent_plans": independent_plans,
        "total": len(todos)
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
# Plan 管理功能
# ============================================

def load_plan_from_file(file_path):
    """从 Markdown 文件加载 Plan（解析 YAML frontmatter）"""
    if not file_path.exists():
        return None
    
    try:
        content = file_path.read_text(encoding='utf-8')
        
        # 解析 YAML frontmatter
        frontmatter = {}
        lines = content.split('\n')
        if lines and lines[0] == '---':
            # 找到第二个 ---
            end_idx = None
            for i in range(1, len(lines)):
                if lines[i] == '---':
                    end_idx = i
                    break
            
            if end_idx:
                # 解析 frontmatter
                for line in lines[1:end_idx]:
                    if ':' in line:
                        key, value = line.split(':', 1)
                        frontmatter[key.strip()] = value.strip()
        
        # 提取计划内容（frontmatter 之后的部分）
        plan_content = '\n'.join(lines[end_idx+2:]) if end_idx else content
        
        return {
            "id": frontmatter.get('id', ''),
            "title": frontmatter.get('title', '').strip('"'),
            "priority": frontmatter.get('priority', 'medium'),
            "status": frontmatter.get('status', 'pending'),
            "created_at": frontmatter.get('created_at', ''),
            "updated_at": frontmatter.get('updated_at', ''),
            "tags": frontmatter.get('tags', []),
            "file_path": str(file_path),
            "content": plan_content
        }
    except Exception as e:
        print(f"Error loading plan file: {e}")
        return None

def load_all_plans():
    """加载所有 Plan"""
    plans = []
    if PLANS_DIR.exists():
        for plan_file in PLANS_DIR.glob("*.md"):
            # 跳过设计方案文件
            if plan_file.name.startswith("Plan-Mode-"):
                continue
            
            plan = load_plan_from_file(plan_file)
            if plan:
                plans.append(plan)
    
    # 按创建时间倒序排列
    plans.sort(key=lambda x: x.get('created_at', ''), reverse=True)
    return plans

@app.route('/api/plans', methods=['GET'])
def get_plans():
    """获取 Plan 列表"""
    plans = load_all_plans()
    return jsonify({
        "success": True,
        "plans": plans,
        "total": len(plans)
    })

@app.route('/api/plans/<plan_id>/status', methods=['PUT'])
def update_plan_status(plan_id):
    """更新 Plan 状态（approve/reject）"""
    data = request.json
    new_status = data.get('status', 'pending')
    comment = data.get('comment', '')
    
    # 找到对应的 plan 文件
    plan_file = None
    for f in PLANS_DIR.glob("*.md"):
        plan = load_plan_from_file(f)
        if plan and plan.get('id') == plan_id:
            plan_file = f
            break
    
    if not plan_file:
        return jsonify({
            "success": False,
            "message": f"Plan {plan_id} 不存在"
        }), 404
    
    # 更新 plan 文件
    try:
        content = plan_file.read_text(encoding='utf-8')
        lines = content.split('\n')
        
        # 更新 frontmatter 中的 status
        if lines and lines[0] == '---':
            for i in range(1, len(lines)):
                if lines[i] == '---':
                    break
                if lines[i].startswith('status:'):
                    lines[i] = f"status: {new_status}"
                if lines[i].startswith('updated_at:'):
                    lines[i] = f"updated_at: '{datetime.now().isoformat()}'"
        
        # 添加 review 记录
        if comment:
            review_note = f"\n\n## Review 记录\n- {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {new_status}\n- 评论: {comment}\n"
            lines.append(review_note)
        
        new_content = '\n'.join(lines)
        plan_file.write_text(new_content, encoding='utf-8')
        
        return jsonify({
            "success": True,
            "message": f"Plan {plan_id} 状态已更新为 {new_status}"
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"更新 Plan 失败: {e}"
        }), 500

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
        'completed_at': data.get('completed_at', ''),
        'commit_hash': data.get('commit_hash', '')
    }
    
    # 记录任务创建日志
    if TASK_LOGGER_AVAILABLE and task_logger:
        try:
            task_logger.log_info(
                new_task['id'],
                TaskStage.PENDING,
                "任务已创建",
                {"title": new_task['title'], "priority": new_task['priority']}
            )
        except Exception as e:
            print(f"⚠️ 记录任务创建日志失败: {e}")
    
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
            old_status = tasks[i].get('status')
            tasks[i]['status'] = new_status
            
            # 记录任务状态变更日志
            if TASK_LOGGER_AVAILABLE and task_logger:
                try:
                    agent = 'trae'  # 默认使用 trae
                    if new_status == 'in-progress' and old_status != 'in-progress':
                        task_logger.start_task(task_id, agent=agent)
                        task_logger.log_info(
                            task_id,
                            TaskStage.PLANNING,
                            "任务开始执行",
                            {"old_status": old_status},
                            agent=agent
                        )
                    elif new_status == 'completed' and old_status != 'completed':
                        tasks[i]['completed_at'] = datetime.now().isoformat()
                        # 自动获取当前的git commit hash
                        commit_result = run_git_command(['git', 'rev-parse', 'HEAD'])
                        if commit_result.get('success'):
                            tasks[i]['commit_hash'] = commit_result.get('stdout', '').strip()
                        task_logger.complete_task(task_id, agent=agent)
                        task_logger.log_success(
                            task_id,
                            TaskStage.COMPLETED,
                            "任务完成",
                            {"commit_hash": tasks[i].get('commit_hash', '')},
                            agent=agent
                        )
                    elif new_status == 'pending' and old_status != 'pending':
                        task_logger.log_info(
                            task_id,
                            TaskStage.PENDING,
                            "任务回到待办",
                            {"old_status": old_status},
                            agent=agent
                        )
                except Exception as e:
                    print(f"⚠️ 记录任务状态变更日志失败: {e}")
            
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

@app.route('/api/tasks/<task_id>/plan-review', methods=['POST'])
def review_plan(task_id):
    """Review Plan（通过或不通过）"""
    data = request.json
    approved = data.get('approved', False)
    review_comment = data.get('comment', '')
    
    # 加载现有数据
    todos_data = load_todos_from_json(TODOS_FILE)
    tasks = todos_data.get('todos', [])
    
    # 找到任务
    task_found = False
    for i, task in enumerate(tasks):
        if task.get('id') == task_id:
            if 'plan' not in tasks[i]:
                return jsonify({
                    "success": False,
                    "message": f"任务 {task_id} 没有 Plan"
                }), 400
            
            # 添加 Plan review 记录
            if 'plan_review_history' not in tasks[i]:
                tasks[i]['plan_review_history'] = []
            
            review_record = {
                'reviewed_at': datetime.now().isoformat(),
                'approved': approved,
                'comment': review_comment
            }
            tasks[i]['plan_review_history'].append(review_record)
            
            if approved:
                # 通过：更新 Plan 状态为 approved
                tasks[i]['plan']['status'] = 'approved'
                message = f"Plan 已通过审核"
            else:
                # 不通过：更新 Plan 状态为 rejected，附带 review 意见
                tasks[i]['plan']['status'] = 'rejected'
                tasks[i]['plan_review_comment'] = review_comment
                
                # 把 Review 意见写入 progress
                if review_comment:
                    review_note = f"📝 Plan Review 不通过意见：{review_comment}"
                    if tasks[i].get('progress'):
                        tasks[i]['progress'] = f"{tasks[i]['progress']}\n\n{review_note}"
                    else:
                        tasks[i]['progress'] = review_note
                
                message = f"Plan 已退回，附带 review 意见"
            
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
            "message": message
        })
    else:
        return jsonify({
            "success": False,
            "message": "保存任务失败"
        }), 500

@app.route('/api/tasks/<task_id>/review', methods=['POST'])
def review_task(task_id):
    """Review 任务（通过或不通过）"""
    data = request.json
    approved = data.get('approved', False)
    review_comment = data.get('comment', '')
    
    # 加载现有数据
    todos_data = load_todos_from_json(TODOS_FILE)
    tasks = todos_data.get('todos', [])
    
    # 找到任务
    task_found = False
    for i, task in enumerate(tasks):
        if task.get('id') == task_id:
            # 添加 review 记录
            if 'review_history' not in tasks[i]:
                tasks[i]['review_history'] = []
            
            review_record = {
                'reviewed_at': datetime.now().isoformat(),
                'approved': approved,
                'comment': review_comment
            }
            tasks[i]['review_history'].append(review_record)
            
            if approved:
                # 通过：归档任务
                # 先从当前任务列表移除
                task_to_archive = tasks.pop(i)
                task_to_archive['archived_at'] = datetime.now().isoformat()
                
                # 保存到归档文件（按月份）
                archive_month = datetime.now().strftime('%Y-%m')
                archive_file = TODO_ARCHIVE_DIR / f"{archive_month}.json"
                
                archive_data = load_todos_from_json(archive_file)
                archive_data['todos'].append(task_to_archive)
                save_todos_to_json(archive_data, archive_file)
                
                message = f"任务 {task_id} 已通过 review 并归档"
            else:
                # 不通过：回到进行中，附带 review 意见
                tasks[i]['status'] = 'in-progress'
                tasks[i]['review_comment'] = review_comment
                
                # 把 Review 意见写入 progress，让 AI 能够理解
                if review_comment:
                    review_note = f"📝 Review 不通过意见：{review_comment}"
                    if tasks[i].get('progress'):
                        tasks[i]['progress'] = f"{tasks[i]['progress']}\n\n{review_note}"
                    else:
                        tasks[i]['progress'] = review_note
                
                message = f"任务 {task_id} 已退回，附带 review 意见"
            
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
            "message": message
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
        valid_priorities = ['high', 'medium', 'low'] + [f'P{i}' for i in range(10)]
        if priority and priority not in valid_priorities:
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
# 任务执行日志 API
# ============================================

@app.route('/api/execution-logs', methods=['GET'])
def get_execution_logs():
    """获取任务执行日志"""
    if not TASK_LOGGER_AVAILABLE or not task_logger:
        return jsonify({
            "success": False,
            "message": "任务执行日志系统不可用"
        }), 503
    
    # 读取今天的日志文件
    logs = []
    try:
        if task_logger.log_file.exists():
            with open(task_logger.log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        logs.append(json.loads(line))
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"读取日志失败: {e}"
        }), 500
    
    return jsonify({
        "success": True,
        "logs": logs,
        "total": len(logs)
    })

@app.route('/api/execution-metrics', methods=['GET'])
def get_execution_metrics():
    """获取任务执行指标"""
    if not TASK_LOGGER_AVAILABLE or not task_logger:
        return jsonify({
            "success": False,
            "message": "任务执行日志系统不可用"
        }), 503
    
    try:
        metrics = task_logger.get_overall_metrics()
        alerts = task_logger.check_alerts()
        return jsonify({
            "success": True,
            "metrics": metrics,
            "alerts": alerts
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"获取指标失败: {e}"
        }), 500

@app.route('/api/execution-logs/<task_id>', methods=['GET'])
def get_task_execution_logs(task_id):
    """获取特定任务的执行日志"""
    if not TASK_LOGGER_AVAILABLE or not task_logger:
        return jsonify({
            "success": False,
            "message": "任务执行日志系统不可用"
        }), 503
    
    logs = []
    try:
        if task_logger.log_file.exists():
            with open(task_logger.log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        log_entry = json.loads(line)
                        if log_entry.get('task_id') == task_id:
                            logs.append(log_entry)
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"读取日志失败: {e}"
        }), 500
    
    # 尝试加载任务产物
    artifact = None
    try:
        artifact_data = task_logger.load_artifact(task_id)
        if artifact_data:
            from dataclasses import asdict
            artifact = asdict(artifact_data)
    except:
        pass
    
    return jsonify({
        "success": True,
        "task_id": task_id,
        "logs": logs,
        "artifact": artifact,
        "total": len(logs)
    })


@app.route('/api/execution-time-series', methods=['GET'])
def get_execution_time_series():
    """获取执行时间序列数据（用于图表）"""
    if not TASK_LOGGER_AVAILABLE or not task_logger:
        return jsonify({
            "success": False,
            "message": "任务执行日志系统不可用"
        }), 503
    
    try:
        # 获取所有历史日志文件
        time_series_data = []
        completed_tasks = []
        
        for log_file in task_logger.logs_dir.glob("task_execution_*.jsonl"):
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            log_entry = json.loads(line)
                            time_series_data.append(log_entry)
            except Exception as e:
                print(f"Error reading {log_file}: {e}", file=sys.stderr)
        
        # 从 metrics 中获取已完成任务
        for task_id, metrics in task_logger.metrics.items():
            if (metrics.status == 'completed' and 
                metrics.started_at and 
                metrics.completed_at):
                completed_tasks.append({
                    'task_id': task_id,
                    'agent': metrics.agent or 'unknown',
                    'started_at': metrics.started_at,
                    'completed_at': metrics.completed_at,
                    'execution_time_seconds': metrics.execution_time_seconds,
                    'execution_time_minutes': round(metrics.execution_time_seconds / 60, 2)
                })
        
        # 按时间排序
        completed_tasks.sort(key=lambda x: x['started_at'])
        
        # 生成时间序列数据点
        chart_data = []
        cumulative_tasks = 0
        for task in completed_tasks:
            cumulative_tasks += 1
            chart_data.append({
                'date': task['started_at'][:10],
                'timestamp': task['started_at'],
                'task_id': task['task_id'],
                'agent': task['agent'],
                'execution_time_minutes': task['execution_time_minutes'],
                'execution_time_seconds': task['execution_time_seconds'],
                'cumulative_tasks': cumulative_tasks
            })
        
        # 按日期聚合
        daily_aggregates = {}
        for task in completed_tasks:
            date = task['started_at'][:10]
            if date not in daily_aggregates:
                daily_aggregates[date] = {
                    'date': date,
                    'task_count': 0,
                    'total_execution_minutes': 0,
                    'avg_execution_minutes': 0,
                    'max_execution_minutes': 0,
                    'min_execution_minutes': float('inf')
                }
            
            daily_aggregates[date]['task_count'] += 1
            daily_aggregates[date]['total_execution_minutes'] += task['execution_time_minutes']
            daily_aggregates[date]['max_execution_minutes'] = max(
                daily_aggregates[date]['max_execution_minutes'],
                task['execution_time_minutes']
            )
            daily_aggregates[date]['min_execution_minutes'] = min(
                daily_aggregates[date]['min_execution_minutes'],
                task['execution_time_minutes']
            )
        
        # 计算每日平均值
        for date, agg in daily_aggregates.items():
            agg['avg_execution_minutes'] = round(
                agg['total_execution_minutes'] / agg['task_count'],
                2
            )
        
        daily_data = sorted(daily_aggregates.values(), key=lambda x: x['date'])
        
        # 按 Agent 聚合
        agent_aggregates = {}
        for task in completed_tasks:
            agent = task['agent']
            if agent not in agent_aggregates:
                agent_aggregates[agent] = {
                    'agent': agent,
                    'task_count': 0,
                    'total_execution_minutes': 0,
                    'avg_execution_minutes': 0,
                    'execution_times': []
                }
            
            agent_aggregates[agent]['task_count'] += 1
            agent_aggregates[agent]['total_execution_minutes'] += task['execution_time_minutes']
            agent_aggregates[agent]['execution_times'].append(task['execution_time_minutes'])
        
        # 计算每个 Agent 的平均值
        for agent, agg in agent_aggregates.items():
            agg['avg_execution_minutes'] = round(
                agg['total_execution_minutes'] / agg['task_count'] if agg['task_count'] > 0 else 0,
                2
            )
            if agg['execution_times']:
                agg['max_execution_minutes'] = max(agg['execution_times'])
                agg['min_execution_minutes'] = min(agg['execution_times'])
            del agg['execution_times']
        
        agent_data = list(agent_aggregates.values())
        
        return jsonify({
            "success": True,
            "chart_data": chart_data,
            "daily_data": daily_data,
            "agent_data": agent_data,
            "completed_tasks_count": len(completed_tasks)
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"获取时间序列数据失败: {e}"
        }), 500

# ============================================
# 配置 API
# ============================================

@app.route('/api/config', methods=['GET'])
def get_config():
    """获取配置"""
    return jsonify({
        "success": True,
        "config": config
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
    print("Todos Web Manager - 后端服务")
    print("=" * 60)
    print(f"仓库根目录: {REPO_ROOT}")
    print(f"任务文件: {TODOS_FILE}")
    print(f"归档目录: {TODO_ARCHIVE_DIR}")
    print(f"INBOX 文件: {INBOX_FILE}")
    print("=" * 60)
    print("可用的 API:")
    print("  - GET    /api/tasks              - 获取任务列表")
    print("  - POST   /api/tasks              - 添加新任务")
    print("  - PUT    /api/tasks/<id>         - 更新任务")
    print("  - DELETE /api/tasks/<id>         - 删除任务")
    print("  - PUT    /api/tasks/<id>/status  - 更新任务状态")
    print("  - POST   /api/tasks/<id>/review  - Review 任务（通过/不通过）")
    print("  - GET    /api/tasks/archive      - 获取归档任务")
    print("  - GET    /api/plans               - 获取 Plan 列表")
    print("  - PUT    /api/plans/<id>/status   - 更新 Plan 状态（approve/reject）")
    print("  - GET    /api/git/status          - 获取 Git 状态")
    print("  - POST   /api/git/commit          - 提交 Git 更改")
    print("  - POST   /api/git/push            - 推送到远程仓库")
    print("  - POST   /api/git/pull            - 从远程仓库拉取")
    print("=" * 60)
    print("启动服务器: http://localhost:5000")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=5000, debug=True)
