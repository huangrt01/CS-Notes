# JSON 单一数据源 + Web 可视化 - 设计方案

**日期**: 2026-02-19  
**作者**: AI  
**状态**: ✅ 设计完成！

## 问题背景

当前痛点：
- ❌ 需要同时维护 Markdown 和 JSON 两份数据
- ❌ 容易出现数据不同步的问题
- ❌ 增加维护成本
- ❌ 用户体验不好

**目标**：
- ✅ 完全用 JSON 作为单一数据源
- ✅ Web 可视化直接读取/写入 JSON
- ✅ 不再需要维护 Markdown 格式
- ✅ 提供 Markdown 导出功能（可选，用于备份）

---

## 方案概述

### 核心思路

```
┌─────────────────────────────────────────────────────────┐
│                    单一数据源：JSON                        │
│  .trae/todos/todos.json                                  │
│  .trae/todos/archive.json                                │
│  .trae/todos/inbox.json                                  │
└────────────────────┬────────────────────────────────────┘
                     │
         ┌───────────┴───────────┐
         │                       │
         ▼                       ▼
┌──────────────────┐   ┌──────────────────┐
│  Web 可视化界面    │   │  命令行工具       │
│  (index.html)    │   │  (todo-cli.py)   │
└──────────────────┘   └──────────────────┘
         │                       │
         └───────────┬───────────┘
                     │
                     ▼
         ┌──────────────────┐
         │   Git 自动同步    │
         │  (JSON 文件变更)  │
         └──────────────────┘
```

---

## JSON 数据结构设计

### 1. 主任务文件：`.trae/todos/todos.json`

```json
{
  "version": "1.0.0",
  "lastUpdated": "2026-02-19T14:00:00.000Z",
  "tasks": [
    {
      "id": "task-20260219-001",
      "title": "实现 JSON 单一数据源",
      "description": "完全用 JSON 存储任务，不再维护 Markdown",
      "status": "in-progress",
      "priority": "high",
      "assignee": "AI",
      "feedbackRequired": false,
      "links": [
        ".trae/documents/JSON-单一数据源-Web-可视化-设计方案.md"
      ],
      "definitionOfDone": [
        "设计 JSON 数据结构",
        "实现 JSON 读写功能",
        "更新 Web 可视化界面",
        "提供 Markdown 导出功能（可选）"
      ],
      "progress": "正在设计 JSON 数据结构",
      "startedAt": "2026-02-19",
      "completedAt": null,
      "tags": ["json", "web", "todo-manager"],
      "metadata": {
        "createdBy": "AI",
        "createdAt": "2026-02-19T14:00:00.000Z",
        "updatedBy": "AI",
        "updatedAt": "2026-02-19T14:00:00.000Z"
      }
    }
  ],
  "stats": {
    "total": 1,
    "pending": 0,
    "inProgress": 1,
    "completed": 0
  }
}
```

### 2. 归档任务文件：`.trae/todos/archive.json`

```json
{
  "version": "1.0.0",
  "lastUpdated": "2026-02-19T14:00:00.000Z",
  "archives": [
    {
      "date": "2026-02-19",
      "tasks": [
        {
          "id": "task-20260218-001",
          "title": "已完成的任务",
          "description": "这是一个已完成的任务",
          "status": "completed",
          "priority": "high",
          "assignee": "AI",
          "startedAt": "2026-02-18",
          "completedAt": "2026-02-18",
          "result": "任务完成的结果描述"
        }
      ]
    }
  ]
}
```

### 3. INBOX 文件：`.trae/todos/inbox.json`

```json
{
  "version": "1.0.0",
  "lastUpdated": "2026-02-19T14:00:00.000Z",
  "inbox": [
    {
      "id": "inbox-20260219-001",
      "title": "快速记录的任务",
      "description": "从手机快速提交的任务",
      "priority": "medium",
      "assignee": "User",
      "createdAt": "2026-02-19T14:00:00.000Z",
      "source": "web-mobile"
    }
  ]
}
```

---

## 文件结构

```
.trae/todos/
├── todos.json          # 主任务文件（单一数据源）
├── archive.json        # 归档任务文件
├── inbox.json          # INBOX 文件
├── schema.json         # JSON Schema（用于验证）
└── backup/             # 备份目录
    ├── todos-2026-02-19.json
    ├── todos-2026-02-18.json
    └── ...
```

---

## 实施路线图

### Phase 1: 基础 JSON 功能（1-2 天）

**目标**：建立 JSON 单一数据源的基础

- [x] 设计 JSON 数据结构
- [ ] 创建 `.trae/todos/` 目录
- [ ] 创建 JSON Schema 文件（schema.json）
- [ ] 实现 Python JSON 读写工具类
- [ ] 实现 JSON 数据验证功能
- [ ] 实现自动备份功能

### Phase 2: 数据迁移（1 天）

**目标**：从 Markdown 迁移到 JSON

- [ ] 编写 Markdown → JSON 迁移脚本
- [ ] 解析现有的 `todos管理系统.md`
- [ ] 解析现有的 `TODO_ARCHIVE.md`
- [ ] 解析现有的 `INBOX.md`
- [️ 生成对应的 JSON 文件
- [ ] 验证数据完整性
- [ ] 备份原 Markdown 文件

### Phase 3: Web 可视化更新（1-2 天）

**目标**：更新 Web 界面直接操作 JSON

- [ ] 更新 `index-enhanced.html` 直接读取 JSON
- [ ] 实现 JSON 文件上传/下载功能
- [ ] 实现任务 CRUD（创建/读取/更新/删除）
- [ ] 实现任务筛选和搜索
- [ ] 实现任务排序
- 实现批量操作

### Phase 4: 命令行工具（1 天）

**目标**：提供命令行工具操作 JSON

- [ ] 创建 `todo-cli.py` 命令行工具
- [ ] 支持 `list` - 列出任务
- [ ] 支持 `add` - 添加任务
- [ ] 支持 `update` - 更新任务
- [ ] 支持 `complete` - 完成任务
- [ ] 支持 `archive` - 归档任务
- [ ] 支持 `export` - 导出 Markdown（可选）

### Phase 5: Markdown 兼容性（可选，1 天）

**目标**：提供 Markdown 导出功能（用于备份）

- [ ] 实现 JSON → Markdown 导出功能
- [ ] 支持导出为 Markdown 格式
- [ ] 支持按日期分组导出
- [ ] 支持自定义导出模板

---

## JSON 工具类设计

### Python 工具类：`todo_json.py`

```python
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

class TodoJSONManager:
    """JSON 任务管理器"""
    
    def __init__(self, todos_dir: Path):
        self.todos_dir = todos_dir
        self.todos_file = todos_dir / "todos.json"
        self.archive_file = todos_dir / "archive.json"
        self.inbox_file = todos_dir / "inbox.json"
        self.schema_file = todos_dir / "schema.json"
        
        # 确保目录存在
        self.todos_dir.mkdir(parents=True, exist_ok=True)
    
    def load_todos(self) -> Dict:
        """加载任务"""
        if not self.todos_file.exists():
            return self._create_empty_todos()
        
        with open(self.todos_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def save_todos(self, data: Dict):
        """保存任务"""
        data['lastUpdated'] = datetime.now().isoformat()
        
        # 自动备份
        self._backup()
        
        with open(self.todos_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def add_task(self, task: Dict) -> str:
        """添加任务"""
        todos = self.load_todos()
        
        # 生成任务 ID
        task_id = f"task-{datetime.now().strftime('%Y%m%d')}-{len(todos['tasks']) + 1:03d}"
        task['id'] = task_id
        task['metadata'] = {
            'createdBy': 'AI',
            'createdAt': datetime.now().isoformat(),
            'updatedBy': 'AI',
            'updatedAt': datetime.now().isoformat()
        }
        
        todos['tasks'].append(task)
        self._update_stats(todos)
        self.save_todos(todos)
        
        return task_id
    
    def update_task(self, task_id: str, updates: Dict) -> bool:
        """更新任务"""
        todos = self.load_todos()
        
        for task in todos['tasks']:
            if task['id'] == task_id:
                task.update(updates)
                task['metadata']['updatedBy'] = 'AI'
                task['metadata']['updatedAt'] = datetime.now().isoformat()
                self._update_stats(todos)
                self.save_todos(todos)
                return True
        
        return False
    
    def complete_task(self, task_id: str) -> bool:
        """完成任务"""
        return self.update_task(task_id, {
            'status': 'completed',
            'completedAt': datetime.now().strftime('%Y-%m-%d')
        })
    
    def archive_task(self, task_id: str) -> bool:
        """归档任务"""
        todos = self.load_todos()
        
        # 找到任务
        task_index = None
        task_to_archive = None
        for i, task in enumerate(todos['tasks']):
            if task['id'] == task_id:
                task_index = i
                task_to_archive = task
                break
        
        if not task_to_archive:
            return False
        
        # 从主列表移除
        todos['tasks'].pop(task_index)
        
        # 添加到归档
        archives = self._load_archives()
        today = datetime.now().strftime('%Y-%m-%d')
        
        # 找到今天的归档分组
        today_archive = next((a for a in archives['archives'] if a['date'] == today), None)
        if not today_archive:
            today_archive = {'date': today, 'tasks': []}
            archives['archives'].insert(0, today_archive)
        
        today_archive['tasks'].append(task_to_archive)
        self._save_archives(archives)
        
        # 更新主列表
        self._update_stats(todos)
        self.save_todos(todos)
        
        return True
    
    def _update_stats(self, todos: Dict):
        """更新统计信息"""
        todos['stats'] = {
            'total': len(todos['tasks']),
            'pending': len([t for t in todos['tasks'] if t['status'] == 'pending']),
            'inProgress': len([t for t in todos['tasks'] if t['status'] == 'in-progress']),
            'completed': len([t for t in todos['tasks'] if t['status'] == 'completed'])
        }
    
    def _create_empty_todos(self) -> Dict:
        """创建空的任务数据"""
        return {
            'version': '1.0.0',
            'lastUpdated': datetime.now().isoformat(),
            'tasks': [],
            'stats': {
                'total': 0,
                'pending': 0,
                'inProgress': 0,
                'completed': 0
            }
        }
    
    def _load_archives(self) -> Dict:
        """加载归档数据"""
        if not self.archive_file.exists():
            return {
                'version': '1.0.0',
                'lastUpdated': datetime.now().isoformat(),
                'archives': []
            }
        
        with open(self.archive_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _save_archives(self, archives: Dict):
        """保存归档数据"""
        archives['lastUpdated'] = datetime.now().isoformat()
        
        with open(self.archive_file, 'w', encoding='utf-8') as f:
            json.dump(archives, f, ensure_ascii=False, indent=2)
    
    def _backup(self):
        """自动备份"""
        backup_dir = self.todos_dir / "backup"
        backup_dir.mkdir(exist_ok=True)
        
        # 备份当前文件
        today = datetime.now().strftime('%Y-%m-%d')
        backup_file = backup_dir / f"todos-{today}.json"
        
        if self.todos_file.exists():
            import shutil
            shutil.copy2(self.todos_file, backup_file)
        
        # 只保留最近 7 天的备份
        backups = sorted(backup_dir.glob("todos-*.json"), reverse=True)
        for old_backup in backups[7:]:
            old_backup.unlink()
```

---

## 迁移脚本设计

### Markdown → JSON 迁移脚本：`migrate_markdown_to_json.py`

```python
#!/usr/bin/env python3
"""
Markdown → JSON 迁移脚本
"""

import re
import json
from pathlib import Path
from datetime import datetime

def parse_todos_from_markdown(file_path: Path) -> list:
    """从 Markdown 解析任务"""
    if not file_path.exists():
        return []
    
    content = file_path.read_text(encoding='utf-8')
    tasks = []
    current_task = None
    
    lines = content.split('\n')
    for line in lines:
        # 匹配任务行
        task_match = re.match(r'^(\*|-)\s+\[([ x])\]\s+(.*)$', line)
        if task_match:
            if current_task:
                tasks.append(current_task)
            
            list_marker, status_marker, title = task_match.groups()
            current_task = {
                'id': f"task-{datetime.now().strftime('%Y%m%d')}-{len(tasks) + 1:03d}",
                'title': title.strip(),
                'status': 'completed' if status_marker == 'x' else 'pending',
                'priority': 'medium',
                'assignee': 'User',
                'description': '',
                'links': [],
                'definitionOfDone': [],
                'progress': '',
                'startedAt': '',
                'completedAt': '',
                'tags': [],
                'metadata': {
                    'createdBy': 'migration',
                    'createdAt': datetime.now().isoformat(),
                    'updatedBy': 'migration',
                    'updatedAt': datetime.now().isoformat()
                }
            }
        elif current_task:
            # 解析任务属性（省略具体实现）
            pass
    
    if current_task:
        tasks.append(current_task)
    
    return tasks

def main():
    """主函数"""
    repo_root = Path(__file__).parent.parent.parent
    todos_dir = repo_root / ".trae/todos"
    todos_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 解析 todos管理系统.md
    print("正在解析 todos管理系统.md...")
    todos_file = repo_root / ".trae/documents/todos管理系统.md"
    tasks = parse_todos_from_markdown(todos_file)
    
    # 2. 生成 todos.json
    print(f"正在生成 todos.json（{len(tasks)} 个任务）...")
    todos_data = {
        'version': '1.0.0',
        'lastUpdated': datetime.now().isoformat(),
        'tasks': tasks,
        'stats': {
            'total': len(tasks),
            'pending': len([t for t in tasks if t['status'] == 'pending']),
            'inProgress': len([t for t in tasks if t['status'] == 'in-progress']),
            'completed': len([t for t in tasks if t['status'] == 'completed'])
        }
    }
    
    output_file = todos_dir / "todos.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(todos_data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 迁移完成！输出文件：{output_file}")
    print(f"   - 总任务数：{todos_data['stats']['total']}")
    print(f"   - 待处理：{todos_data['stats']['pending']}")
    print(f"   - 进行中：{todos_data['stats']['inProgress']}")
    print(f"   - 已完成：{todos_data['stats']['completed']}")

if __name__ == '__main__':
    main()
```

---

## 优势总结

### ✅ 使用 JSON 单一数据源的优势

1. **数据一致性**
   - 不再需要同时维护 Markdown 和 JSON
   - 避免数据不同步的问题
   - 减少维护成本

2. **Web 可视化更简单**
   - 直接读取/写入 JSON
   - 不需要复杂的 Markdown 解析
   - 性能更好

3. **更丰富的数据结构**
   - 支持嵌套对象
   - 支持元数据（createdAt、updatedAt、createdBy 等）
   - 支持更复杂的查询和筛选

4. **更好的工具支持**
   - JSON 是标准格式，有大量工具支持
   - 可以使用 JSON Schema 验证数据
   - 可以使用 JSON Patch 进行增量更新

5. **Git 友好**
   - JSON 格式清晰，diff 易读
   - 可以使用 Git 进行版本控制
   - 自动备份功能

---

## 总结

### 🎯 核心目标

- ✅ **完全用 JSON 作为单一数据源**
- ✅ **Web 可视化直接操作 JSON**
- ✅ **不再需要维护 Markdown 格式**
- ✅ **提供 Markdown 导出功能（可选）**

### 📋 实施步骤

1. **Phase 1**: 基础 JSON 功能（1-2 天）
2. **Phase 2**: 数据迁移（1 天）
3. **Phase 3**: Web 可视化更新（1-2 天）
4. **Phase 4**: 命令行工具（1 天）
5. **Phase 5**: Markdown 兼容性（可选，1 天）

### 🚀 立即开始

可以立即开始实施 Phase 1，建立 JSON 单一数据源的基础！

---

**方案完成时间**: 2026-02-19  
**下一步**: 开始实施 Phase 1，创建 JSON 工具类和数据结构！
