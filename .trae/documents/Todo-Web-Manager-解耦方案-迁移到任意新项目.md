# Todo Web Manager 解耦方案 - 迁移到任意新项目

## 设计日期：2026-02-23

## 1. 问题分析

### 当前架构的强耦合点

#### 后端（server.py）
1. **路径硬编码**
   - `REPO_ROOT = Path(__file__).parent.parent.parent`
   - `TODOS_FILE = REPO_ROOT / ".trae/todos/todos.json"`
   - `TODO_ARCHIVE_DIR = REPO_ROOT / ".trae/todos/archive"`
   - `PLANS_DIR = REPO_ROOT / ".trae/plans"`
   - `INBOX_FILE = REPO_ROOT / ".trae/documents/INBOX.md"`

2. **注释中的路径**
   - `/Users/bytedance/CS-Notes/.trae/web-manager`（Mac 路径）

3. **项目名称**
   - "CS-Notes Todos Web Manager"（标题中的项目名称）

#### 前端（index-enhanced.html）
1. **标题**
   - "CS-Notes Todos Web Manager"
   - 注释中的 "CS-Notes Todos Web Manager - 使用说明"

---

## 2. 解耦方案

### 方案1：配置文件方式（推荐）

#### 2.1 创建配置文件 config.json

在 `.trae/web-manager/` 目录下创建 `config.json`：

```json
{
  "project": {
    "name": "CS-Notes",
    "title": "Todos Web Manager"
  },
  "paths": {
    "repo_root": "../..",
    "todos_file": ".trae/todos/todos.json",
    "todo_archive_dir": ".trae/todos/archive",
    "plans_dir": ".trae/plans",
    "inbox_file": ".trae/documents/INBOX.md"
  },
  "priority": {
    "enabled": true,
    "system": "P0-P9",
    "default": "P4"
  }
}
```

#### 2.2 修改 server.py

1. **读取配置文件**
```python
import json
from pathlib import Path

CONFIG_FILE = Path(__file__).parent / "config.json"

def load_config():
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return get_default_config()

def get_default_config():
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
```

2. **使用配置文件中的路径**
```python
config = load_config()
REPO_ROOT = Path(__file__).parent.parent.parent
TODOS_FILE = REPO_ROOT / config['paths']['todos_file']
TODO_ARCHIVE_DIR = REPO_ROOT / config['paths']['todo_archive_dir']
PLANS_DIR = REPO_ROOT / config['paths']['plans_dir']
INBOX_FILE = REPO_ROOT / config['paths']['inbox_file']
```

#### 2.3 修改前端 index-enhanced.html

1. **添加配置API端点**
```python
@app.route('/api/config', methods=['GET'])
def get_config():
    """获取配置"""
    config = load_config()
    return jsonify({
        "success": True,
        "config": config
    })
```

2. **前端使用配置**
```javascript
async function loadConfig() {
    const response = await fetch('/api/config');
    const data = await response.json();
    if (data.success) {
        document.title = `${data.config.project.name} ${data.config.project.title}`;
    }
}
```

---

### 方案2：环境变量方式

#### 2.1 使用环境变量

```python
import os

REPO_ROOT = Path(os.getenv('REPO_ROOT', Path(__file__).parent.parent.parent))
TODOS_FILE = REPO_ROOT / os.getenv('TODOS_FILE', '.trae/todos/todos.json')
TODO_ARCHIVE_DIR = REPO_ROOT / os.getenv('TODO_ARCHIVE_DIR', '.trae/todos/archive')
PLANS_DIR = REPO_ROOT / os.getenv('PLANS_DIR', '.trae/plans')
INBOX_FILE = REPO_ROOT / os.getenv('INBOX_FILE', '.trae/documents/INBOX.md')
```

---

## 3. 目录结构标准化

### 推荐的标准目录结构

```
项目根目录/
├── .trae/
│   ├── todos/
│   │   ├── todos.json
│   │   └── archive/
│   ├── plans/
│   ├── documents/
│   │   └── INBOX.md
│   ├── web-manager/
│   │   ├── server.py
│   │   ├── index-enhanced.html
│   │   └── config.json
│   └── skills/
└── ...
```

---

## 4. 迁移步骤

### 步骤1：创建配置文件
在新项目的 `.trae/web-manager/` 目录下创建 `config.json`

### 步骤2：复制文件
将以下文件复制到新项目：
- `.trae/web-manager/server.py`
- `.trae/web-manager/index-enhanced.html`
- `.trae/web-manager/config.json`

### 步骤3：修改配置
根据新项目的需求修改 `config.json`

### 步骤4：创建目录结构
在新项目中创建标准的 `.trae/` 目录结构

### 步骤5：测试验证
启动server.py，测试所有功能是否正常工作

---

## 5. 前端可配置化

### 5.1 项目标题可配置
```javascript
// 从配置中获取项目名称
const projectName = config.project.name;
document.title = `${projectName} Todos Web Manager`;
```

### 5.2 主题色可配置
```json
{
  "theme": {
    "primary": "#667eea",
    "secondary": "#764ba2",
    "success": "#10b981",
    "warning": "#f59e0b",
    "danger": "#ef4444"
  }
}
```

---

## 6. Python 包化（远期）

### 6.1 创建独立的 Python 包
```
todos-web-manager/
├── setup.py
├── todos_web_manager/
│   ├── __init__.py
│   ├── server.py
│   ├── config.py
│   └── static/
│       └── index.html
```

### 6.2 使用 pip 安装
```bash
pip install todos-web-manager
```

### 6.3 启动服务
```bash
todos-web-manager --config config.json
```

---

## 7. 总结

### 推荐方案
使用**方案1：配置文件方式**，因为：
1. 简单易用
2. 配置清晰可见
3. 易于迁移
4. 不需要环境变量配置

### 关键改进
1. ✅ 路径可配置
2. ✅ 项目名称可配置
3. ✅ 目录结构标准化
4. ✅ 易于迁移到任意新项目

