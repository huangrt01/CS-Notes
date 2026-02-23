# Todo Web Manager 解耦方案 - 迁移到任意新项目

## 设计日期：2026-02-23

## 1. 问题分析

### 当前架构的强耦合点

#### 技术耦合
1. **路径硬编码**
   - `REPO_ROOT = Path(__file__).parent.parent.parent`
   - `TODOS_FILE = REPO_ROOT / ".trae/todos/todos.json"`
   - `TODO_ARCHIVE_DIR = REPO_ROOT / ".trae/todos/archive"`
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

### 知识与经验耦合

除了技术层面的耦合，还需要解决**知识与经验层面的耦合**，使规则文档和记忆经验能够通用化到任意新项目：

1. **项目规则文档强耦合**
   - `.trae/rules/` 目录下的项目特定规则
   - `.trae/documents/` 目录下的项目特定文档
   - 这些规则和文档是为 CS-Notes 项目定制的

2. **AI 工作流经验强耦合**
   - todo 管理的异步体系经验
   - Plan 机制的执行经验
   - 优先级分配策略经验

3. **需要通用化的方面**
   - 将项目特定规则抽象为通用模板
   - 将经验沉淀为可复用的最佳实践
   - 提供规则自定义和扩展机制

---

## 2. 解耦方案

### 方案1：配置文件方式（推荐）

#### 2.1 创建配置文件

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
    "inbox_file": ".trae/documents/INBOX.md"
  },
  "priority": {
    "enabled": true,
    "system": "P0-P9",
    "default": "P4",
    "definitions": {
      "P0": "最高优先级，立即处理（笔记整理类任务固定）",
      "P1": "核心功能、用户体验关键",
      "P2": "重要功能、用户体验优化",
      "P3": "有用的功能、体验提升",
      "P4": "常规功能、一般优化",
      "P5": "次要功能、小优化",
      "P6": "可有可无的功能",
      "P7": "锦上添花",
      "P8": "未来考虑",
      "P9": "几乎不做"
    }
  },
  "plan": {
    "enabled": true,
    "default_enabled": false,
    "review_required": true
  },
  "ui": {
    "layout": "two-column",
    "show_all_tasks_tab": true
  },
  "theme": {
    "primary": "#667eea",
    "secondary": "#764ba2",
    "success": "#10b981",
    "warning": "#f59e0b",
    "danger": "#ef4444"
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
            "inbox_file": ".trae/documents/INBOX.md"
        },
        "priority": {
            "enabled": true,
            "system": "P0-P9",
            "default": "P4",
            "definitions": {
                "P0": "最高优先级，立即处理",
                "P1": "核心功能、用户体验关键",
                "P2": "重要功能、用户体验优化",
                "P3": "有用的功能、体验提升",
                "P4": "常规功能、一般优化",
                "P5": "次要功能、小优化",
                "P6": "可有可无的功能",
                "P7": "锦上添花",
                "P8": "未来考虑",
                "P9": "几乎不做"
            }
        },
        "plan": {
            "enabled": true,
            "default_enabled": false,
            "review_required": true
        },
        "ui": {
            "layout": "two-column",
            "show_all_tasks_tab": true
        },
        "theme": {
            "primary": "#667eea",
            "secondary": "#764ba2",
            "success": "#10b981",
            "warning": "#f59e0b",
            "danger": "#ef4444"
        }
    }
```

2. **使用配置文件中的路径**
```python
config = load_config()
REPO_ROOT = Path(__file__).parent.parent.parent
TODOS_FILE = REPO_ROOT / config['paths']['todos_file']
TODO_ARCHIVE_DIR = REPO_ROOT / config['paths']['todo_archive_dir']
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

## 3. 知识与经验通用化方案（基于 review 反馈）

### 3.1 规则文档通用化

#### 规则分层架构：
1. **核心规则层（通用）
   - 不依赖具体项目的通用规则
   - 如：Todo 数据结构、优先级体系定义
   
2. **模板规则层（可配置）**
   - 项目特定规则的模板
   - 如：项目规则模板、写作风格模板

3. **项目规则层（定制）**
   - 具体项目的定制规则
   - 如：CS-Notes 项目特定规则

#### 规则文件结构：
```
.trae/
├── rules/
│   ├── core/              # 核心规则（通用）
│   │   ├── todo-data-model.md
│   │   └── priority-system.md
│   ├── templates/         # 规则模板（可配置）
│   │   ├── project-rules-template.md
│   │   └── writing-style-template.md
│   └── project/         # 项目规则（定制）
│       └── project-rules.md
```

---

### 3.2 经验沉淀为最佳实践

将 AI 工作流经验沉淀为可复用的最佳实践文档，包括 `.openclaw-memory/` 中的经验：

#### 3.2.1 Todo 管理最佳实践
- 如何拆解复杂任务
- 如何编写清晰的 definition_of_done
- 如何设置合理的优先级
- 单一数据源原则
- Plan 机制本质化重构原则
- Todo 归档前的用户确认流程

#### 3.2.2 Plan 机制最佳实践
- 如何设计高质量的 Plan
- Plan 的验收标准
- Plan 的 Review 标准
- 通过/不通过的处理流程

#### 3.2.3 AI 协作最佳实践
- 人机协作的工作流
- 如何有效利用 AI 能力
- 用户消息优先原则
- 长时间任务后台化原则
- 自主推进原则（只有需要用户做选择题时才找用户确认）

#### 3.2.4 笔记整理最佳实践（基于 MEMORY.md）
- 笔记整理核心原则
- 引用原则
- 无幻觉原则
- 简洁原则
- Markdown 笔记层级注意事项
- 用户发资料/链接时的两种意图处理方式

#### 3.2.5 Git 操作 SOP（基于 MEMORY.md）
- 使用 todo-push.sh 和 todo-pull.sh 作为标准流程
- 白名单/黑名单机制
- 公司项目文档绝对不能 push 到 git

#### 3.2.6 安全意识最佳实践（基于 SOUL.md 和 MEMORY.md）
- 敏感内容绝对不允许上传到公开 GitHub 仓库
- Prompt Injection Defense
- Skills / Plugin Poisoning Defense
- Explicit Confirmation for Sensitive Actions
- Restricted Paths（Never Access Unless User Explicitly Requests）
- Anti‑Leak Output Discipline
- Suspicion Protocol（Stop First）

#### 3.2.7 快捷指令和工作模式（基于 AGENTS.md）
- 快捷指令：推进todo
- 快捷指令：沉淀
- 快捷指令：沉淀v2.0
- 稳妥的工作模式
- 任务推进原则（注重更充分的探索）

#### 3.2.8 任务交付原则（基于 MEMORY.md）
- E2E 落地思路
- 落地的定义
- 任务交付要更侧重于更 solid 的端到端验证

#### 3.2.9 错误反思与改进原则（基于 MEMORY.md）
- 失败尝试记录原则
- 及时沉淀经验
- 小步快跑
- 快速迭代

---

### 3.3 配置文件扩展

扩展 config.json，支持规则和经验配置：

```json
{
  "project": {
    "name": "CS-Notes",
    "title": "Todos Web Manager"
  },
  "rules": {
    "core_rules_dir": ".trae/rules/core",
    "template_rules_dir": ".trae/rules/templates",
    "project_rules_dir": ".trae/rules/project"
  },
  "best_practices": {
    "enabled": true,
    "auto_apply": true
  },
  "paths": {
    "repo_root": "../..",
    "todos_file": ".trae/todos/todos.json",
    "todo_archive_dir": ".trae/todos/archive",
    "inbox_file": ".trae/documents/INBOX.md"
  }
}
```

---

## 5. 目录结构标准化

### 推荐的标准目录结构

基于 Plan 机制本质化重构，Plan 现在是 Todo 的一个字段，不再需要独立的 plans 目录：

```
项目根目录/
├── .trae/
│   ├── todos/
│   │   ├── todos.json          # 主任务文件（包含 Plan 字段）
│   │   └── archive/            # 归档任务（按月份）
│   ├── documents/
│   │   └── INBOX.md            # 收件箱
│   ├── rules/
│   │   ├── core/              # 核心规则（通用）
│   │   ├── templates/         # 规则模板（可配置）
│   │   └── project/         # 项目规则（定制）
│   ├── best-practices/        # 最佳实践文档
│   │   ├── todo-management.md
│   │   ├── plan-mechanism.md
│   │   ├── ai-collaboration.md
│   │   ├── note-organization.md
│   │   ├── git-sop.md
│   │   ├── security.md
│   │   ├── shortcuts.md
│   │   ├── task-delivery.md
│   │   └── error-reflection.md
│   ├── web-manager/
│   │   ├── server.py           # Flask 后端
│   │   ├── index-enhanced.html # 增强版前端
│   │   └── config.json         # 配置文件
│   └── skills/
├── .openclaw-memory/          # OpenClaw 记忆体系（可迁移）
│   ├── memory/                # 每日记忆
│   │   └── YYYY-MM-DD.md
│   ├── AGENTS.md              # 工作模式、快捷指令
│   ├── MEMORY.md              # 长期记忆、最佳实践
│   ├── SOUL.md                # 核心身份、安全护栏
│   ├── USER.md                # 用户信息
│   ├── HEARTBEAT.md          # 心跳配置
│   └── TOOLS.md              # 工具配置
└── ...
```

---

## 6. 迁移步骤

### 步骤1：创建配置文件
在新项目的 `.trae/web-manager/` 目录下创建 `config.json`

### 步骤2：复制核心文件
将以下文件复制到新项目：
- `.trae/web-manager/server.py`
- `.trae/web-manager/index-enhanced.html`
- `.trae/web-manager/config.json`

### 步骤3：复制通用规则和最佳实践
将以下内容复制到新项目：
- `.trae/rules/core/` - 核心规则（通用）
- `.trae/rules/templates/` - 规则模板
- `.trae/best-practices/` - 最佳实践文档

### 步骤4：复制 OpenClaw 记忆体系（关键！）
将以下内容复制到新项目（这是 AI 经验的核心）：
- `.openclaw-memory/AGENTS.md` - 工作模式、快捷指令
- `.openclaw-memory/SOUL.md` - 核心身份、安全护栏
- `.openclaw-memory/MEMORY.md` - 长期记忆、最佳实践（可根据新项目裁剪）
- `.openclaw-memory/HEARTBEAT.md` - 心跳配置
- `.openclaw-memory/TOOLS.md` - 工具配置
- `.openclaw-memory/memory/` - 每日记忆模板（可选）

**注意**：`.openclaw-memory/USER.md` 包含用户个人信息，需要在新项目中重新创建

### 步骤5：定制项目规则和记忆
1. 根据新项目需求，基于模板创建 `.trae/rules/project/` 下的项目规则
2. 修改 `.openclaw-memory/MEMORY.md`，移除 CS-Notes 特定内容，保留通用最佳实践
3. 在新项目中创建新的 `.openclaw-memory/USER.md`

### 步骤6：修改配置
根据新项目的需求修改 `config.json`

### 步骤7：创建目录结构
在新项目中创建标准的 `.trae/` 和 `.openclaw-memory/` 目录结构

### 步骤8：测试验证
启动server.py，测试所有功能是否正常工作

---

## 7. 功能配置化

### 7.1 项目标题可配置
```javascript
// 从配置中获取项目名称
const projectName = config.project.name;
document.title = `${projectName} Todos Web Manager`;
```

### 7.2 优先级体系可配置
基于 todos.json 中的反馈，已实现 P0-P9 优先级体系，支持：
- P0: 最高优先级（笔记整理类任务固定为 P0）
- P1-P9: 依次降低的优先级
- 兼容性：同时支持传统 high/medium/low

配置示例：
```json
{
  "priority": {
    "enabled": true,
    "system": "P0-P9",
    "default": "P4",
    "definitions": {
      "P0": "最高优先级，立即处理",
      "P1": "核心功能、用户体验关键",
      "P2": "重要功能、用户体验优化",
      "P3": "有用的功能、体验提升",
      "P4": "常规功能、一般优化",
      "P5": "次要功能、小优化",
      "P6": "可有可无的功能",
      "P7": "锦上添花",
      "P8": "未来考虑",
      "P9": "几乎不做"
    }
  }
}
```

### 7.3 Plan 机制配置化
基于 todos.json 中的反馈，已实现 Plan 机制本质化重构：
- Plan 不再是独立状态，而是 Todo 的一个字段
- 一个 Todo 可以有 0 或 1 个 Plan
- Plan 包含：content（Markdown）、status（pending/approved/rejected）、review_comment 等

配置示例：
```json
{
  "plan": {
    "enabled": true,
    "default_enabled": false,
    "review_required": true
  }
}
```

### 7.4 主题色可配置
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

### 7.5 UI 布局可配置
基于 todos.json 中的反馈，已实现前端结构本质化重构：
- 两大主标签页：「👤 我的待办」和「🤖 AI 工作流」
- 「我的待办」包含：待审核 Plans、待审核 Tasks
- 「AI 工作流」包含：进行中、待处理、已审核 Plans、归档

配置示例：
```json
{
  "ui": {
    "layout": "two-column",
    "show_assignee_tabs": false,
    "show_all_tasks_tab": true
  }
}
```

---

## 8. Python 包化（远期）

### 8.1 创建独立的 Python 包
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

### 8.2 使用 pip 安装
```bash
pip install todos-web-manager
```

### 8.3 启动服务
```bash
todos-web-manager --config config.json
```

---

## 9. 总结

### 推荐方案
使用**方案1：配置文件方式**，因为：
1. 简单易用
2. 配置清晰可见
3. 易于迁移
4. 不需要环境变量配置

### 关键改进（基于 todos.json 反馈）

#### 技术解耦
1. ✅ 路径可配置
2. ✅ 项目名称可配置
3. ✅ 目录结构标准化
4. ✅ 易于迁移到任意新项目

#### 功能增强
5. ✅ **P0-P9 优先级体系**（笔记整理固定为 P0）
6. ✅ **Plan 机制本质化重构**（Plan 作为 Todo 字段，而非独立状态）
7. ✅ **前端结构本质化优化**（按人/AI 两大分类）
8. ✅ **功能配置化**（优先级、Plan、UI 布局均可配置）
9. ✅ **向后兼容**（同时支持传统 high/medium/low）

#### 知识与经验通用化（基于 review 反馈）
10. ✅ **规则文档分层架构**（核心规则层、模板规则层、项目规则层）
11. ✅ **经验沉淀为最佳实践**（涵盖 9 大类最佳实践）
12. ✅ **配置文件扩展**（支持 rules 和 best_practices 配置）
13. ✅ **完整迁移流程**（包含规则和最佳实践的迁移）
14. ✅ **OpenClaw 记忆体系迁移**（AGENTS.md、SOUL.md、MEMORY.md 等）

### 最新架构特性

#### 技术特性
- **数据模型**：Todo 包含 Plan 字段，Plan 不再独立
- **优先级体系**：P0-P9 精细粒度，支持传统值兼容
- **UI 布局**：两大主标签页（「👤 我的待办」、「🤖 AI 工作流」）
- **可配置化**：所有核心功能均可通过 config.json 配置

#### 知识与经验特性
- **规则分层**：核心规则（通用）→ 模板规则（可配置）→ 项目规则（定制）
- **最佳实践库**：沉淀 9 大类最佳实践（Todo 管理、Plan 机制、AI 协作、笔记整理、Git SOP、安全意识、快捷指令、任务交付、错误反思）
- **OpenClaw 记忆体系**：完整迁移 AGENTS.md、SOUL.md、MEMORY.md 等经验文件
- **灵活扩展**：新项目可基于模板快速定制自己的规则体系和记忆
- **知识迁移**：不仅迁移工具，更迁移 AI 工作流的经验和智慧，让 AI 在新项目中"继承"已有经验

