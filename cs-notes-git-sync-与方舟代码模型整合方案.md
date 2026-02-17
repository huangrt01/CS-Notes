# cs-notes-git-sync 与方舟代码模型整合方案

## 概述

本文档分析 cs-notes-git-sync skill 的工作流，并设计与方舟代码模型的整合方案。

## cs-notes-git-sync Skill 工作流分析

### 1. Skill 功能概述

cs-notes-git-sync Skill 的核心功能：
- 接收消息（如 Lark 消息）并解析为 todo 格式
- 克隆/拉取 CS-Notes 仓库
- 写入 INBOX.md 并自动 commit & push

### 2. 工作流步骤

#### 步骤 1：确保仓库存在
```python
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
```

#### 步骤 2：拉取最新代码
```python
def git_pull():
    """拉取最新代码"""
    print("Pulling latest changes...")
    code, stdout, stderr = run_command(["git", "pull"], cwd=REPO_PATH)
    if code != 0:
        print(f"Pull failed: {stderr}")
        return False
    print("Pull successful")
    return True
```

#### 步骤 3：解析消息为 todo 格式
```python
def parse_message_to_todo(message: str) -> str:
    """将消息解析为 todo 格式"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = []
    lines.append(f"- [{timestamp}] {message.strip()}")
    lines.append("  - Priority：medium")
    lines.append("  - Links：")
    lines.append("  - Due：")
    return "\n".join(lines)
```

#### 步骤 4：写入 INBOX.md
```python
def write_to_inbox(todo_content: str):
    """写入 INBOX.md"""
    INBOX_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    if not INBOX_PATH.exists():
        INBOX_PATH.write_text("# INBOX\n\n", encoding="utf-8")
    
    with open(INBOX_PATH, "a", encoding="utf-8") as f:
        f.write("\n")
        f.write(todo_content)
        f.write("\n")
```

#### 步骤 5：Commit & Push
```python
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
```

### 3. 完整工作流流程图

```
用户发送消息（Lark/Telegram/WebChat）
        ↓
OpenClaw Gateway 接收消息
        ↓
cs-notes-git-sync Skill 被触发
        ↓
确保仓库存在（不存在则克隆）
        ↓
拉取最新代码（git pull）
        ↓
解析消息为 todo 格式
        ↓
写入 INBOX.md
        ↓
Commit & Push（git add → git commit → git push）
        ↓
完成！
```

---

## 与方舟代码模型的整合方案设计

### 1. 整合目标

设计一个工作流，实现：
- **Lark 消息 → INBOX.md → 方舟代码模型执行 → Git 同步**

### 2. 整合工作流设计

#### 方案一：双 Skill 方案（推荐）

**架构**：
- **Skill 1：cs-notes-git-sync**（已存在）
  - 功能：接收 Lark 消息，写入 INBOX.md，Git 同步
- **Skill 2：cs-notes-todo-executor**（新增）
  - 功能：监控 INBOX.md，提取新任务，调用方舟代码模型执行

**工作流**：
```
Lark 消息
    ↓
cs-notes-git-sync Skill
    ↓
写入 INBOX.md
    ↓
Git commit & push
    ↓
cs-notes-todo-executor Skill（定期轮询或 Git webhook 触发）
    ↓
检测到新任务
    ↓
调用方舟代码模型执行任务
    ↓
执行完成后，更新 todo 状态
    ↓
Git commit & push
```

#### 方案二：单 Skill 增强方案

**架构**：
- **增强 cs-notes-git-sync Skill**
  - 在现有功能基础上，增加任务执行功能

**工作流**：
```
Lark 消息
    ↓
cs-notes-git-sync Skill
    ↓
写入 INBOX.md
    ↓
判断是否需要立即执行
    ↓
如果需要，调用方舟代码模型执行
    ↓
Git commit & push
```

### 3. 推荐方案：方案一（双 Skill 方案）

**理由**：
1. **职责分离**：cs-notes-git-sync 负责接收消息和 Git 同步，cs-notes-todo-executor 负责任务执行
2. **可维护性**：两个 Skill 职责清晰，易于维护和扩展
3. **灵活性**：可以独立启动/停止某个 Skill
4. **可观测性**：两个 Skill 可以独立记录日志和状态

---

## cs-notes-todo-executor Skill 设计

### 1. Skill 功能

- 监控 INBOX.md 的变化
- 提取新添加的任务
- 判断任务是否需要 AI 执行（Assignee: AI）
- 调用方舟代码模型执行任务
- 执行完成后，更新 todo 状态
- Git commit & push

### 2. 目录结构

```
cs-notes-todo-executor/
├── skill.json          # Skill 配置文件
├── main.py             # 主程序
├── executor.py         # 任务执行器
├── git_utils.py        # Git 工具
└── README.md           # 说明文档
```

### 3. 核心功能模块

#### a) INBOX.md 监控
- 定期轮询 INBOX.md 的修改时间
- 或者使用 Git webhook 触发
- 检测新添加的任务

#### b) 任务解析
- 解析 INBOX.md 中的任务格式
- 提取任务的 Assignee、Priority 等字段
- 判断是否需要 AI 执行

#### c) 方舟代码模型调用
- 调用 OpenClaw 的编码能力
- 执行任务
- 记录执行结果

#### d) Todo 状态更新
- 更新 todo 的状态（进行中/已完成）
- 记录执行产物（摘要/链接/diff/复现命令）

#### e) Git 同步
- Git add
- Git commit
- Git push

---

## 实施计划

### Phase 1：分析与设计（已完成）
- ✅ 分析 cs-notes-git-sync skill 的工作流
- ✅ 设计与方舟代码模型的整合方案

### Phase 2：创建 cs-notes-todo-executor Skill
- [ ] 创建 Skill 目录结构
- [ ] 实现 INBOX.md 监控功能
- [ ] 实现任务解析功能
- [ ] 实现方舟代码模型调用功能
- [ ] 实现 Todo 状态更新功能
- [ ] 实现 Git 同步功能

### Phase 3：测试与验证
- [ ] 端到端测试：Lark 消息 → INBOX.md → 方舟代码模型执行 → Git 同步
- [ ] 验证任务执行结果
- [ ] 验证 Git 同步结果

### Phase 4：文档与优化
- [ ] 编写使用文档
- [ ] 优化性能
- [ ] 添加错误处理

---

## 总结

### cs-notes-git-sync Skill 的工作流
1. 确保仓库存在
2. 拉取最新代码
3. 解析消息为 todo 格式
4. 写入 INBOX.md
5. Commit & Push

### 与方舟代码模型的整合方案
- **推荐方案**：双 Skill 方案
- **Skill 1**：cs-notes-git-sync（已存在）
- **Skill 2**：cs-notes-todo-executor（新增）
- **工作流**：Lark 消息 → INBOX.md → 方舟代码模型执行 → Git 同步
