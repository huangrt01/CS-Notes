---
name: todo-finder
description: 从 todos.json 和归档中以多种匹配方式查找 todo
---

# Todo Finder

## 适用场景

当需要查找 todo 时使用该技能。支持从 todos.json 和归档中以多种匹配方式查找 todo。

## 使用步骤

1. 准备查找条件（ID、标题、关键词等）
2. 运行脚本 `python scripts/todo_finder.py "<query>"`
3. 脚本将返回查找结果

## 匹配方式

- `--id <todo_id>` - 按 todo ID 精确匹配
- `--title <keyword>` - 按标题关键词匹配
- `--keyword <keyword>` - 按关键词匹配（标题、内容、链接等）
- `--status <status>` - 按状态匹配（pending/in-progress/completed）
- `--priority <priority>` - 按优先级匹配（P0-P9）
- `--assignee <assignee>` - 按负责人匹配
- `--archived` - 只查找归档中的 todo
- `--all` - 查找所有 todo（包括归档）

## 输出格式

- 列表格式：显示 todo ID、标题、状态、优先级
- 详情格式：显示 todo 的完整信息

## 示例

```bash
# 按 todo ID 精确匹配
python scripts/todo_finder.py --id todo-20260220-008

# 按标题关键词匹配
python scripts/todo_finder.py --title "Top Lean AI"

# 按关键词匹配
python scripts/todo_finder.py --keyword "调研"

# 按状态匹配
python scripts/todo_finder.py --status in-progress

# 按优先级匹配
python scripts/todo_finder.py --priority P2

# 查找归档中的 todo
python scripts/todo_finder.py --archived --keyword "调研"

# 查找所有 todo（包括归档）
python scripts/todo_finder.py --all --keyword "调研"
```
