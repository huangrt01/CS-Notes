# Todo Updater Skill

使用Todos Web Manager的API来更新todo状态，避免直接编辑todos.json导致的语法错误。

## 功能特性

- ✅ 通过Todos Web Manager的API更新todo状态
- ✅ 支持更新todo的所有字段
- ✅ 避免直接编辑todos.json导致的语法错误
- ✅ 自动处理JSON格式，确保格式正确

## 安装与使用

### 前置条件

1. 确保Todos Web Manager的server.py正在运行
2. 确保可以访问 http://localhost:5000

### 使用方法

#### 方式一：更新todo状态

```bash
python3 main.py <task_id> <new_status> [progress]
```

示例：

```bash
# 更新todo状态为completed
python3 main.py todo-20260225-008 completed

# 更新todo状态为completed，并设置进度
python3 main.py todo-20260225-008 completed "✅ 已完成！"

# 更新todo状态为in-progress
python3 main.py todo-20260225-008 in-progress
```

#### 方式二：从stdin读取JSON更新todo

```bash
cat update.json | python3 main.py <task_id>
```

update.json示例：

```json
{
  "status": "completed",
  "progress": "✅ 已完成！",
  "completed_at": "2026-02-25T18:20:00.000000"
}
```

## API说明

### 更新任务状态

- **URL**: `PUT /api/tasks/<task_id>/status`
- **参数**:
  - `status`: 新状态（pending/in-progress/completed）
  - `progress`: （可选）进度描述

### 更新任务

- **URL**: `PUT /api/tasks/<task_id>`
- **参数**: 任意todo字段

## 注意事项

1. 如果Todos Web Manager的server.py没有运行，会提示连接失败
2. 启动server.py的命令：`cd .trae/web-manager && python3 server.py`
3. 如果API调用失败，可以考虑直接编辑todos.json，但要非常小心，确保JSON语法正确

## 相关技能

- [todo-adder](../todo-adder/) - 添加todo的skill
