
# Todo Adder Skill

使用Todos Web Manager的API来添加todo，避免直接编辑todos.json导致的语法错误。

## 使用方法

### 基础用法

```bash
cd /root/.openclaw/workspace/CS-Notes/.trae/openclaw-skills/todo-adder
python3 main.py "测试todo" P2 ai
```

### 从stdin读取JSON

```bash
cat todo.json | python3 main.py
```

### JSON格式示例

```json
{
  "title": "测试todo",
  "status": "pending",
  "priority": "P2",
  "assignee": "ai",
  "feedback_required": false,
  "created_at": "2026-02-24T20:35:00.000000",
  "links": [],
  "definition_of_done": [],
  "progress": ""
}
```

## 前置条件

确保Todos Web Manager的server.py正在运行：

```bash
cd /root/.openclaw/workspace/CS-Notes/.trae/web-manager
python3 server.py
```

## 为什么使用这个Skill？

直接编辑todos.json容易导致JSON语法错误，使用Todos Web Manager的API可以：
- ✅ 避免JSON语法错误
- ✅ 自动生成正确的任务ID
- ✅ 自动记录任务创建日志
- ✅ 更安全、更可靠

