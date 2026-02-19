# Smart Task Parser Skill

智能任务解析 Skill - 基于 LLM 智能解析口述式任务，写入 todos.json

## 功能

- 智能解析口述式任务
- 基于 LLM 的智能理解
- 自动写入 todos.json
- 通过 OpenClaw 消息触发

## 使用方法

```bash
# 解析任务
python3 main.py "高优先级：把 X 文档改成 Y 风格，关联链接 Z，明天前完成"

# 使用配置文件
python3 main.py "任务文本" --config config.json
```

## 配置

```json
{
  "todos_json_path": "/path/to/todos.json",
  "workspace": "/path/to/workspace"
}
```

## 工作流程

1. 用户通过 OpenClaw 消息发送任务
2. Skill 接收消息并解析任务
3. 基于 LLM 智能理解任务
4. 结构化任务并写入 todos.json
5. 通过 OpenClaw 消息通知用户

## 下一步

- [ ] 集成 LLM 智能解析
- [ ] 实现解析结果校验
- [ ] 实现基础流程回退
- [ ] 与 Todos Web Manager 融合
