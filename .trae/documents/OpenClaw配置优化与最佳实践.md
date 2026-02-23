# OpenClaw 配置优化与最佳实践

**日期**: 2026-02-24  
**状态**: 🔄 进行中

---

## 一、当前配置评估

### 整体评价
✅ **当前配置已经很好，不需要大改！**

### 优点
1. ✅ **Models 配置合理** - 200K 上下文窗口，支持文本和图像输入
2. ✅ **Agents 配置平衡** - 并发数合理，心跳间隔 30m
3. ✅ **Tools 配置完整** - 功能齐全，使用 `profile: "full"`
4. ✅ **Gateway 配置安全** - 本地模式 + token 认证，`bind: "loopback"`
5. ✅ **Heartbeat 间隔合理** - 30m，平衡及时性和资源消耗

### 建议的微调
1. 📋 可以禁用不需要的 plugins（DingTalk、WeCom、QQBot）
2. 📋 可以考虑添加更多模型（如通用对话模型）
3. 📋 可以根据任务密度动态调整 heartbeat 间隔

---

## 二、配置详解

### Models 配置
```json
{
  "models": {
    "mode": "merge",
    "providers": {
      "ark": {
        "baseUrl": "https://ark.cn-beijing.volces.com/api/v3",
        "apiKey": "YOUR_API_KEY",
        "api": "openai-completions",
        "models": [
          {
            "id": "doubao-seed-2-0-code-preview-260215",
            "name": "doubao-seed-2-0-code-preview-260215",
            "reasoning": true,
            "input": ["text", "image"],
            "contextWindow": 200000,
            "maxTokens": 8192
          }
        ]
      }
    }
  }
}
```

### Agents 配置
```json
{
  "agents": {
    "defaults": {
      "model": {
        "primary": "ark/doubao-seed-2-0-code-preview-260215"
      },
      "workspace": "/root/.openclaw/workspace",
      "compaction": {
        "mode": "safeguard"
      },
      "blockStreamingDefault": "on",
      "blockStreamingBreak": "text_end",
      "heartbeat": {
        "every": "30m"
      },
      "maxConcurrent": 8,
      "subagents": {
        "maxConcurrent": 16
      }
    }
  }
}
```

### Gateway 配置
```json
{
  "gateway": {
    "port": 18789,
    "mode": "local",
    "bind": "loopback",
    "auth": {
      "mode": "token",
      "token": "YOUR_GATEWAY_TOKEN"
    },
    "tailscale": {
      "mode": "off"
    }
  }
}
```

---

## 三、最佳实践

### 安全最佳实践
1. ✅ 使用 `bind: "loopback"` - 只允许本地访问
2. ✅ 使用 `auth.mode: "token"` - 启用 token 认证
3. ✅ 使用强密码/随机 token
4. ❌ 永远不要把 token 提交到公开仓库
5. ❌ 不要绑定到公网 IP（`0.0.0.0`）

### Git 操作最佳实践
1. ✅ 使用 `Notes/snippets/todo-push.sh` 和 `Notes/snippets/todo-pull.sh` 作为标准 git 操作流程
2. ✅ todo-push.sh 白名单机制：仅允许 `Notes/`、`.trae/`、`创作/` 三个文件夹
3. ✅ todo-push.sh 黑名单机制：绝对禁止 `公司项目/` 文件夹
4. ✅ 在 commit 前检查 `git status`

### 任务执行最佳实践
1. ✅ 所有任务执行必须使用 task_execution_logger
2. ✅ 开始任务前：调用 `logger.start_task(task_id)`
3. ✅ 执行中：记录关键步骤日志
4. ✅ 完成任务：调用 `logger.complete_task(task_id)`
5. ✅ 沉淀产物：使用 `logger.save_artifact()` 保存执行摘要

---

## 四、不需要改的配置

1. ❌ 不要改 heartbeat 间隔（30m 已经很好）
2. ❌ 不要改 maxConcurrent（8 已经很好）
3. ❌ 不要改 gateway 安全配置（已经很安全）
4. ❌ 不要改 block streaming 配置（已经启用）

---

**文档生成时间**: 2026-02-24 03:00