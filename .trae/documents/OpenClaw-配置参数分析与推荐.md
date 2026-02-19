# OpenClaw 配置参数分析与推荐

**日期**: 2026-02-19  
**作者**: AI  
**状态**: ✅ 分析完成

## 免责声明

⚠️ **重要安全提示**:
- 本文档中的所有敏感信息（API Key、token、secret 等）都已替换为占位符
- **永远不要把真实的敏感信息提交到公开仓库！**
- 所有配置示例都使用占位符：`YOUR_API_KEY`、`YOUR_TOKEN`、`YOUR_SECRET` 等

---

## 当前配置分析

### 1. Models 配置

**当前配置**:
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
            "reasoning": false,
            "input": ["text", "image"],
            "cost": {
              "input": 0,
              "output": 0,
              "cacheRead": 0,
              "cacheWrite": 0
            },
            "contextWindow": 200000,
            "maxTokens": 8192,
            "headers": {
              "X-Client-Request-Id": "ecs-openclaw/0212.1/i-yefw1029dsvv7taanzig"
            },
            "compat": {
              "supportsDeveloperRole": false
            }
          }
        ]
      }
    }
  }
}
```

**分析**:
- ✅ **好的配置**:
  - `mode: "merge"` - 合并模式，可以同时使用多个 provider
  - `contextWindow: 200000` - 200K 上下文窗口，非常大
  - `maxTokens: 8192` - 8K 最大输出 token
  - `input: ["text", "image"]` - 支持文本和图像输入
- ⚠️ **可优化**:
  - `reasoning: false` - 如果模型支持推理模式，可以考虑开启
  - 可以添加更多模型（如通用对话模型、推理模型等）

**推荐配置**:
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
            "reasoning": false,
            "input": ["text", "image"],
            "cost": {
              "input": 0,
              "output": 0,
              "cacheRead": 0,
              "cacheWrite": 0
            },
            "contextWindow": 200000,
            "maxTokens": 8192,
            "headers": {
              "X-Client-Request-Id": "ecs-openclaw/0212.1/i-yefw1029dsvv7taanzig"
            },
            "compat": {
              "supportsDeveloperRole": false
            }
          },
          {
            "id": "doubao-pro-32k",
            "name": "doubao-pro-32k",
            "reasoning": false,
            "input": ["text"],
            "cost": {
              "input": 0,
              "output": 0
            },
            "contextWindow": 32000,
            "maxTokens": 4096
          }
        ]
      }
    }
  }
}
```

---

### 2. Agents 配置

**当前配置**:
```json
{
  "agents": {
    "defaults": {
      "model": {
        "primary": "ark/doubao-seed-2-0-code-preview-260215"
      },
      "models": {
        "ark/doubao-seed-2-0-code-preview-260215": {}
      },
      "workspace": "/root/.openclaw/workspace",
      "compaction": {
        "mode": "safeguard"
      },
      "heartbeat": {
        "every": "30m"
      },
      "maxConcurrent": 4,
      "subagents": {
        "maxConcurrent": 8
      }
    }
  }
}
```

**分析**:
- ✅ **好的配置**:
  - `workspace: "/root/.openclaw/workspace"` - 工作目录配置正确
  - `compaction.mode: "safeguard"` - 安全的上下文压缩模式
  - `heartbeat.every: "30m"` - 30 分钟心跳间隔（合理）
  - `maxConcurrent: 4` - 最大并发任务数 4（合理）
  - `subagents.maxConcurrent: 8` - 子 agent 最大并发数 8（合理）
- ⚠️ **可优化**:
  - 可以考虑为不同任务类型配置不同的模型
  - 可以调整 `heartbeat.every` 为动态间隔（根据任务密度）

**推荐配置（保持当前配置）**:
```json
{
  "agents": {
    "defaults": {
      "model": {
        "primary": "ark/doubao-seed-2-0-code-preview-260215"
      },
      "models": {
        "ark/doubao-seed-2-0-code-preview-260215": {}
      },
      "workspace": "/root/.openclaw/workspace",
      "compaction": {
        "mode": "safeguard"
      },
      "heartbeat": {
        "every": "30m"
      },
      "maxConcurrent": 4,
      "subagents": {
        "maxConcurrent": 8
      }
    }
  }
}
```

**配置说明**:
- `compaction.mode: "safeguard"` - 安全模式，只在必要时压缩上下文
- `heartbeat.every: "30m"` - 30 分钟心跳间隔，平衡及时性和资源消耗
- `maxConcurrent: 4` - 最大并发任务数 4，避免资源耗尽
- `subagents.maxConcurrent: 8` - 子 agent 最大并发数 8，可以并行执行更多任务

---

### 3. Tools 配置

**当前配置**:
```json
{
  "tools": {
    "profile": "full"
  }
}
```

**分析**:
- ✅ **好的配置**:
  - `profile: "full"` - 启用所有工具，功能最全
- ⚠️ **可优化**:
  - 如果某些工具不需要，可以考虑使用 `"minimal"` 或自定义配置
  - 可以根据需要启用/禁用特定工具

**推荐配置（保持当前配置）**:
```json
{
  "tools": {
    "profile": "full"
  }
}
```

**配置说明**:
- `"full"` - 启用所有工具，适合我们的场景（需要完整的工具能力）
- 其他选项：
  - `"minimal"` - 只启用核心工具
  - 自定义配置：可以单独启用/禁用特定工具

---

### 4. Channels 配置

**当前配置**:
```json
{
  "channels": {
    "feishu": {
      "appId": "YOUR_APP_ID",
      "appSecret": "YOUR_APP_SECRET"
    }
  }
}
```

**分析**:
- ✅ **好的配置**:
  - Feishu 渠道配置正确，可以正常接收和发送消息
- ⚠️ **可优化**:
  - 可以添加更多渠道（如 Telegram、Discord 等）
  - 可以配置渠道优先级

**推荐配置（保持当前配置）**:
```json
{
  "channels": {
    "feishu": {
      "appId": "YOUR_APP_ID",
      "appSecret": "YOUR_APP_SECRET"
    }
  }
}
```

---

### 5. Gateway 配置

**当前配置**:
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
      "mode": "off",
      "resetOnExit": false
    }
  }
}
```

**分析**:
- ✅ **好的配置**:
  - `mode: "local"` - 本地模式，安全
  - `bind: "loopback"` - 只绑定到本地回环地址，安全
  - `auth.mode: "token"` - Token 认证，安全
  - `tailscale.mode: "off"` - 不需要 Tailscale，简化配置
- ⚠️ **安全注意**:
  - **永远不要把真实的 token 提交到公开仓库！**
  - 确保 `bind: "loopback"`，不要绑定到公网 IP

**推荐配置（保持当前配置）**:
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
      "mode": "off",
      "resetOnExit": false
    }
  }
}
```

**安全最佳实践**:
- ✅ 使用 `bind: "loopback"` - 只允许本地访问
- ✅ 使用 `auth.mode: "token"` - 启用 token 认证
- ✅ 使用强密码/随机 token
- ❌ 永远不要把 token 提交到公开仓库
- ❌ 不要绑定到公网 IP（`0.0.0.0`）

---

### 6. Plugins 配置

**当前配置**:
```json
{
  "plugins": {
    "entries": {
      "dingtalk-connector": { "enabled": true },
      "wecom": { "enabled": true },
      "qqbot": { "enabled": true },
      "ai-assistant-security-openclaw": { "enabled": false },
      "feishu": { "enabled": true }
    }
  }
}
```

**分析**:
- ✅ **好的配置**:
  - Feishu 插件启用（我们需要的）
  - 安全插件禁用（`ai-assistant-security-openclaw: false`）- 如果不需要，可以保持禁用
- ⚠️ **可优化**:
  - 如果不需要 DingTalk、WeCom、QQBot，可以禁用它们，减少资源消耗
  - 如果需要额外的安全功能，可以考虑启用 `ai-assistant-security-openclaw`

**推荐配置（精简版）**:
```json
{
  "plugins": {
    "entries": {
      "dingtalk-connector": { "enabled": false },
      "wecom": { "enabled": false },
      "qqbot": { "enabled": false },
      "ai-assistant-security-openclaw": { "enabled": false },
      "feishu": { "enabled": true }
    }
  }
}
```

**配置说明**:
- 只启用 Feishu 插件（我们当前需要的）
- 禁用其他不需要的插件，减少资源消耗
- 如果将来需要其他渠道，可以再启用

---

## 综合推荐配置

基于以上分析，我推荐以下配置：

### 推荐配置（保持当前配置，微调 plugins）

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
            "reasoning": false,
            "input": ["text", "image"],
            "cost": {
              "input": 0,
              "output": 0,
              "cacheRead": 0,
              "cacheWrite": 0
            },
            "contextWindow": 200000,
            "maxTokens": 8192,
            "headers": {
              "X-Client-Request-Id": "ecs-openclaw/0212.1/i-yefw1029dsvv7taanzig"
            },
            "compat": {
              "supportsDeveloperRole": false
            }
          }
        ]
      }
    }
  },
  "agents": {
    "defaults": {
      "model": {
        "primary": "ark/doubao-seed-2-0-code-preview-260215"
      },
      "models": {
        "ark/doubao-seed-2-0-code-preview-260215": {}
      },
      "workspace": "/root/.openclaw/workspace",
      "compaction": {
        "mode": "safeguard"
      },
      "heartbeat": {
        "every": "30m"
      },
      "maxConcurrent": 4,
      "subagents": {
        "maxConcurrent": 8
      }
    }
  },
  "tools": {
    "profile": "full"
  },
  "channels": {
    "feishu": {
      "appId": "YOUR_APP_ID",
      "appSecret": "YOUR_APP_SECRET"
    }
  },
  "gateway": {
    "port": 18789,
    "mode": "local",
    "bind": "loopback",
    "auth": {
      "mode": "token",
      "token": "YOUR_GATEWAY_TOKEN"
    },
    "tailscale": {
      "mode": "off",
      "resetOnExit": false
    }
  },
  "plugins": {
    "entries": {
      "dingtalk-connector": { "enabled": false },
      "wecom": { "enabled": false },
      "qqbot": { "enabled": false },
      "ai-assistant-security-openclaw": { "enabled": false },
      "feishu": { "enabled": true }
    }
  }
}
```

---

## 配置选项对比

### Heartbeat 间隔对比

| 间隔 | 优点 | 缺点 | 推荐场景 |
|------|------|------|----------|
| 5m | 响应及时 | 资源消耗大 | 任务密集时 |
| 15m | 平衡 | - | 通用场景 |
| 30m | 资源消耗小 | 响应稍慢 | **推荐（当前配置）** |
| 60m | 资源消耗最小 | 响应慢 | 任务稀疏时 |

### Max Concurrent 对比

| 数量 | 优点 | 缺点 | 推荐场景 |
|------|------|------|----------|
| 2 | 稳定 | 并发低 | 资源受限 |
| 4 | 平衡 | - | **推荐（当前配置）** |
| 8 | 并发高 | 资源消耗大 | 资源充足 |

---

## 安全最佳实践

### 1. 敏感信息保护

✅ **必须做**:
- 永远不要把 API Key、token、secret 等敏感信息提交到公开仓库
- 使用占位符（`YOUR_API_KEY`、`YOUR_TOKEN`）代替真实值
- 把敏感信息保存在本地配置文件中，不要提交到 git

❌ **绝对不要做**:
- 把真实的 API Key 提交到公开仓库
- 在文档中展示真实的 token
- 把敏感信息硬编码到代码中

### 2. Gateway 安全

✅ **必须做**:
- 使用 `bind: "loopback"` - 只允许本地访问
- 使用 `auth.mode: "token"` - 启用 token 认证
- 使用强密码/随机 token

❌ **绝对不要做**:
- 把 token 提交到公开仓库
- 绑定到公网 IP（`0.0.0.0`）
- 禁用认证（`auth.mode: "none"`）

### 3. Git 安全

✅ **必须做**:
- 使用 `.gitignore` 排除敏感文件
- 使用 `todo-push.sh` 和 `todo-pull.sh` 作为标准 git 操作流程
- 在 commit 前检查 `git status`

❌ **绝对不要做**:
- 把 `~/.openclaw/openclaw.json` 提交到仓库
- 把包含敏感信息的配置文件提交到仓库

---

## 总结

### 当前配置评估

**整体评价**: ✅ **当前配置已经很好，不需要大改！**

**优点**:
- ✅ Models 配置合理，200K 上下文窗口
- ✅ Agents 配置平衡，并发数合理
- ✅ Tools 配置完整，功能齐全
- ✅ Gateway 配置安全，本地模式 + token 认证
- ✅ Heartbeat 间隔合理（30m）

**建议的微调**:
- 📋 可以禁用不需要的 plugins（DingTalk、WeCom、QQBot）
- 📋 可以考虑添加更多模型（如通用对话模型）
- 📋 可以根据任务密度动态调整 heartbeat 间隔

### 推荐方案

**推荐保持当前配置，只做微小调整**：
1. ✅ 保持当前的 models、agents、tools、gateway 配置
2. 📋 可以考虑禁用不需要的 plugins（可选）
3. 📋 可以考虑添加更多模型（可选）

**不需要改的配置**：
- ❌ 不要改 heartbeat 间隔（30m 已经很好）
- ❌ 不要改 maxConcurrent（4 已经很好）
- ❌ 不要改 gateway 安全配置（已经很安全）

---

**分析完成时间**: 2026-02-19  
**安全提示**: 本文档中的所有敏感信息都已替换为占位符，永远不要把真实的敏感信息提交到公开仓库！
