# OpenClaw 配置参数分析与推荐

## 📋 概述

本文档分析 OpenClaw 的各种配置参数，独立分析各种配置参数的优缺点，推荐更适合我们场景的配置参数，供用户选择。

---

## 🎯 当前配置分析

### 当前配置（`/root/.openclaw/openclaw.json`）

```json5
{
  meta: {
    lastTouchedVersion: "2026.2.15",
    lastTouchedAt: "2026-02-16T17:23:20.626Z",
  },
  models: {
    mode: "merge",
    providers: {
      ark: {
        baseUrl: "https://ark.cn-beijing.volces.com/api/v3",
        apiKey: "5c8e3162-1475-4db4-bf0b-efc3e37c340e",
        api: "openai-completions",
        models: [
          {
            id: "doubao-seed-2-0-code-preview-260215",
            name: "doubao-seed-2-0-code-preview-260215",
            reasoning: false,
            input: ["text"],
            cost: {
              input: 0,
              output: 0,
              cacheRead: 0,
              cacheWrite: 0,
            },
            contextWindow: 200000,
            maxTokens: 8192,
            headers: {
              "X-Client-Request-Id": "ecs-openclaw/0212.1/i-yefw1029dsvv7taanzig",
            },
            compat: { supportsDeveloperRole: false },
          },
        ],
      },
    },
  },
  agents: {
    defaults: {
      model: {
        primary: "ark/doubao-seed-2-0-code-preview-260215",
      },
      models: {
        "ark/doubao-seed-2-0-code-preview-260215": {},
      },
      workspace: "/root/.openclaw/workspace",
      compaction: { mode: "safeguard" },
      heartbeat: { every: "30m" },
      maxConcurrent: 4,
      subagents: { maxConcurrent: 8 },
    },
  },
  messages: {
    ackReactionScope: "group-mentions",
  },
  commands: {
    native: "auto",
    nativeSkills: "auto",
  },
  channels: {
    feishu: {
      appId: "cli_a918617f05b8dbb5",
      appSecret: "3NmBPo6YPCOBE3XVnz9fTgwSPD4AxzJv",
    },
  },
  gateway: {
    port: 18789,
    mode: "local",
    bind: "loopback",
    auth: {
      mode: "token",
      token: "59ac1f34670bb1c61a7bef9e29745b55507f0bb9170b35b1",
    },
    tailscale: {
      mode: "off",
      resetOnExit: false,
    },
  },
  plugins: {
    entries: {
      "dingtalk-connector": { enabled: true },
      wecom: { enabled: true },
      qqbot: { enabled: true },
      "ai-assistant-security-openclaw": { enabled: false },
      feishu: { enabled: true },
    },
    installs: {
      "dingtalk-connector": {
        source: "npm",
        spec: "https://github.com/DingTalk-Real-AI/dingtalk-moltbot-connector.git",
        installPath: "/root/.openclaw/extensions/dingtalk-connector",
        version: "0.6.0",
        installedAt: "2026-02-16T15:03:49.556Z",
      },
      wecom: {
        source: "npm",
        spec: "@openclaw-china/wecom@latest",
        installPath: "/root/.openclaw/extensions/wecom",
        version: "0.1.21",
        installedAt: "2026-02-16T15:03:56.576Z",
      },
      qqbot: {
        source: "path",
        sourcePath: "/root/qqbot",
        installPath: "/root/.openclaw/extensions/qqbot",
        version: "1.2.3",
        installedAt: "2026-02-03T09:14:05.915Z",
      },
      "ai-assistant-security-openclaw": {
        source: "npm",
        spec: "@omni-shield/ai-assistant-security-openclaw",
        installPath: "/root/.openclaw/extensions/ai-assistant-security-openclaw",
        version: "1.0.0",
        installedAt: "2026-02-16T15:03:58.189Z",
      },
    },
  },
}
```

---

## 📊 配置参数分析与推荐

### 1. Heartbeat 配置（用户已要求调整）

**当前配置：**
```json5
{
  agents: {
    defaults: {
      heartbeat: { every: "30m" },
    },
  },
}
```

**分析：**
- 当前 heartbeat 间隔是 30 分钟
- 用户已要求调整到 60 分钟
- 60 分钟的间隔可以减少不必要的检查，同时仍能及时发现问题

**推荐配置：**
```json5
{
  agents: {
    defaults: {
      heartbeat: { every: "60m" },
    },
  },
}
```

---

### 2. Session 配置

**当前配置：**
```json5
{
  // 没有显式配置 session，使用默认值
}
```

**分析：**
- 当前没有显式配置 session，使用默认值
- 可以配置 session 自动重置、session 维护等，避免 session 过长导致的问题

**推荐配置：**
```json5
{
  session: {
    scope: "per-sender",
    dmScope: "main", // 所有 DM 共享 main session
    reset: {
      mode: "daily", // 每天重置
      atHour: 4, // 凌晨 4 点重置
    },
    maintenance: {
      mode: "warn", // 警告模式
      pruneAfter: "30d", // 30 天后 pruning
      maxEntries: 500, // 最多 500 条
      rotateBytes: "10mb", // 10MB 后 rotate
    },
  },
}
```

---

### 3. 最大并发配置

**当前配置：**
```json5
{
  agents: {
    defaults: {
      maxConcurrent: 4, // 最多 4 个并发任务
      subagents: { maxConcurrent: 8 }, // 最多 8 个并发子 agent
    },
  },
}
```

**分析：**
- 当前配置：maxConcurrent: 4, subagents.maxConcurrent: 8
- 对于我们的场景，这个配置是合理的
- 如果需要更多并发，可以适当调整

**推荐配置：**
```json5
{
  agents: {
    defaults: {
      maxConcurrent: 4, // 保持当前配置
      subagents: { maxConcurrent: 8 }, // 保持当前配置
    },
  },
}
```

---

### 4. 工具配置

**当前配置：**
```json5
{
  // 没有显式配置 tools，使用默认值
}
```

**分析：**
- 当前没有显式配置 tools，使用默认值
- 可以配置 tools.profile、tools.allow、tools.deny 等，限制不必要的工具
- 对于我们的场景，coding 配置文件是合适的

**推荐配置：**
```json5
{
  tools: {
    profile: "coding", // coding 配置文件（包含 group:fs、group:runtime、group:sessions、group:memory、image）
    // allow: ["*"], // 允许所有工具（默认）
    // deny: ["browser", "canvas"], // 禁用某些工具
  },
}
```

---

### 5. Messages 配置

**当前配置：**
```json5
{
  messages: {
    ackReactionScope: "group-mentions",
  },
}
```

**分析：**
- 当前配置：ackReactionScope: "group-mentions" - 只在群组提及时发送确认反应
- 这个配置是合理的

**推荐配置：**
```json5
{
  messages: {
    ackReactionScope: "group-mentions", // 保持当前配置
  },
}
```

---

### 6. Compaction 配置

**当前配置：**
```json5
{
  agents: {
    defaults: {
      compaction: { mode: "safeguard" },
    },
  },
}
```

**分析：**
- 当前配置：compaction.mode: "safeguard" - 安全模式
- 这个配置是合理的，避免意外压缩

**推荐配置：**
```json5
{
  agents: {
    defaults: {
      compaction: { mode: "safeguard" }, // 保持当前配置
    },
  },
}
```

---

### 7. Gateway 配置

**当前配置：**
```json5
{
  gateway: {
    port: 18789,
    mode: "local",
    bind: "loopback",
    auth: {
      mode: "token",
      token: "59ac1f34670bb1c61a7bef9e29745b55507f0bb9170b35b1",
    },
    tailscale: {
      mode: "off",
      resetOnExit: false,
    },
  },
}
```

**分析：**
- 当前配置：mode: "local", bind: "loopback" - 只在本地访问
- 这个配置是安全的，避免外部访问
- 如果需要从外部访问，可以配置 Tailscale 或调整 bind 地址

**推荐配置：**
```json5
{
  gateway: {
    port: 18789,
    mode: "local",
    bind: "loopback", // 保持当前配置，只在本地访问
    auth: {
      mode: "token",
      token: "59ac1f34670bb1c61a7bef9e29745b55507f0bb9170b35b1",
    },
    tailscale: {
      mode: "off", // 保持当前配置
      resetOnExit: false,
    },
  },
}
```

---

## 🎁 完整推荐配置

```json5
{
  meta: {
    lastTouchedVersion: "2026.2.15",
    lastTouchedAt: "2026-02-16T17:23:20.626Z",
  },
  models: {
    mode: "merge",
    providers: {
      ark: {
        baseUrl: "https://ark.cn-beijing.volces.com/api/v3",
        apiKey: "5c8e3162-1475-4db4-bf0b-efc3e37c340e",
        api: "openai-completions",
        models: [
          {
            id: "doubao-seed-2-0-code-preview-260215",
            name: "doubao-seed-2-0-code-preview-260215",
            reasoning: false,
            input: ["text"],
            cost: {
              input: 0,
              output: 0,
              cacheRead: 0,
              cacheWrite: 0,
            },
            contextWindow: 200000,
            maxTokens: 8192,
            headers: {
              "X-Client-Request-Id": "ecs-openclaw/0212.1/i-yefw1029dsvv7taanzig",
            },
            compat: { supportsDeveloperRole: false },
          },
        ],
      },
    },
  },
  agents: {
    defaults: {
      model: {
        primary: "ark/doubao-seed-2-0-code-preview-260215",
      },
      models: {
        "ark/doubao-seed-2-0-code-preview-260215": {},
      },
      workspace: "/root/.openclaw/workspace",
      compaction: { mode: "safeguard" },
      heartbeat: { every: "60m" }, // ⚠️ 已调整：从 30m 改为 60m
      maxConcurrent: 4,
      subagents: { maxConcurrent: 8 },
    },
  },
  session: {
    // ⚠️ 新增：session 配置
    scope: "per-sender",
    dmScope: "main",
    reset: {
      mode: "daily",
      atHour: 4,
    },
    maintenance: {
      mode: "warn",
      pruneAfter: "30d",
      maxEntries: 500,
      rotateBytes: "10mb",
    },
  },
  tools: {
    // ⚠️ 新增：tools 配置
    profile: "coding",
  },
  messages: {
    ackReactionScope: "group-mentions",
  },
  commands: {
    native: "auto",
    nativeSkills: "auto",
  },
  channels: {
    feishu: {
      appId: "cli_a918617f05b8dbb5",
      appSecret: "3NmBPo6YPCOBE3XVnz9fTgwSPD4AxzJv",
    },
  },
  gateway: {
    port: 18789,
    mode: "local",
    bind: "loopback",
    auth: {
      mode: "token",
      token: "59ac1f34670bb1c61a7bef9e29745b55507f0bb9170b35b1",
    },
    tailscale: {
      mode: "off",
      resetOnExit: false,
    },
  },
  plugins: {
    entries: {
      "dingtalk-connector": { enabled: true },
      wecom: { enabled: true },
      qqbot: { enabled: true },
      "ai-assistant-security-openclaw": { enabled: false },
      feishu: { enabled: true },
    },
    installs: {
      "dingtalk-connector": {
        source: "npm",
        spec: "https://github.com/DingTalk-Real-AI/dingtalk-moltbot-connector.git",
        installPath: "/root/.openclaw/extensions/dingtalk-connector",
        version: "0.6.0",
        installedAt: "2026-02-16T15:03:49.556Z",
      },
      wecom: {
        source: "npm",
        spec: "@openclaw-china/wecom@latest",
        installPath: "/root/.openclaw/extensions/wecom",
        version: "0.1.21",
        installedAt: "2026-02-16T15:03:56.576Z",
      },
      qqbot: {
        source: "path",
        sourcePath: "/root/qqbot",
        installPath: "/root/.openclaw/extensions/qqbot",
        version: "1.2.3",
        installedAt: "2026-02-03T09:14:05.915Z",
      },
      "ai-assistant-security-openclaw": {
        source: "npm",
        spec: "@omni-shield/ai-assistant-security-openclaw",
        installPath: "/root/.openclaw/extensions/ai-assistant-security-openclaw",
        version: "1.0.0",
        installedAt: "2026-02-16T15:03:58.189Z",
      },
    },
  },
}
```

---

## 📝 需要用户选择的配置

| 配置项 | 当前值 | 推荐值 | 说明 |
|--------|--------|--------|------|
| `heartbeat.every` | `30m` | `60m` | Heartbeat 间隔（用户已要求调整） |
| `session.reset.mode` | （默认） | `daily` | Session 重置模式 |
| `session.reset.atHour` | （默认） | `4` | 每天重置时间（凌晨 4 点） |
| `session.maintenance.mode` | （默认） | `warn` | Session 维护模式 |
| `tools.profile` | （默认） | `coding` | 工具配置文件 |

---

## 🎯 下一步

1. 用户查看本文档，确认推荐的配置参数
2. 如果同意，更新 `/root/.openclaw/openclaw.json`
3. 重启 OpenClaw Gateway 使配置生效

