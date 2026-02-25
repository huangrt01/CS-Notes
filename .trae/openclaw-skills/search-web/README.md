# Search Web Skill

使用火山引擎联网问答智能体 API 进行网络搜索和问答。

## 功能特性

- ✅ 联网搜索并获取智能体回答
- ✅ 支持参考来源展示（URL、标题、发布时间）
- ✅ 支持追问建议
- ✅ 支持流式和非流式响应
- ✅ Token 使用统计
- ✅ 高级功能：引用角标、图文混排、百科划线词等

## 环境变量配置

| 环境变量 | 必需 | 说明 |
|---------|------|------|
|
SEARCH_BOT_ID | 是 | 智能体 ID，在[控制台](https://console.volcengine.com/ask-echo/my-agent)创建智能体后获取 |
| MODEL_SEARCH_API_KEY | 否* | 搜索 API Key |
| ARK_API_KEY | 否* | 通用 Ark API Key |

* 至少需要配置其中一个 API Key

## 使用方法

### 基本用法

```bash
cd /root/clawd/skills/search-web
python scripts/search_web.py "荣耀手机的最新动态"
```

### 流式输出

```bash
python scripts/search_web.py "推荐一些好看的电影" --stream
```

## 输出示例

```
=== Response ===

荣耀手机最新动态汇总...

=== References ===

[1] 荣耀Magic7系列
    URL: https://example.com/...
    Source: 搜索引擎
    Published: 2025-01-15 10:30:00

=== Suggested Follow-up Questions ===
1. 荣耀Magic8系列发布时间
2. MagicOS 10.0 Beta推送机型

=== Token Usage ===
  Prompt tokens: 6673
  Completion tokens: 854
  Total tokens: 7527
```

## API 文档

详细 API 文档请参考：`/root/api-docs/search-agent.md`
