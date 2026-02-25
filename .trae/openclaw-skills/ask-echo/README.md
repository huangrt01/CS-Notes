# Ask-Echo Web Search Skill

使用火山引擎 Ask-Echo 融合信息搜索 API 进行 web 搜索、web 搜索总结版和图片搜索的 Clawdbot 技能。

## 功能特性

- ✅ Web 信息搜索
- ✅ Web 搜索总结版（含 LLM 总结）
- ✅ 图片搜索
- ✅ 精准摘要
- ✅ 时间范围过滤
- ✅ 站点范围限制
- ✅ 权威度过滤
- ✅ Token 使用统计（web_summary）

## 安装

该 skill 已安装在 `/root/clawd/skills/ask-echo/`

## 配置

在使用前，需要配置以下环境变量：

### 1. 获取火山引擎 API Key

1. 访问 [火山引擎控制台 - 融合信息搜索](https://console.volcengine.com/ask-echo/api-key)
2. 创建 API Key 并妥善保存

### 2. 设置环境变量

在 Clawdbot 环境中设置以下变量：

```bash
export VOLCENGINE_ASK_ECHO="your_api_key"
```

或者在 `~/.clawdbot/.env` 文件中添加：
```
VOLCENGINE_ASK_ECHO=your_api_key
```

## 使用方法

### 基本用法

```bash
cd /root/clawd/skills/ask-echo
python scripts/ask_echo.py "搜索问题"
```

### Web 搜索

```bash
python scripts/ask_echo.py "北京旅游攻略" --type web
```

### Web 搜索总结版

```bash
python scripts/ask_echo.py "北京旅游攻略" --type web_summary
```

### 图片搜索

```bash
python scripts/ask_echo.py "故宫" --type image
```

## 高级参数

### 返回条数

```bash
# Web 搜索最多 50 条
python scripts/ask_echo.py "北京旅游攻略" --type web --count 20

# 图片搜索最多 5 条
python scripts/ask_echo.py "故宫" --type image --count 3
```

### 时间范围

```bash
python scripts/ask_echo.py "最新科技新闻" --type web --time-range OneWeek
```

可选值：
- `OneDay`：1 天内
- `OneWeek`：1 周内
- `OneMonth`：1 月内
- `OneYear`：1 年内
- `YYYY-MM-DD..YYYY-MM-DD`：自定义日期范围

### 精准摘要

```bash
python scripts/ask_echo.py "北京旅游攻略" --type web --need-summary
```

### 站点过滤

```bash
# 只搜索指定站点
python scripts/ask_echo.py "云计算" --type web --sites "zhihu.com|csdn.net"

# 屏蔽指定站点
python scripts/ask_echo.py "云计算" --type web --block-hosts "example.com"
```

### 仅返回有正文的结果

```bash
python scripts/ask_echo.py "深度学习教程" --type web --need-content
```

### 权威度过滤

```bash
# 仅返回非常权威的内容
python scripts/ask_echo.py "医疗健康" --type web --auth-level 1
```

## 输出示例

### Web 搜索输出

```
=== Search Results (2 items) ===

[1] 北京五日游攻略及路线，北京玩五天四晚大概多少钱?来到北京旅游必打卡的地方!
    Site: 搜狐网
    URL: https://m.sohu.com/a/905840001_122260725/
    Published: 2025-06-19T15:10:00+08:00
    Authority: 正常权威 (Level: 2)
    
    Snippet:
    北京旅游攻略北京必打卡景点1. 故宫博物院：穿越600年皇权史，感受古代建筑与艺术的巅峰。2. 八达岭长城：攀登世界奇迹，俯瞰山峦壮阔，铭记历史印记。

=== Search Info ===
  Query: 北京旅游攻略
  Type: web
  Time Cost: 372 ms
  Log ID: 202506191859387C810A0EB6D7ECB1BCCF
```

### Web 搜索总结版输出

```
=== LLM Summary ===
根据搜索结果，北京旅游的必打卡景点包括：

1. 故宫博物院：穿越600年皇权史，感受古代建筑与艺术的巅峰
2. 八达岭长城：攀登世界奇迹，俯瞰山峦壮阔
3. 天安门广场+升旗仪式：庄严的仪式感与历史厚重感
...

=== Search Results (5 items) ===
[1] 今日早报 每日热点15条新闻简报
    Site: 网易手机网
    URL: http://m.163.com/dy/article/JQ0237KG05566QF4.html
    ...

=== Token Usage ===
  Prompt tokens: 11468
  Completion tokens: 94
  Total tokens: 11562
  Search time: 120 ms
  First token time: 150 ms
  Total time: 270 ms
```

### 图片搜索输出

```
=== Image Results (2 items) ===

[1] 故宫博物院
    Site: 百度图片
    URL: https://example.com/image1.jpg
    Published: 2025-01-01T00:00:00+08:00
    
    Image Info:
      URL: https://example.com/image1.jpg
      Size: 1920x1080
      Shape: 横长方形
```

## API 限制

- 默认 5 QPS（可申请扩容）
- Web 搜索和 Web 搜索总结版各享有 5000 次免费调用额度
- Query 长度：1-100 个字符

## API 文档

详细 API 文档请参考：
- `/root/api-docs/ask-echo.md`
- [火山引擎融合信息搜索文档](https://www.volcengine.com/docs/85508/1650263?lang=zh)

## 故障排除

### 错误：MODEL_SEARCH_API_KEY environment variable is required

**解决方法**：设置 `MODEL_SEARCH_API_KEY` 环境变量

### 错误：Request failed: 401 Unauthorized

**解决方法**：检查 API Key 是否正确

### 错误：Request failed: 400 Bad Request

**解决方法**：检查查询内容是否符合要求（1-100 字符）

## 技术支持

如有问题，请参考：
- [火山引擎文档](https://www.volcengine.com/docs)
- [Clawdbot 文档](https://docs.clawd.bot)
