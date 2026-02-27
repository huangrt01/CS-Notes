---
name: ask-echo
description: 使用火山引擎 Ask-Echo 融合信息搜索 API 进行 web 搜索、web 搜索总结版和图片搜索。
---

# Ask-Echo Web Search

## 适用场景

当需要使用火山引擎 Ask-Echo 融合信息 API 进行信息搜索时使用该技能。支持：

- **web 搜索**：常规网页搜索，返回站点信息
- **web 搜索总结版**：返回搜索结果及 LLM 总结内容
- **image 搜索**：图片搜索，返回图片信息

## 使用步骤

1. 准备清晰具体的搜索问题。
2. 运行脚本 `python scripts/ask_echo.py "<query>"`。
3. 脚本将返回搜索结果。



## 输出格式

- 输出搜索结果列表（标题、URL、摘要、正文等）
- 对于 web_summary 类型，输出 LLM 总结内容
- 显示 Token 使用情况（仅 web_summary）
- 显示搜索耗时

## 示例

```bash
# Web 搜索
python scripts/ask_echo.py "北京旅游攻略" --type web

# Web 搜索总结版
python scripts/ask_echo.py "北京旅游攻略" --type web_summary

# 图片搜索
python scripts/ask_echo.py "故宫" --type image
```

## 高级参数

- `--count`：返回条数（web 最多 50 条，image 最多 5 条）
- `--time-range`：时间范围（OneDay/OneWeek/OneMonth/OneYear）
- `--need-summary`：是否需要精准摘要
- `--need-content`：是否仅返回有正文的结果
- `--sites`：指定搜索站点范围（多个用 | 分隔）
- `--block-hosts`：屏蔽的站点（多个用 | 分隔）

## API 文档

参考：/root/api-docs/ask-echo.md
