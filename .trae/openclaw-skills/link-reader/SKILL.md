---
name: link-reader
description: 使用内置 link_reader 函数读取网页、PDF或抖音视频内容。准备 URL 列表。运行脚本 `python scripts/link_reader.py "url1" "url2" ...`。
license: Complete terms in LICENSE.txt
---

# Link Reader

## 适用场景

当需要获取网页、PDF 或抖音视频的标题和正文内容时，使用该技能调用 `link_reader` 函数。

## 使用步骤

1. 准备 URL 列表。
2. 运行脚本 `python scripts/link_reader.py "url1" "url2" ...`。运行之前cd到对应的目录。
3. 返回结果包含每个 URL 的标题和内容。

## 输出格式

- JSON 格式的列表，每个元素包含 URL 对应的标题和内容。

## 示例

```bash
python scripts/link_reader.py "https://example.com"
```
