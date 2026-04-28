---
name: arxiv-paper-reader
description: "Fetch and extract full text from arxiv paper HTML pages. Invoke when user asks to read an arxiv paper, analyze a paper's content, or when given an arxiv ID/URL."
---

# Arxiv Paper Reader

从 arxiv 论文的 HTML 页面抓取并提取全文纯文本。arxiv 的 HTML 版本比 PDF 更容易程序化提取，且包含完整内容。

## 适用场景

- 用户给出 arxiv 论文 ID 或 URL，要求阅读/分析论文内容
- 需要获取论文的完整文本用于笔记整理、技术分析
- WebFetch 工具无法直接访问 arxiv 页面时的替代方案

## 核心原理

arxiv 论文有两种可读格式：
1. **abs 页面**：`https://arxiv.org/abs/<ID>` — 只有摘要
2. **HTML 页面**：`https://arxiv.org/html/<ID>v1` — 完整论文文本（LaTeXML 渲染）

HTML 页面是完整论文的网页版，可通过 curl 获取后提取纯文本。这比 PDF 解析更可靠。

## 使用步骤

1. 从用户输入中提取 arxiv ID（如 `2507.02259`）
2. 运行脚本：

```bash
python .trae/skills/arxiv-paper-reader/scripts/arxiv_reader.py <arxiv_id_or_url>
```

3. 可选参数：
   - `--output FILE`：保存到文件（默认输出到 stdout）
   - `--sections`：添加章节分隔符
   - `--raw`：跳过数学符号清理

## 示例

```bash
# 基本用法
python .trae/skills/arxiv-paper-reader/scripts/arxiv_reader.py 2507.02259

# 从 URL 提取
python .trae/skills/arxiv-paper-reader/scripts/arxiv_reader.py https://arxiv.org/abs/2507.02259

# 保存到文件并添加章节分隔
python .trae/skills/arxiv-paper-reader/scripts/arxiv_reader.py 2507.02259 --output /tmp/paper.txt --sections

# 读取后用 Read 工具分段查看
python .trae/skills/arxiv-paper-reader/scripts/arxiv_reader.py 2507.02259 --output /tmp/paper.txt
# 然后 Read /tmp/paper.txt
```

## 输出说明

- 纯文本格式，保留论文的章节结构
- 数学公式会被简化清理（LaTeXML 渲染的 LaTeX 标记会被部分还原）
- 包含标题、摘要、正文、参考文献等完整内容

## 注意事项

- 依赖 `curl` 命令（macOS/Linux 默认可用）
- 部分论文可能没有 HTML 版本（较老的论文），此时脚本会报错
- 版本号默认使用 v1，如需其他版本可手动构造 URL
- 对于超长论文，建议 `--output` 保存到文件后用 Read 工具分段查看
