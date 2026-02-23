# PDF Parser Skill

使用 PyMuPDF 解析 PDF 文件，支持解析为 Markdown、JSON、提取图片和表格。

## 功能特性

- ✅ 解析 PDF 为 Markdown 格式
- ✅ 解析 PDF 为 JSON 格式
- ✅ 提取 PDF 中的图片
- ✅ 提取 PDF 中的表格（基础版本）
- ✅ 集成 task_execution_logger，记录执行日志
- ✅ 支持多语言（lang 参数）

## 安装依赖

```bash
pip3 install PyMuPDF
```

## 使用方法

### 基本用法

```bash
# 解析为 Markdown
python3 main.py parse ./document.pdf

# 解析为 JSON
python3 main.py parse ./document.pdf --format json

# 同时解析为 Markdown 和 JSON
python3 main.py parse ./document.pdf --format both

# 提取图片
python3 main.py parse ./document.pdf --images

# 提取表格
python3 main.py parse ./document.pdf --tables

# 完整示例
python3 main.py parse ./document.pdf --format both --images --tables --lang zh --outroot ./my-output
```

### 参数说明

- `pdf`: PDF 文件路径（必需）
- `--format`: 输出格式，可选 `md`、`json`、`both`（默认：`md`）
- `--images`: 是否提取图片（默认：否）
- `--tables`: 是否提取表格（默认：否）
- `--lang`: 语言，可选 `en`、`zh` 等（默认：`en`）
- `--outroot`: 输出根目录（默认：`./pdf-output`）

## 输出文件

解析完成后，会在输出目录中生成以下文件：

```
pdf-output/
└── document/
    ├── output.md          # Markdown 格式输出
    ├── output.json        # JSON 格式输出
    ├── tables.json        # 表格数据（如果启用 --tables）
    └── images/            # 图片目录（如果启用 --images）
        ├── page-1-img-1.png
        ├── page-1-img-2.png
        └── ...
```

## 集成到 OpenClaw

这个 skill 已经集成了 task_execution_logger，可以记录任务执行日志。在 OpenClaw 中使用时，会自动记录执行过程。

## 示例

### 示例 1: 解析论文 PDF

```bash
python3 main.py parse ./paper.pdf --format both --images
```

### 示例 2: 解析中文 PDF

```bash
python3 main.py parse ./chinese-document.pdf --format md --lang zh
```

## 注意事项

- PyMuPDF 的表格提取功能是基础版本，对于复杂表格可能效果有限
- 提取图片可能会占用较多磁盘空间
- 大文件解析可能需要较长时间

## 依赖

- Python 3.7+
- PyMuPDF (fitz)

---

**Created:** 2026-02-23  
**Updated:** 2026-02-23
