# Markdown TOC Skill

## 功能
提取 Markdown 文件的目录层级结构

## 使用方法

### 命令行模式
```bash
cd /path/to/markdown-toc
python3 main.py /path/to/markdown/file.md
```

### OpenClaw Skill 模式
```json
{
  "command": "extract",
  "args": {
    "file_path": "/path/to/markdown/file.md"
  }
}
```

## 输出示例
```
目录结构：
--------------------------------------------------
# AI-Applied-Algorithms (line 1)
  ## RAG (line 7)
    ### Intro (line 9)
      #### 为什么需要 (line 11)
```

## 适用场景
- 整理笔记时，先使用markdown-toc skill查看文件结构，找到最合适的已有的笔记
- 找到最合适的 section，而不是随便找个地方就放
- 将内容整合到合适的 section 中，而不是创建新文件
