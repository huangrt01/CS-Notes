#!/usr/bin/env python3
"""
PDF Parser Skill - 使用 PyMuPDF 解析 PDF 文件
支持解析为 Markdown、JSON、提取图片和表格
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

# 添加 snippets 目录到 sys.path，以便导入 task_execution_logger
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "Notes" / "snippets"))

try:
    from task_execution_logger import (
        TaskExecutionLogger,
        TaskStage,
        LogLevel,
        TaskArtifact,
        create_logger
    )
    TASK_LOGGER_AVAILABLE = True
except ImportError:
    TASK_LOGGER_AVAILABLE = False

# 配置路径 - 支持多路径检测
REPO_ROOT_CANDIDATES = [
    Path("/root/.openclaw/workspace/CS-Notes"),
    Path("/Users/bytedance/CS-Notes"),
    Path(__file__).parent.parent.parent
]

REPO_ROOT = None
for candidate in REPO_ROOT_CANDIDATES:
    if candidate.exists():
        REPO_ROOT = candidate
        break

if REPO_ROOT is None:
    REPO_ROOT = Path.cwd()

# 初始化任务日志系统
task_logger = None
if TASK_LOGGER_AVAILABLE:
    try:
        task_logger = create_logger(REPO_ROOT)
    except Exception as e:
        print(f"⚠️ 初始化任务日志系统失败: {e}")

# 导入 pdf_parser
sys.path.insert(0, str(REPO_ROOT / "Notes" / "snippets"))
try:
    import fitz  # PyMuPDF
    from pdf_parser import extract_markdown, extract_json, extract_images, extract_tables_basic
    PDF_PARSER_AVAILABLE = True
except ImportError:
    PDF_PARSER_AVAILABLE = False


class PDFParserSkill:
    """PDF 解析器 Skill"""
    
    def __init__(self):
        self.repo_root = REPO_ROOT
    
    def parse_pdf(self, pdf_path: str, output_format: str = "md", 
                 extract_images: bool = False, extract_tables: bool = False,
                 lang: str = "en", outroot: str = "./pdf-output") -> dict:
        """
        解析 PDF 文件
        
        参数:
            pdf_path: PDF 文件路径
            output_format: 输出格式 (md, json, both)
            extract_images: 是否提取图片
            extract_tables: 是否提取表格
            lang: 语言
            outroot: 输出根目录
        
        返回:
            解析结果字典
        """
        pdf_path_obj = Path(pdf_path)
        if not pdf_path_obj.exists():
            return {
                "success": False,
                "error": f"PDF 文件不存在: {pdf_path}"
            }
        
        if not PDF_PARSER_AVAILABLE:
            return {
                "success": False,
                "error": "PDF 解析器不可用，请确保已安装 PyMuPDF"
            }
        
        try:
            outdir = Path(outroot) / pdf_path_obj.stem
            outdir.mkdir(parents=True, exist_ok=True)
            
            with fitz.open(pdf_path_obj) as doc:
                result = {
                    "success": True,
                    "pdf_path": str(pdf_path_obj),
                    "output_dir": str(outdir),
                    "page_count": len(doc)
                }
                
                # 解析为 Markdown
                if output_format in ("md", "both"):
                    md = extract_markdown(doc)
                    md_file = outdir / "output.md"
                    md_file.write_text(md, encoding="utf-8")
                    result["markdown_file"] = str(md_file)
                    result["markdown_preview"] = md[:500] + "..." if len(md) > 500 else md
                
                # 解析为 JSON
                if output_format in ("json", "both"):
                    data = extract_json(doc, lang)
                    json_file = outdir / "output.json"
                    json_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
                    result["json_file"] = str(json_file)
                
                # 提取图片
                if extract_images:
                    img_dir = outdir / "images"
                    img_dir.mkdir(exist_ok=True)
                    img_count = extract_images(doc, img_dir)
                    result["images_dir"] = str(img_dir)
                    result["image_count"] = img_count
                
                # 提取表格
                if extract_tables:
                    tables = extract_tables_basic(doc)
                    tables_file = outdir / "tables.json"
                    tables_file.write_text(json.dumps(tables, ensure_ascii=False, indent=2), encoding="utf-8")
                    result["tables_file"] = str(tables_file)
                
                return result
        
        except Exception as e:
            return {
                "success": False,
                "error": f"解析 PDF 失败: {type(e).__name__}: {e}"
            }


def handle_parse():
    """处理解析 PDF 命令"""
    task_id = f"pdf-parser-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    agent = "openclaw"
    
    # 记录任务开始
    if TASK_LOGGER_AVAILABLE and task_logger:
        try:
            task_logger.start_task(task_id, agent=agent)
            task_logger.log_info(
                task_id,
                TaskStage.PLANNING,
                "开始解析 PDF",
                {},
                agent=agent
            )
        except Exception as e:
            print(f"⚠️ 记录任务日志失败: {e}")
    
    parser = argparse.ArgumentParser(description="PDF Parser Skill")
    parser.add_argument("pdf", help="Path to PDF file")
    parser.add_argument("--format", default="md", choices=["md", "json", "both"], 
                       help="Output format (default: md)")
    parser.add_argument("--images", action="store_true", help="Extract images")
    parser.add_argument("--tables", action="store_true", help="Extract simple tables")
    parser.add_argument("--lang", default="en", help="Language (default: en)")
    parser.add_argument("--outroot", default="./pdf-output", help="Output root directory")
    
    # 解析参数
    if len(sys.argv) < 2:
        error_msg = "Missing PDF file path"
        if TASK_LOGGER_AVAILABLE and task_logger:
            try:
                task_logger.fail_task(task_id, error_msg)
            except Exception as e:
                print(f"⚠️ 记录任务失败日志失败: {e}")
        print("Usage: python main.py parse <pdf_path> [--format md|json|both] [--images] [--tables] [--lang en] [--outroot ./pdf-output]")
        return json.dumps({"success": False, "error": error_msg})
    
    # 移除 "parse" 命令
    if sys.argv[1] == "parse":
        sys.argv = sys.argv[:1] + sys.argv[2:]
    
    try:
        args = parser.parse_args()
        
        if TASK_LOGGER_AVAILABLE and task_logger:
            task_logger.log_info(
                task_id,
                TaskStage.EXECUTING,
                "解析 PDF 文件",
                {"pdf_path": args.pdf, "format": args.format},
                agent=agent
            )
        
        skill = PDFParserSkill()
        result = skill.parse_pdf(
            args.pdf,
            output_format=args.format,
            extract_images=args.images,
            extract_tables=args.tables,
            lang=args.lang,
            outroot=args.outroot
        )
        
        if TASK_LOGGER_AVAILABLE and task_logger:
            if result.get("success"):
                task_logger.log_success(
                    task_id,
                    TaskStage.COMPLETED,
                    "PDF 解析成功",
                    {"page_count": result.get("page_count")},
                    agent=agent
                )
                task_logger.complete_task(task_id, agent=agent)
            else:
                task_logger.fail_task(task_id, result.get("error", "Unknown error"))
        
        return json.dumps(result, ensure_ascii=False)
    
    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        if TASK_LOGGER_AVAILABLE and task_logger:
            try:
                task_logger.fail_task(task_id, error_msg)
            except Exception as log_e:
                print(f"⚠️ 记录任务失败日志失败: {log_e}")
        return json.dumps({"success": False, "error": error_msg})


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "parse":
        print(handle_parse())
    else:
        print("Usage: python main.py parse <pdf_path> [options]")
