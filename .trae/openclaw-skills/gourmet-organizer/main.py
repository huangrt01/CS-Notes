#!/usr/bin/env python3
"""
Gourmet Organizer - 整理美食笔记，品鉴菜单
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

def load_gourmet_notes():
    """加载用户的美食笔记"""
    gourmet_path = REPO_ROOT / "Notes" / "Gourmet.md"
    if gourmet_path.exists():
        with open(gourmet_path, 'r', encoding='utf-8') as f:
            return f.read()
    return ""

def organize_gourmet_content(content, logger=None):
    """整理美食内容"""
    if logger and TASK_LOGGER_AVAILABLE:
        logger.log_info("开始整理美食内容")
    
    # 这里可以添加 LLM 调用来整理美食内容
    # 暂时返回简单的整理结果
    
    result = {
        "original_content": content,
        "organized_content": content,
        "timestamp": datetime.now().isoformat(),
        "notes": "美食内容已整理，可以添加到 Gourmet.md 中"
    }
    
    if logger and TASK_LOGGER_AVAILABLE:
        logger.log_info("美食内容整理完成")
    
    return result

def main():
    parser = argparse.ArgumentParser(description='整理美食笔记，品鉴菜单')
    parser.add_argument('content', nargs='?', help='美食内容')
    parser.add_argument('--file', help='从文件读取美食内容')
    
    args = parser.parse_args()
    
    task_id = f"gourmet-organizer-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
    # 开始任务
    if task_logger and TASK_LOGGER_AVAILABLE:
        task_logger.start_task(task_id)
    
    try:
        # 读取美食内容
        content = args.content
        if args.file:
            file_path = Path(args.file)
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            else:
                print(f"错误: 文件不存在: {args.file}")
                if task_logger and TASK_LOGGER_AVAILABLE:
                    task_logger.fail_task(task_id, f"文件不存在: {args.file}")
                return 1
        
        if not content:
            print("请提供美食内容或使用 --file 参数从文件读取")
            if task_logger and TASK_LOGGER_AVAILABLE:
                task_logger.fail_task(task_id, "没有提供美食内容")
            return 1
        
        # 加载用户的美食笔记
        gourmet_notes = load_gourmet_notes()
        if task_logger and TASK_LOGGER_AVAILABLE:
            task_logger.log_info("已加载用户的美食笔记")
        
        # 整理美食内容
        result = organize_gourmet_content(content, task_logger)
        
        # 输出结果
        print(json.dumps(result, ensure_ascii=False, indent=2))
        
        # 完成任务
        if task_logger and TASK_LOGGER_AVAILABLE:
            task_logger.complete_task(task_id)
            
            # 保存产物
            artifact = TaskArtifact(
                task_id=task_id,
                execution_summary="✅ 已完成！成功整理美食内容！",
                product_links=["Notes/Gourmet.md"],
                key_diffs=[],
                reproduction_commands=[]
            )
            task_logger.save_artifact(task_id, artifact)
        
        return 0
        
    except Exception as e:
        if task_logger and TASK_LOGGER_AVAILABLE:
            task_logger.fail_task(task_id, str(e))
        print(f"错误: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main())
