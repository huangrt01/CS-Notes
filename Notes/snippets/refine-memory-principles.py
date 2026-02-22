#!/usr/bin/env python3
"""
提炼最近 3 天的 memory 中的原则和最佳实践到 MEMORY.md
"""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# 配置
WORKSPACE = Path("/root/.openclaw/workspace")
MEMORY_DIR = WORKSPACE / "memory"
MEMORY_MD = WORKSPACE / "MEMORY.md"
CS_NOTES = WORKSPACE / "CS-Notes"

def get_recent_memory_files(days=3):
    """获取最近几天的 memory 文件"""
    if not MEMORY_DIR.exists():
        return []
    
    files = []
    cutoff_date = datetime.now() - timedelta(days=days)
    
    for file in MEMORY_DIR.glob("*.md"):
        try:
            # 从文件名提取日期
            date_str = file.stem
            file_date = datetime.strptime(date_str, "%Y-%m-%d")
            if file_date >= cutoff_date:
                files.append(file)
        except ValueError:
            continue
    
    return sorted(files)

def extract_principles_from_file(file_path):
    """从 memory 文件中提取原则和最佳实践"""
    content = file_path.read_text(encoding="utf-8")
    lines = content.split("\n")
    
    principles = []
    in_principle = False
    current_section = ""
    
    for line in lines:
        # 识别原则相关的标题
        if any(keyword in line.lower() for keyword in [
            "原则", "最佳实践", "经验", "规范", "流程", "规则",
            "principle", "best practice", "experience", "rule"
        ]):
            if line.startswith("#"):
                in_principle = True
                current_section = line.strip()
                principles.append(current_section)
            continue
        
        if in_principle:
            # 收集内容直到下一个大标题
            if line.startswith("##") and not any(keyword in line.lower() for keyword in [
                "原则", "最佳实践", "经验", "规范", "流程", "规则"
            ]):
                in_principle = False
                continue
            if line.strip():
                principles.append(line)
    
    return "\n".join(principles)

def refine_memory_md():
    """精炼 MEMORY.md"""
    print(f"📝 开始提炼最近 3 天的 memory 原则...")
    
    # 获取最近 3 天的 memory 文件
    recent_files = get_recent_memory_files(3)
    print(f"📂 找到 {len(recent_files)} 个最近的 memory 文件")
    
    for file in recent_files:
        print(f"  - {file.name}")
    
    # 读取当前的 MEMORY.md
    if MEMORY_MD.exists():
        current_content = MEMORY_MD.read_text(encoding="utf-8")
    else:
        current_content = ""
    
    # 这里需要 AI 来做实际的提炼工作
    # 我们通过 OpenClaw 的 message 功能来触发 AI 执行
    print("🤖 请 AI 助手提炼原则和最佳实践...")
    
    # 构建消息
    message = """请将最近 3 天的 memory 文件中，对这个笔记仓库适用的原则、最佳实践，非常精炼提炼到 MEMORY.md。

最近的 memory 文件：
"""
    
    for file in recent_files:
        message += f"- {file.name}\n"
    
    message += """

请按照之前的格式，将原则和最佳实践精炼地整理到 MEMORY.md 中。
"""
    
    print("✅ 任务准备完成！")
    print("📋 下一步：请 AI 助手执行提炼任务")
    
    return True

if __name__ == "__main__":
    try:
        success = refine_memory_md()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
