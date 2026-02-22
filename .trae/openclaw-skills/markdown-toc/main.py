
#!/usr/bin/env python3
"""
Markdown TOC Skill for OpenClaw
提取 Markdown 文件的目录层级结构
"""

import sys
import re
import json


def extract_toc(file_path):
    """
    提取 Markdown 文件的目录
    
    Args:
        file_path: Markdown 文件路径
        
    Returns:
        目录列表，每个元素为 (层级, 标题, 行号)
    """
    toc = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        for line_num, line in enumerate(lines, start=1):
            # 匹配标题行
            match = re.match(r'^(#{1,6})\s+(.*)$', line.strip())
            if match:
                level = len(match.group(1))
                title = match.group(2).strip()
                toc.append((level, title, line_num))
                
    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        
    return toc


def format_toc(toc):
    """
    格式化目录为字符串
    
    Args:
        toc: 目录列表
        
    Returns:
        格式化后的目录字符串
    """
    if not toc:
        return "未找到标题"
        
    result = "目录结构：\n"
    result += "-" * 50 + "\n"
    
    for level, title, line_num in toc:
        indent = "  " * (level - 1)
        marker = "#" * level
        result += f"{indent}{marker} {title} (line {line_num})\n"
        
    return result


def handle_extract(args):
    """
    处理 extract 命令
    
    Args:
        args: 命令参数，包含 file_path
        
    Returns:
        包含目录的字典
    """
    file_path = args.get("file_path")
    
    if not file_path:
        return {
            "success": False,
            "error": "缺少 file_path 参数"
        }
        
    toc = extract_toc(file_path)
    
    return {
        "success": True,
        "toc": toc,
        "formatted_toc": format_toc(toc)
    }


def main():
    # 读取输入
    if len(sys.argv) > 1:
        # 命令行模式
        file_path = sys.argv[1]
        toc = extract_toc(file_path)
        print(format_toc(toc))
    else:
        # OpenClaw Skill 模式
        input_data = json.load(sys.stdin)
        command = input_data.get("command")
        args = input_data.get("args", {})
        
        if command == "extract":
            result = handle_extract(args)
        else:
            result = {
                "success": False,
                "error": f"未知命令: {command}"
            }
            
        # 输出结果
        json.dump(result, sys.stdout, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

