#!/usr/bin/env python3
"""
Todo 管理工具 - 帮助从 Inbox 搬运任务到 Pending
支持多种格式：极简版、带链接版、结构化版
"""

import re
from pathlib import Path

# 文件路径
REPO_ROOT = Path(__file__).parent.parent.parent
INBOX_PATH = REPO_ROOT / ".trae/documents/INBOX.md"
TODOS_PATH = REPO_ROOT / ".trae/documents/todos管理系统.md"


def extract_urls(text: str):
    """从文本中提取 URL"""
    url_pattern = re.compile(r'https?://[^\s]+')
    return url_pattern.findall(text)


def parse_inbox_tasks(content: str):
    """解析 INBOX.md 中的任务条目，支持多种格式"""
    # 先移除代码块
    content_no_code = re.sub(r'```[\s\S]*?```', '', content, flags=re.MULTILINE)
    
    tasks = []
    
    # 格式 3: 结构化版
    structured_pattern = re.compile(
        r'- 内容：(.*?)\n\s*优先级：(.*?)\n\s*关联文件：(.*?)\n\s*参考链接：(.*?)\n\s*截止时间：(.*?)(?=\n- 内容：|\Z)',
        re.DOTALL
    )
    for match in structured_pattern.findall(content_no_code):
        content_text, priority, links, ref_links, due = match
        if content_text.strip() and content_text.strip() != "[任务描述]":
            tasks.append({
                "content": content_text.strip(),
                "priority": priority.strip(),
                "links": links.strip(),
                "ref_links": ref_links.strip(),
                "due": due.strip(),
                "format": "structured"
            })
    
    # 格式 1 & 2: 极简版和带链接版
    # 匹配以 "- " 开头的行，但排除已匹配的结构化任务
    lines = content_no_code.split('\n')
    in_structured = False
    for line in lines:
        line = line.strip()
        
        # 检测是否进入结构化任务
        if line.startswith('- 内容：'):
            in_structured = True
            continue
        if in_structured:
            if not line or line.startswith('##'):
                in_structured = False
            continue
        
        # 匹配极简版任务
        if line.startswith('- ') and len(line) > 2:
            task_content = line[2:].strip()
            if task_content and not task_content.startswith('['):
                urls = extract_urls(task_content)
                tasks.append({
                    "content": task_content,
                    "priority": "",
                    "links": "",
                    "ref_links": ", ".join(urls) if urls else "",
                    "due": "",
                    "format": "simple"
                })
    
    return tasks


def format_task_for_pending(task: dict):
    """将任务格式化为 Pending 列表条目"""
    lines = [f"- [ ] {task['content']}"]
    if task['priority']:
        lines.append(f"  - Priority：{task['priority']}")
    if task['links']:
        lines.append(f"  - Links：{task['links']}")
    if task['ref_links']:
        lines.append(f"  - Reference：{task['ref_links']}")
    if task['due']:
        lines.append(f"  - Due：{task['due']}")
    return "\n".join(lines)


def main():
    print("=" * 60)
    print("Todo 管理工具")
    print("=" * 60)
    
    # 读取 INBOX.md
    if not INBOX_PATH.exists():
        print(f"错误：找不到文件 {INBOX_PATH}")
        return
    
    with open(INBOX_PATH, "r", encoding="utf-8") as f:
        inbox_content = f.read()
    
    # 解析任务
    tasks = parse_inbox_tasks(inbox_content)
    
    if not tasks:
        print("Inbox 中没有找到有效的任务条目")
        print("\n提示：你可以在 Inbox.md 中添加一行简单的任务，例如：")
        print('  - 把这篇文章整理到笔记')
        print('  - https://example.com/article - 很有价值的文章')
        return
    
    print(f"\n找到 {len(tasks)} 个任务：\n")
    for i, task in enumerate(tasks, 1):
        format_label = "（极简）" if task.get('format') == 'simple' else "（结构化）"
        print(f"  {i}. {task['content']} {format_label}")
    
    print("\n" + "=" * 60)
    print("提示：请手动将这些任务移动到 todos管理系统.md 的 Pending 部分")
    print("      移动后可以清空 Inbox 中的这些条目")
    print("=" * 60)
    
    print("\n以下是格式化好的任务条目，可以直接复制到 Pending：\n")
    for task in tasks:
        print(format_task_for_pending(task))
        print()


if __name__ == "__main__":
    main()

