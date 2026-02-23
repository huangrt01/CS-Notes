#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能模板同步工具
只同步真正通用的变更，不同步项目特定内容
"""

import sys
import re
import difflib
from pathlib import Path
from typing import Tuple


class SmartTemplateSync:
    """智能模板同步处理器"""
    
    def __init__(self, original_path: Path, template_path: Path):
        self.original_path = original_path
        self.template_path = template_path
        self.original_content = original_path.read_text()
        self.template_content = template_path.read_text()
    
    def extract_generic_updates(self) -> str:
        """
        智能提取通用更新，只保留真正通用的内容
        
        Returns:
            新的模板内容
        """
        result = self.template_content
        
        # === Todo 驱动工作流 - 我们刚才修复的真正通用的变更 ===
        # 1. 更新 priority-task-reader 调用方式（添加只返回 pending 的说明）
        result = re.sub(
            r'\* 当我说"根据 todos 执行 / 执行 todos / 清空待办"、"纵观本仓库全局,推进todos"时:\n  - 先读取 `.trae/todos/todos.json`,确定当前 Pending/进行中任务\n  - 使用 priority-task-reader skill\(`\.trae/openclaw-skills/priority-task-reader/main\.py --next`\)找到下一个应该执行的任务\n  - 选择最重要且可独立完成的一项开始执行,完成后立刻回写进度与产物链接,再继续下一项\n  - 涉及复杂改动时,先给出 Plan\(目标、假设、改动点、验收标准、风险\),Plan 明确后再动手\n  - 执行中产生的\*\*核心经验教训必须精炼沉淀到本文件（project_rules\.md）\*\*中（确保 Trae 能读取到）\n  - 详细的经验教训记录可以同时沉淀到 `\.trae/documents/PROGRESS\.md` 作为补充\n\* \*\*【强制】开始推进 todo 前必须先标记为 in-progress\*\*:避免多 session 同时执行同一个 task 造成冲突\n  - \*\*核心原则\*\*:当决定开始推进某个 todo 时,第一时间先把它从 pending 改为 in-progress\n  - \*\*为什么重要\*\*:如果有多个 session 在同时运行,不先标记的话可能导致两个 session 同时执行同一个任务\n  - \*\*标准流程\*\*:\n    1\. 使用 priority-task-reader 找到下一个应该执行的任务\n    2\. 立即编辑 todos\.json,把该任务的 status 从 pending 改为 in-progress\n    3\. 添加 started_at 时间戳\n    4\. 然后才开始实际执行任务\n  - \*\*新建 session 规则\*\*:新建的 session 不要执行 status 为 in-progress 的任务,只执行 pending 状态的任务',
            r'''* 当我说"根据 todos 执行 / 执行 todos / 清空待办"、"纵观本仓库全局，推进todos"时：
  - 先读取 `.trae/todos/todos.json`
  - 使用 priority-task-reader skill（`.trae/openclaw-skills/priority-task-reader/main.py --next`）找到下一个应该执行的任务（**只返回 pending 状态的任务**）
  - **【强制】立即编辑 todos.json，把该任务的 status 从 pending 改为 in-progress**，添加 started_at 时间戳
  - 然后才开始实际执行任务
  - 完成后立刻回写进度与产物链接，再继续下一项
  - 涉及复杂改动时，先给出 Plan（目标、假设、改动点、验收标准、风险），Plan 明确后再动手
  - 执行中产生的**核心经验教训必须精炼沉淀到本文件（project_rules.md）**中（确保 Trae 能读取到）
  - 详细的经验教训记录可以同时沉淀到 `.trae/documents/PROGRESS.md` 作为补充
* **为什么要先标记为 in-progress**：避免多 session 同时执行同一个 task 造成冲突
  - priority-task-reader 保证只返回 pending 状态的任务
  - 拿到任务后第一时间标记为 in-progress
  - 新建 session 不会再次选择这个任务''',
            result
        )
        
        # === 统一一些符号（全角 -> 半角），这是通用的 ===
        result = re.sub(r':（', r': (', result)  # 冒号后的全角括号
        result = re.sub(r'）\.', r').', result)    # 全角右括号后点
        result = re.sub(r'）,', r'),', result)    # 全角右括号后逗号
        result = re.sub(r'）\*', r')*', result)    # 全角右括号后星号
        result = re.sub(r'）\*', r')*', result)
        
        return result
    
    def sync(self) -> Tuple[bool, str]:
        """
        执行同步
        
        Returns:
            (是否有变更, 变更摘要)
        """
        new_template_content = self.extract_generic_updates()
        
        if new_template_content == self.template_content:
            return False, "模板已是最新"
        
        diff = difflib.unified_diff(
            self.template_content.splitlines(keepends=True),
            new_template_content.splitlines(keepends=True),
            fromfile=str(self.template_path),
            tofile='new_template'
        )
        diff_str = ''.join(diff)
        
        self.template_path.write_text(new_template_content)
        
        return True, diff_str


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='智能模板同步工具 - 只同步通用变更')
    parser.add_argument('original', help='原始文件路径')
    parser.add_argument('template', help='模板文件路径')
    parser.add_argument('--dry-run', action='store_true', help='仅显示变更，不实际修改')
    
    args = parser.parse_args()
    
    original_path = Path(args.original)
    template_path = Path(args.template)
    
    if not original_path.exists():
        print(f"错误: 原始文件不存在: {original_path}")
        return 1
    
    if not template_path.exists():
        print(f"错误: 模板文件不存在: {template_path}")
        return 1
    
    filename = original_path.name
    print(f"文件: {filename}")
    
    if filename != 'project_rules.md':
        print(f"  ✅ {filename} 不需要同步，保持原样")
        return 0
    
    sync = SmartTemplateSync(original_path, template_path)
    
    if args.dry_run:
        new_content = sync.extract_generic_updates()
        diff = difflib.unified_diff(
            sync.template_content.splitlines(keepends=True),
            new_content.splitlines(keepends=True),
            fromfile=str(template_path),
            tofile='new_template'
        )
        print(''.join(diff))
    else:
        has_change, summary = sync.sync()
        if has_change:
            print(f"✅ 模板已更新!\n\n变更:\n{summary}")
        else:
            print(f"✅ {summary}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
