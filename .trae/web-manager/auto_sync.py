#!/usr/bin/env python3
"""
智能模板同步工具
自动识别原始文件与模板文件的diff，并应用通用变更到模板
"""

import sys
import os
import re
import difflib
from pathlib import Path
from typing import Tuple, List, Optional

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent


class TemplateSync:
    """模板同步处理器"""
    
    def __init__(self, original_path: Path, template_path: Path):
        self.original_path = original_path
        self.template_path = template_path
        self.original_content = original_path.read_text()
        self.template_content = template_path.read_text()
    
    def extract_generic_content(self, content: str, file_type: str) -> str:
        """
        从原始内容中提取通用化内容
        
        Args:
            content: 原始文件内容
            file_type: 文件类型 ('project_rules', 'memory', 'agents')
        
        Returns:
            通用化后的内容
        """
        result = content
        
        if file_type == 'project_rules':
            # CS-Notes 特定的项目目标
            result = re.sub(
                r'\* 本仓库的整体目标是：笔记知识管理、创作写作、todos 管理、新项目建设、效率工具建设，多功能一体化。',
                '* 请在此定义你的项目目标',
                result
            )
            
            # CS-Notes 特定的目录约定
            result = re.sub(
                r'\* `Notes/`：可复用、可长期演进的知识笔记（evergreen notes）',
                '* `Notes/`：可复用知识笔记（evergreen notes）',
                result
            )
            result = re.sub(
                r'\* `创作/`：面向输出的写作与实验文档（阶段性、可复盘、可迭代）',
                '* `创作/`：面向输出的写作',
                result
            )
            
            # Git 操作 SOP - CS-Notes 特定部分
            result = re.sub(
                r'\* \*\*公用 Skill 能力\*\*：使用 `Notes/snippets/todo-push.sh` 和 `Notes/snippets/todo-pull.sh` 作为标准 git 操作流程',
                '* **公用 Skill 能力**：使用项目提供的 git 脚本（如果有）或遵循团队约定',
                result
            )
            result = re.sub(
                r'\* \*\*todo-push\.sh 白名单机制\*\*：仅允许 `Notes/`、`.trae/`、`创作/` 三个文件夹',
                '* **白名单机制**：根据项目需求配置白名单',
                result
            )
            result = re.sub(
                r'\* \*\*todo-push\.sh 黑名单机制\*\*：绝对禁止 `公司项目/` 文件夹',
                '* **黑名单机制**：根据项目需求配置黑名单',
                result
            )
            result = re.sub(
                r'\* \*\*todo-push\.sh 排除模式\*\*：排除 `.trae/logs/`、`\*\.pyc`、`__pycache__/`、`.DS_Store`',
                '* **排除模式**：根据项目需求配置排除模式',
                result
            )
            result = re.sub(
                r'\* \*\*\.gitignore 配置\*\*：确保 `\*\*/公司项目/\*\*` 在 \.gitignore 中（已有配置）',
                '* **.gitignore 配置**：确保敏感目录在 .gitignore 中',
                result
            )
            result = re.sub(
                r'\* \*\*公司项目/ 目录规则\*\*：该目录下的所有内容永远不要 git add 到公开仓库',
                '* **敏感目录规则**：根据项目需求配置敏感目录',
                result
            )
            result = re.sub(
                r'\* 每次进行 git commit 并 push 后，必须在回复中包含对应的 GitHub commit 链接',
                '* 每次进行 git commit 并 push 后，根据需要包含 commit 链接',
                result
            )
            
            # 笔记整理专家角色 - 通用化
            result = re.sub(
                r'\* 你是一位笔记整理专家，且精通Markdown语法、精通计算机、AI、人文等各领域知识',
                '* 请在此处定义你的项目角色和能力',
                result
            )
            
            # P0 固定为笔记整理 - 通用化
            result = re.sub(
                r'\* \*\*P0\*\*：最高优先级 - 阻断性问题、必须立即处理（笔记整理任务固定为P0）',
                '* **P0**：最高优先级 - 阻断性问题、必须立即处理',
                result
            )
            
            # 参考文章列表 - 移除 CS-Notes 特定文章
            start_marker = '### 参考文章'
            end_marker = '### AI协作原则'
            if start_marker in result and end_marker in result:
                before = result.split(start_marker)[0]
                after = result.split(end_marker)[1]
                result = before + '### 参考文章\n\n*请在此处添加项目参考文章*\n\n' + end_marker + after
            
            # 公司项目创作 - 通用化
            result = re.sub(
                r'\*\*重要：公司项目创作必须遵循 `公司项目/01-公司项目创作pipeline\.md` 中定义的流程\*\*：',
                '**重要：项目创作必须遵循项目定义的流程**：',
                result
            )
            result = re.sub(
                r'\*\*当进行公司项目创作时，必须先读取 `公司项目/01-公司项目创作pipeline\.md`，严格按照该 pipeline 执行。\*\*',
                '**当进行项目创作时，必须先读取项目创作流程文档，严格按照该 pipeline 执行。**',
                result
            )
            
            # 快捷指令 - 移除 CS-Notes 特定路径
            result = re.sub(
                r'1\. \*\*运行迁移脚本\*\*：执行 `\.trae/web-manager/migrate\.sh`',
                '1. **运行迁移脚本**：执行项目迁移脚本',
                result
            )
            result = re.sub(
                r'- 详细工作流请查看 `\.trae/web-manager/WORKFLOW\.md`',
                '- 详细工作流请查看项目工作流文档',
                result
            )
            
            # 新项目建设 - 通用化
            result = re.sub(
                r'\* 新项目建设：\n  - 默认在仓库根目录创建与项目同名目录（或 `Projects/` 下新建），并在 `\.trae/documents/PROJECT_CONTEXT\.md` 中登记入口与目标',
                '* 新项目建设：\n  - 默认在仓库根目录创建与项目同名目录，并在项目文档中登记入口与目标',
                result
            )
        
        elif file_type == 'memory' or file_type == 'agents':
            # MEMORY.md 和 AGENTS.md 是 CS-Notes 特定的，不做通用化处理
            # 直接保持原样，因为这两个文件本身就是高度定制化的
            result = self.template_content
            return result
        
        return result
    
    def sync(self, file_type: str) -> Tuple[bool, str]:
        """
        同步原始文件到模板
        
        Args:
            file_type: 文件类型 ('project_rules', 'memory', 'agents')
        
        Returns:
            (是否有变更, 变更摘要)
        """
        # 从原始文件提取通用化内容
        new_template_content = self.extract_generic_content(self.original_content, file_type)
        
        # 比较新旧模板内容
        if new_template_content == self.template_content:
            return False, "模板已是最新"
        
        # 生成 diff
        diff = difflib.unified_diff(
            self.template_content.splitlines(keepends=True),
            new_template_content.splitlines(keepends=True),
            fromfile=str(self.template_path),
            tofile='new_template'
        )
        diff_str = ''.join(diff)
        
        # 写入新模板
        self.template_path.write_text(new_template_content)
        
        return True, diff_str


def get_file_type(original_path: Path) -> str:
    """根据原始文件路径判断文件类型"""
    filename = original_path.name
    if filename == 'project_rules.md':
        return 'project_rules'
    elif filename == 'MEMORY.md':
        return 'memory'
    elif filename == 'AGENTS.md':
        return 'agents'
    else:
        return 'unknown'


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='智能模板同步工具')
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
    
    file_type = get_file_type(original_path)
    print(f"文件类型: {file_type}")
    
    sync = TemplateSync(original_path, template_path)
    
    if args.dry_run:
        # 仅显示变更
        new_content = sync.extract_generic_content(sync.original_content, file_type)
        diff = difflib.unified_diff(
            sync.template_content.splitlines(keepends=True),
            new_content.splitlines(keepends=True),
            fromfile=str(template_path),
            tofile='new_template'
        )
        print(''.join(diff))
    else:
        # 实际同步
        has_change, summary = sync.sync(file_type)
        if has_change:
            print(f"✅ 模板已更新!\n\n变更:\n{summary}")
        else:
            print(f"✅ {summary}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
