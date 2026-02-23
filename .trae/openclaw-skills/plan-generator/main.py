#!/usr/bin/env python3
"""
Plan Generator Skill - 自动生成任务执行计划
增强版：能够基于任务描述生成更详细、更实用的 Plan
"""

import os
import sys
import json
import uuid
import re
from pathlib import Path
from datetime import datetime

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

PLANS_DIR = REPO_ROOT / ".trae/plans"

class PlanGenerator:
    def __init__(self):
        self.repo_root = REPO_ROOT
        self.plans_dir = PLANS_DIR
        self.plans_dir.mkdir(parents=True, exist_ok=True)
    
    def _analyze_task(self, task_description: str) -> dict:
        """
        分析任务描述，提取关键信息
        """
        task_lower = task_description.lower()
        
        # 分析任务类型
        task_types = []
        if any(keyword in task_lower for keyword in ["改进", "优化", "改进", "enhance", "optimize", "improve", "refactor"]):
            task_types.append("优化改进")
        if any(keyword in task_lower for keyword in ["实现", "开发", "写", "implement", "develop", "write", "create", "build", "make", "模拟", "抓取", "scrape", "crawl"]):
            task_types.append("功能实现")
        if any(keyword in task_lower for keyword in ["修复", "调试", "debug", "fix", "repair", "solve", "解决"]):
            task_types.append("问题修复")
        if any(keyword in task_lower for keyword in ["调研", "分析", "研究", "research", "analyze", "investigate", "explore", "探索"]):
            task_types.append("调研分析")
        if any(keyword in task_lower for keyword in ["整理", "整合", "组织", "organize", "整理", "integrate", "整理"]):
            task_types.append("整理整合")
        if any(keyword in task_lower for keyword in ["测试", "验证", "test", "verify", "validate"]):
            task_types.append("测试验证")
        
        if not task_types:
            task_types.append("通用任务")
        
        # 分析涉及的领域/组件
        domains = []
        if any(keyword in task_lower for keyword in ["plan", "plan generator", "计划"]):
            domains.append("Plan Generator")
        if any(keyword in task_lower for keyword in ["web", "前端", "frontend", "ui"]):
            domains.append("Web 前端")
        if any(keyword in task_lower for keyword in ["后端", "backend", "server", "api"]):
            domains.append("后端服务")
        if any(keyword in task_lower for keyword in ["笔记", "笔记库", "note", "notes"]):
            domains.append("笔记系统")
        if any(keyword in task_lower for keyword in ["todo", "任务", "task"]):
            domains.append("任务管理")
        if any(keyword in task_lower for keyword in ["git", "代码", "code"]):
            domains.append("代码管理")
        
        # 分析任务复杂度
        complexity = "中等"
        if any(keyword in task_lower for keyword in ["复杂", "复杂", "comprehensive", "完整", "完整"]):
            complexity = "高"
        if any(keyword in task_lower for keyword in ["简单", "简单", "quick", "快速", "小"]):
            complexity = "低"
        
        return {
            "task_types": task_types,
            "domains": domains,
            "complexity": complexity
        }
    
    def _generate_smart_steps(self, task_description: str, task_analysis: dict) -> list:
        """
        基于任务分析生成智能执行步骤
        """
        task_types = task_analysis["task_types"]
        domains = task_analysis["domains"]
        
        steps = []
        
        # 通用起始步骤
        steps.append("1. 分析任务需求，理解目标和范围")
        steps.append("2. 查看相关文件和代码，了解现有实现")
        
        # 根据任务类型添加特定步骤
        if "优化改进" in task_types:
            steps.append("3. 分析现有实现的问题和瓶颈")
            steps.append("4. 设计优化方案")
            steps.append("5. 实现优化")
            steps.append("6. 测试验证优化效果")
        
        if "功能实现" in task_types:
            steps.append("3. 设计功能架构和实现方案")
            steps.append("4. 编写代码实现功能")
            steps.append("5. 测试功能是否正常工作")
        
        if "问题修复" in task_types:
            steps.append("3. 复现问题，分析根本原因")
            steps.append("4. 设计修复方案")
            steps.append("5. 实现修复")
            steps.append("6. 验证问题是否解决")
        
        if "调研分析" in task_types:
            steps.append("3. 收集相关信息和资料")
            steps.append("4. 分析和整理信息")
            steps.append("5. 输出调研报告或总结")
        
        if "整理整合" in task_types:
            steps.append("3. 整理现有内容和结构")
            steps.append("4. 设计新的组织结构")
            steps.append("5. 执行整理和整合")
            steps.append("6. 验证整理结果")
        
        # 通用收尾步骤
        steps.append("7. 验证整体效果，确保任务目标达成")
        steps.append("8. 如有需要，更新相关文档")
        
        return steps
    
    def _generate_smart_acceptance_criteria(self, task_description: str, task_analysis: dict) -> list:
        """
        基于任务分析生成智能验收标准
        """
        criteria = []
        
        # 通用验收标准
        criteria.append("任务目标达成")
        
        # 根据任务类型添加特定验收标准
        if "优化改进" in task_analysis["task_types"]:
            criteria.append("性能或体验有明显提升")
            criteria.append("没有引入新的问题")
        
        if "功能实现" in task_analysis["task_types"]:
            criteria.append("功能可以正常使用")
            criteria.append("代码质量符合规范")
        
        if "问题修复" in task_analysis["task_types"]:
            criteria.append("问题已彻底解决")
            criteria.append("可以正常复现验证修复效果")
        
        if "调研分析" in task_analysis["task_types"]:
            criteria.append("调研结果完整准确")
            criteria.append("调研报告清晰易读")
        
        if "整理整合" in task_analysis["task_types"]:
            criteria.append("内容组织合理")
            criteria.append("没有信息丢失")
        
        # 通用技术验收标准
        criteria.append("相关代码/文档已更新")
        
        return criteria
    
    def _generate_smart_risks(self, task_description: str, task_analysis: dict) -> list:
        """
        基于任务分析生成智能风险列表
        """
        risks = []
        
        # 通用风险
        risks.append({
            "risk": "AI 生成错误",
            "mitigation": "人工审核"
        })
        
        # 根据任务类型添加特定风险
        if "优化改进" in task_analysis["task_types"]:
            risks.append({
                "risk": "优化可能引入新的 bug",
                "mitigation": "充分测试，回滚方案"
            })
        
        if "功能实现" in task_analysis["task_types"]:
            risks.append({
                "risk": "功能设计与需求不符",
                "mitigation": "提前确认需求，迭代改进"
            })
        
        if "问题修复" in task_analysis["task_types"]:
            risks.append({
                "risk": "修复不彻底，问题复发",
                "mitigation": "深入分析根本原因，多场景测试"
            })
        
        if task_analysis["complexity"] == "高":
            risks.append({
                "risk": "任务复杂度高，预计时间不足",
                "mitigation": "分阶段执行，及时调整计划"
            })
        
        return risks
    
    def _estimate_time(self, task_analysis: dict) -> str:
        """
        基于任务复杂度估算时间
        """
        complexity = task_analysis["complexity"]
        if complexity == "高":
            return "2-4小时"
        elif complexity == "低":
            return "15-30分钟"
        else:
            return "30-60分钟"
    
    def generate_plan(self, task_description: str, priority: str = "medium") -> dict:
        """
        生成智能 Plan
        """
        plan_id = f"plan-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:8]}"
        
        # 分析任务
        task_analysis = self._analyze_task(task_description)
        
        # 生成智能内容
        steps = self._generate_smart_steps(task_description, task_analysis)
        acceptance_criteria = self._generate_smart_acceptance_criteria(task_description, task_analysis)
        risks = self._generate_smart_risks(task_description, task_analysis)
        time_estimate = self._estimate_time(task_analysis)
        
        # 生成标签
        tags = []
        tags.extend(task_analysis["task_types"])
        tags.extend(task_analysis["domains"])
        tags.append(f"复杂度:{task_analysis['complexity']}")
        
        plan = {
            "id": plan_id,
            "title": task_description[:60],
            "goal": task_description,
            "priority": priority,
            "status": "pending",
            "task_analysis": task_analysis,
            "assumptions": [
                "文件系统可访问",
                "AI 模型可用",
                "Git 仓库状态正常"
            ],
            "changes": {
                "modify": [],
                "add": [],
                "delete": []
            },
            "acceptance_criteria": acceptance_criteria,
            "risks": risks,
            "steps": steps,
            "time_estimate": time_estimate,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "tags": tags
        }
        return plan
    
    def write_plan_to_file(self, plan: dict, todo_id: str = None) -> Path:
        """将 Plan 写入单独的文件"""
        slugified_title = re.sub(r'[^\w\-]+', '-', plan['title'].lower()).strip('-')
        filename = f"{datetime.now().strftime('%Y-%m-%d')}-{slugified_title}-{plan['id'].split('-')[-1]}.md"
        file_path = self.plans_dir / filename
        
        plan_markdown = self._format_plan_as_markdown(plan, todo_id)
        
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(plan_markdown)
        
        return file_path
    
    def _format_plan_as_markdown(self, plan: dict, todo_id: str = None) -> str:
        """将 Plan 格式化为 Markdown（带 YAML frontmatter）"""
        lines = []
        
        # YAML frontmatter
        lines.append("---")
        lines.append(f"id: {plan['id']}")
        if todo_id:
            lines.append(f"todo_id: {todo_id}")
        lines.append(f'title: "{plan["title"]}"')
        lines.append(f"priority: {plan['priority']}")
        lines.append(f"status: {plan['status']}")
        lines.append(f"created_at: {plan['created_at']}")
        lines.append(f"updated_at: {plan['updated_at']}")
        lines.append(f"tags: {plan['tags']}")
        lines.append("---")
        lines.append("")
        
        # Plan 内容 - 按照设计方案优化
        lines.append("## 📋 背景与动机")
        lines.append(f"- 任务目标: {plan['goal']}")
        lines.append("")
        
        # 任务分析
        if "task_analysis" in plan:
            lines.append("## 💡 核心洞察")
            lines.append(f"- 任务类型: {', '.join(plan['task_analysis']['task_types'])}")
            if plan['task_analysis']['domains']:
                lines.append(f"- 涉及领域: {', '.join(plan['task_analysis']['domains'])}")
            lines.append(f"- 复杂度: {plan['task_analysis']['complexity']}")
            lines.append("")
        
        lines.append("## 🎯 任务拆解")
        for i, step in enumerate(plan['steps'], 1):
            clean_step = re.sub(r'^\d+\.\s*', '', step)
            lines.append(f"- 步骤 {i}: {clean_step}")
        lines.append("")
        
        lines.append("## 🔨 具体执行步骤")
        for i, step in enumerate(plan['steps'], 1):
            clean_step = re.sub(r'^\d+\.\s*', '', step)
            lines.append(f"### 步骤 {i}")
            lines.append(f"- 做什么: {clean_step}")
            lines.append(f"- 怎么做: 按照步骤要求执行")
            lines.append(f"- 产出: 任务阶段性成果")
            lines.append(f"- 验收标准: 步骤完成")
            lines.append("")
        
        lines.append("## ✅ 验收标准")
        for criteria in plan['acceptance_criteria']:
            lines.append(f"- {criteria}")
        lines.append("")
        
        lines.append("## ⚠️ 风险与挑战")
        for risk in plan['risks']:
            lines.append(f"- {risk['risk']}，应对: {risk['mitigation']}")
        lines.append("")
        
        lines.append("## 📚 相关资源")
        lines.append(f"- 文件: 待补充")
        lines.append(f"- 链接: 待补充")
        lines.append(f"- 笔记: 待补充")
        lines.append("")
        
        lines.append("## ⏱️ 时间估算")
        lines.append(f"- 预计: {plan['time_estimate']}")
        lines.append("")
        lines.append("---")
        lines.append("")
        lines.append("## 📝 执行记录")
        
        return "\n".join(lines)

def handle_generate():
    """处理生成 Plan 命令"""
    if len(sys.argv) < 3:
        print("Usage: python main.py generate <task_description> [priority] [todo_id]")
        return json.dumps({"success": False, "error": "Missing task description"})
    
    task_description = sys.argv[2]
    priority = sys.argv[3] if len(sys.argv) > 3 else "medium"
    todo_id = sys.argv[4] if len(sys.argv) > 4 else None
    
    generator = PlanGenerator()
    plan = generator.generate_plan(task_description, priority)
    file_path = generator.write_plan_to_file(plan, todo_id)
    
    return json.dumps({
        "success": True,
        "plan": plan,
        "file_path": str(file_path),
        "message": f"Plan 已生成并写入 {file_path}"
    })

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "generate":
        print(handle_generate())
