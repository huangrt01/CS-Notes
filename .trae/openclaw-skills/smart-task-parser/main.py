#!/usr/bin/env python3
"""
智能任务解析 Skill - 基于 LLM 智能解析口述式任务，写入 todos.json
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime


class SmartTaskParser:
    """智能任务解析器"""
    
    def __init__(self, config: dict):
        """初始化"""
        self.config = config
        self.todos_json_path = Path(config.get("todos_json_path", ""))
        self.workspace = Path(config.get("workspace", ""))
    
    def parse_task(self, task_text: str) -> dict:
        """
        解析任务文本
        
        基于 LLM 智能解析口述式任务
        """
        # 基于 LLM 的智能解析逻辑
        # 这里是一个示例，实际使用时可以调用 OpenClaw 的 LLM 能力
        
        # 简单的智能解析（示例）
        # 实际使用时应该调用 LLM 进行智能解析
        
        # 分析任务文本，提取关键信息
        priority = "medium"
        if "高优先级" in task_text or "紧急" in task_text or "重要" in task_text:
            priority = "high"
        elif "低优先级" in task_text or "不急" in task_text:
            priority = "low"
        
        assignee = "ai"
        if "你来做" in task_text or "你帮我" in task_text:
            assignee = "ai"
        elif "我来做" in task_text or "我自己" in task_text:
            assignee = "user"
        
        feedback_required = False
        if "需要确认" in task_text or "等我确认" in task_text:
            feedback_required = True
        
        # 提取任务标题（移除一些修饰词）
        title = task_text
        title = title.replace("高优先级：", "").replace("高优先级:", "")
        title = title.replace("低优先级：", "").replace("低优先级:", "")
        title = title.replace("紧急：", "").replace("紧急:", "")
        title = title.strip()
        
        task = {
            "id": f"todo-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            "title": title[:200] if len(title) > 200 else title,
            "status": "pending",
            "priority": priority,
            "assignee": assignee,
            "feedback_required": feedback_required,
            "created_at": datetime.now().isoformat(),
            "definition_of_done": [
                "完成任务",
                "验证结果"
            ],
            "progress": "⏸️ 待执行",
            "original_text": task_text
        }
        
        return task
    
    def write_to_todos_json(self, task: dict) -> bool:
        """写入 todos.json"""
        try:
            if not self.todos_json_path.exists():
                print(f"Error: todos.json not found at {self.todos_json_path}")
                return False
            
            # 读取现有的 todos.json
            with open(self.todos_json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 添加新任务
            if "todos" not in data:
                data["todos"] = []
            
            data["todos"].append(task)
            data["updated_at"] = datetime.now().isoformat()
            
            # 写回文件
            with open(self.todos_json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            print(f"✅ 任务已写入 todos.json: {task['id']}")
            return True
            
        except Exception as e:
            print(f"Error writing to todos.json: {e}")
            return False
    
    def run(self, task_text: str) -> dict:
        """运行解析流程"""
        print("=" * 60)
        print("🤖 智能任务解析")
        print("=" * 60)
        print()
        print(f"输入任务: {task_text}")
        print()
        
        # 解析任务
        print("🔍 解析任务...")
        task = self.parse_task(task_text)
        print(f"✅ 任务解析完成: {task['title']}")
        print()
        
        # 写入 todos.json
        print("📝 写入 todos.json...")
        success = self.write_to_todos_json(task)
        print()
        
        if success:
            print("=" * 60)
            print("✅ 任务解析成功！")
            print("=" * 60)
        else:
            print("=" * 60)
            print("❌ 任务解析失败！")
            print("=" * 60)
        
        return task


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="智能任务解析 Skill")
    parser.add_argument("task_text", help="任务文本")
    parser.add_argument("--config", help="配置文件路径")
    args = parser.parse_args()
    
    # 加载配置
    config = {}
    if args.config:
        config_path = Path(args.config)
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
    
    # 使用默认配置
    if not config:
        config = {
            "todos_json_path": "/root/.openclaw/workspace/CS-Notes/.trae/todos/todos.json",
            "workspace": "/root/.openclaw/workspace/CS-Notes"
        }
    
    # 创建解析器并运行
    parser_instance = SmartTaskParser(config)
    parser_instance.run(args.task_text)


if __name__ == "__main__":
    main()
