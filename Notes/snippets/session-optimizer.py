#!/usr/bin/env python3
"""
OpenClaw Session 优化器
基于 OpenClaw 现有能力，不侵入内部代码
自动监控 session 状态，在需要时提醒用户切换 session

使用方法：
1. 每次对话前运行：python3 session-optimizer.py check
2. 如果看到警告，在 OpenClaw TUI 中使用 `/reset` 命令
"""

import os
import json
from datetime import datetime
from pathlib import Path

class SessionOptimizer:
    def __init__(self, workspace_path=None):
        self.workspace_path = Path(workspace_path) if workspace_path else Path.cwd()
        self.state_file = self.workspace_path / ".openclaw-session-optimizer.json"
        self.state = self.load_state()
    
    def load_state(self):
        """加载状态"""
        if self.state_file.exists():
            with open(self.state_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {
            "session_start_time": datetime.now().isoformat(),
            "message_count": 0,
            "warnings_given": [],
            "last_reset": None,
            "history": []
        }
    
    def save_state(self):
        """保存状态"""
        with open(self.state_file, 'w', encoding='utf-8') as f:
            json.dump(self.state, f, ensure_ascii=False, indent=2)
    
    def check_session(self):
        """检查 session 状态，返回是否需要切换"""
        session_age = (datetime.now() - datetime.fromisoformat(self.state["session_start_time"])).total_seconds()
        message_count = self.state["message_count"]
        
        print("=" * 60)
        print("🔍 OpenClaw Session 检查")
        print("=" * 60)
        print()
        print(f"🕐 Session 开始时间: {self.state['session_start_time']}")
        print(f"⏱️  Session 已运行: {session_age/3600:.1f} 小时")
        print(f"💬 消息数量: {message_count}")
        print()
        
        need_reset = False
        warnings = []
        
        # 检查 1: 消息数量
        if message_count >= 50:
            warnings.append(f"⚠️ 消息数量已达 {message_count} 条，强烈建议切换 session！")
            need_reset = True
        elif message_count >= 30:
            warnings.append(f"📊 消息数量已达 {message_count} 条，请注意")
        
        # 检查 2: 时间
        if session_age >= 24 * 3600:  # 24 小时
            warnings.append(f"⚠️ Session 已运行超过 24 小时，建议切换！")
            need_reset = True
        elif session_age >= 12 * 3600:  # 12 小时
            warnings.append(f"📊 Session 已运行 {session_age/3600:.1f} 小时")
        
        if warnings:
            print("⚠️ 警告:")
            for warning in warnings:
                print(f"  {warning}")
            print()
        
        if need_reset:
            print("🚨 建议立即切换 session！")
            print()
            print("操作步骤:")
            print("1. 在 OpenClaw TUI 中输入 `/reset`")
            print("2. 确认 session 已重置")
            print("3. 继续你的对话")
            print()
        else:
            print("✅ Session 状态良好，继续使用！")
            print()
        
        print("=" * 60)
        
        # 记录警告
        if warnings:
            self.state["warnings_given"].extend(warnings)
            self.save_state()
        
        return need_reset
    
    def log_message(self):
        """记录一条消息"""
        self.state["message_count"] += 1
        self.save_state()
        
        # 检查是否需要警告
        return self.check_session()
    
    def reset_session(self):
        """重置 session（记录状态，实际切换在 OpenClaw TUI 中执行）"""
        print()
        print("🔄 准备重置 session...")
        print()
        print("⚠️ 重要提示：")
        print("   这个脚本只是记录状态，实际的 session 切换")
        print("   需要在 OpenClaw TUI 中执行 `/reset` 命令")
        print()
        
        # 记录历史
        if self.state["message_count"] > 0:
            self.state["history"].append({
                "start_time": self.state["session_start_time"],
                "end_time": datetime.now().isoformat(),
                "message_count": self.state["message_count"],
                "warnings_given": self.state["warnings_given"]
            })
        
        # 重置状态
        self.state["session_start_time"] = datetime.now().isoformat()
        self.state["message_count"] = 0
        self.state["warnings_given"] = []
        self.state["last_reset"] = datetime.now().isoformat()
        self.save_state()
        
        print("✅ 状态已记录！")
        print()
        print("现在请在 OpenClaw TUI 中执行：")
        print("  /reset")
        print()
    
    def print_history(self):
        """打印历史记录"""
        print("=" * 60)
        print("📊 Session 历史记录")
        print("=" * 60)
        print()
        
        if not self.state["history"]:
            print("暂无历史记录")
            print()
            return
        
        for i, session in enumerate(reversed(self.state["history"][-5:])):  # 只显示最近 5 个
            print(f"Session {len(self.state['history']) - i}:")
            print(f"  开始: {session['start_time']}")
            print(f"  结束: {session['end_time']}")
            print(f"  消息数: {session['message_count']}")
            print(f"  警告数: {len(session['warnings_given'])}")
            print()
        
        print("=" * 60)


def main():
    """主函数"""
    import sys
    
    optimizer = SessionOptimizer()
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "check":
            optimizer.check_session()
        elif command == "log":
            optimizer.log_message()
        elif command == "reset":
            optimizer.reset_session()
        elif command == "history":
            optimizer.print_history()
        else:
            print(f"未知命令: {command}")
            print("使用:")
            print("  python session-optimizer.py check     # 检查 session 状态")
            print("  python session-optimizer.py log       # 记录一条消息并检查")
            print("  python session-optimizer.py reset     # 准备重置 session")
            print("  python session-optimizer.py history   # 查看历史记录")
    else:
        optimizer.check_session()


if __name__ == "__main__":
    main()
