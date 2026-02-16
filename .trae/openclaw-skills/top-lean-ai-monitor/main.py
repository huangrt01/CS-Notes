#!/usr/bin/env python3
"""
Top Lean AI 榜单监控 Skill - OpenClaw 集成
支持：
1. 手动命令调用：check、status
2. 定时任务：每日自动检查
3. 飞书通知：利用 OpenClaw 原生能力
"""

import sys
import json
from pathlib import Path

# 添加主脚本路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from top_lean_ai_monitor import TopLeanAIMonitor


def handle_check():
    """
    手动命令：检查榜单更新
    用法：top-lean-ai-monitor check
    """
    monitor = TopLeanAIMonitor()
    result = monitor.check_updates()
    
    if result["success"] and result["new_companies"]:
        # 有新公司，构建消息
        message = build_notification_message(result, monitor)
        
        # 使用 OpenClaw 原生能力发送飞书通知
        # OpenClaw 会识别这个标记并发送通知
        print(f"[OPENCLAW_MESSAGE_SEND]{message}")
        
        return {
            "status": "success",
            "new_companies_count": len(result["new_companies"]),
            "message": f"发现 {len(result['new_companies'])} 家新公司！已发送飞书通知"
        }
    elif result["success"]:
        return {
            "status": "success",
            "new_companies_count": 0,
            "message": "没有新公司上榜",
            "total_companies": result["total_companies"]
        }
    else:
        return {
            "status": "error",
            "error": result.get("error", "Unknown error")
        }


def handle_status():
    """
    手动命令：查看监控状态
    用法：top-lean-ai-monitor status
    """
    monitor = TopLeanAIMonitor()
    status = monitor.get_status()
    
    # 构建可读的状态消息
    message = "📊 Top Lean AI 榜单监控状态\n\n"
    message += f"🕐 上次检查: {status['last_check'] or '从未检查'}\n"
    message += f"🏢 已知公司总数: {status['known_companies_count']}\n"
    message += f"🆕 历史新公司数: {status['new_companies_count']}\n"
    message += f"🔍 检查次数: {status['check_count']}\n"
    
    if status["recent_new_companies"]:
        message += "\n🆕 最近发现的新公司:\n"
        for company in status["recent_new_companies"][-5:]:
            message += f"  - {company['name']} ({company['discovered_at'][:10]})\n"
    
    print(f"[OPENCLAW_MESSAGE_SEND]{message}")
    
    return status


def handle_daily_check():
    """
    定时任务：每日检查
    由 OpenClaw 的 cron 调度器触发
    """
    print("🔔 执行每日 Top Lean AI 榜单检查...")
    
    monitor = TopLeanAIMonitor()
    result = monitor.check_updates()
    
    if result["success"] and result["new_companies"]:
        message = build_notification_message(result, monitor)
        print(f"[OPENCLAW_MESSAGE_SEND]{message}")
        
        return {
            "status": "success",
            "action": "daily_check",
            "new_companies_count": len(result["new_companies"]),
            "notified": True
        }
    elif result["success"]:
        return {
            "status": "success",
            "action": "daily_check",
            "new_companies_count": 0,
            "total_companies": result["total_companies"],
            "notified": False
        }
    else:
        return {
            "status": "error",
            "action": "daily_check",
            "error": result.get("error", "Unknown error")
        }


def build_notification_message(result, monitor):
    """构建飞书通知消息"""
    message = "🔔 Top Lean AI 榜单更新!\n\n"
    message += f"榜单链接: https://leanaileaderboard.com/\n\n"
    message += f"发现 {len(result['new_companies'])} 家新公司:\n\n"
    
    for company in result["new_companies"]:
        message += monitor.format_company_message(company)
        message += "\n"
    
    return message


def main():
    """主入口"""
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python main.py check    # 检查更新")
        print("  python main.py status   # 查看状态")
        print("  python main.py daily    # 每日检查（定时任务用）")
        return
    
    command = sys.argv[1]
    
    if command == "check":
        result = handle_check()
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif command == "status":
        result = handle_status()
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif command == "daily":
        result = handle_daily_check()
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"Unknown command: {command}")


if __name__ == "__main__":
    main()
