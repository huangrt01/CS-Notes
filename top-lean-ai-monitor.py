#!/usr/bin/env python3
"""
Top Lean AI 榜单监控脚本
类似 RSS 订阅方式，每天检查榜单更新，发现新项目时发送飞书通知

数据源：https://leanaileaderboard.com/
创建者：Henry Shi（LinkedIn: https://www.linkedin.com/in/henrythe9th/，X: https://x.com/henrythe9ths/）
资格标准：超过 $5MM ARR、少于 50 名员工、成立不到 5 年
更新频率：每周更新
"""

import os
import json
from datetime import datetime
from pathlib import Path
import time

class TopLeanAIMonitor:
    def __init__(self, workspace_path=None):
        self.workspace_path = Path(workspace_path) if workspace_path else Path.cwd()
        self.state_file = self.workspace_path / ".top-lean-ai-state.json"
        self.leaderboard_url = "https://leanaileaderboard.com/"
        
        # 已知的榜单信息（从笔记中提取）
        self.known_companies = {
            "Perplexity": {"category": "AI Search", "arr": "5000万+", "notes": "AI搜索"},
            "Cursor": {"category": "AI Coding", "arr": "5000万+", "notes": "AI编程"},
            "Runway": {"category": "Content Creation", "arr": "5000万+", "notes": "视频生成"},
            "HeyGen": {"category": "Content Creation", "arr": "5000万+", "notes": "视频生成"},
            "Harvey": {"category": "Legal", "arr": "5000万+", "notes": "法律AI"},
            "Manus": {"category": "General Agent", "arr": "被Meta收购(20亿+)", "notes": "通用Agent，蝴蝶效应"},
            "Genspark": {"category": "AI Search", "arr": "5000万", "notes": "前小度CEO景鲲创立"},
            "OpenArt": {"category": "Content Creation", "arr": "7000万", "notes": "Coco Mao创立，20人团队"},
            "PixVerse": {"category": "Content Creation", "arr": "4000万+", "notes": "视频生成"},
            "Lovart": {"category": "Content Creation", "arr": "3000万+", "notes": "视频生成"}
        }
        
        # 榜单资格标准
        self.qualification_criteria = {
            "min_arr": "5MM ARR (run rate)",
            "max_employees": 50,
            "max_age_years": 5,
            "creator": "Henry Shi",
            "linkedin": "https://www.linkedin.com/in/henrythe9th/",
            "x_twitter": "https://x.com/henrythe9ths/",
            "update_frequency": "weekly",
            "vision": "1-person billion dollar company"
        }
        
        self.state = self.load_state()
    
    def load_state(self):
        """加载状态"""
        if self.state_file.exists():
            with open(self.state_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {
            "last_check": None,
            "known_companies": self.known_companies.copy(),
            "new_companies": [],
            "check_history": [],
            "leaderboard_url": self.leaderboard_url,
            "qualification_criteria": self.qualification_criteria
        }
    
    def save_state(self):
        """保存状态"""
        with open(self.state_file, 'w', encoding='utf-8') as f:
            json.dump(self.state, f, ensure_ascii=False, indent=2)
    
    def search_for_list(self):
        """搜索 Top Lean AI 榜单
        
        数据源：https://leanaileaderboard.com/
        TODO: 需要解析 JavaScript 加载的榜单数据
        当前状态：页面显示 "Loading leaderboard data..."
        """
        print("🔍 正在检查 Top Lean AI 榜单...")
        print(f"📊 榜单 URL: {self.leaderboard_url}")
        print()
        print("📋 榜单资格标准:")
        print(f"   - 超过 {self.qualification_criteria['min_arr']}")
        print(f"   - 少于 {self.qualification_criteria['max_employees']} 名员工")
        print(f"   - 成立不到 {self.qualification_criteria['max_age_years']} 年")
        print()
        print("👤 创建者信息:")
        print(f"   - LinkedIn: {self.qualification_criteria['linkedin']}")
        print(f"   - X (Twitter): {self.qualification_criteria['x_twitter']}")
        print()
        print("⚠️ 注意: 页面显示 'Loading leaderboard data...'")
        print("   需要进一步解析 JavaScript 加载的榜单数据")
        print()
        
        # 返回已知的公司列表作为占位
        return self.known_companies
    
    def check_for_updates(self):
        """检查榜单更新"""
        print("=" * 60)
        print("🔍 Top Lean AI 榜单检查")
        print("=" * 60)
        print()
        
        current_time = datetime.now().isoformat()
        print(f"🕐 检查时间: {current_time}")
        print()
        
        # 获取最新榜单（目前使用已知列表）
        latest_companies = self.search_for_list()
        
        # 比较发现新公司
        new_companies = []
        for name, info in latest_companies.items():
            if name not in self.state["known_companies"]:
                new_companies.append({
                    "name": name,
                    "info": info,
                    "discovered_at": current_time
                })
                print(f"🎉 发现新公司: {name}")
                print(f"   类别: {info.get('category', 'N/A')}")
                print(f"   ARR: {info.get('arr', 'N/A')}")
                print(f"   备注: {info.get('notes', 'N/A')}")
                print()
        
        # 更新状态
        if new_companies:
            self.state["new_companies"].extend(new_companies)
            self.state["known_companies"].update(latest_companies)
        
        self.state["last_check"] = current_time
        self.state["check_history"].append({
            "time": current_time,
            "new_companies_count": len(new_companies)
        })
        
        self.save_state()
        
        print("=" * 60)
        print(f"✅ 检查完成")
        print(f"📊 已知公司总数: {len(self.state['known_companies'])}")
        print(f"🆕 本次发现新公司: {len(new_companies)}")
        print(f"📚 历史新公司总数: {len(self.state['new_companies'])}")
        print("=" * 60)
        
        return new_companies
    
    def send_feishu_notification(self, new_companies):
        """发送飞书通知
        
        TODO: 集成 OpenClaw message send 能力
        """
        if not new_companies:
            return
        
        print()
        print("📧 准备发送飞书通知...")
        print("⚠️ 需要集成 OpenClaw message send 能力")
        print()
        
        # 构建通知内容
        message = "🔔 Top Lean AI 榜单更新!\n\n"
        message += f"榜单链接: {self.leaderboard_url}\n\n"
        message += f"发现 {len(new_companies)} 家新公司:\n\n"
        
        for company in new_companies:
            message += f"🚀 {company['name']}\n"
            message += f"   类别: {company['info'].get('category', 'N/A')}\n"
            message += f"   ARR: {company['info'].get('arr', 'N/A')}\n"
            message += f"   备注: {company['info'].get('notes', 'N/A')}\n\n"
        
        print(message)
        print("TODO: 使用 openclaw message send 发送到飞书")
    
    def get_status(self):
        """获取当前状态"""
        return {
            "last_check": self.state["last_check"],
            "known_companies_count": len(self.state["known_companies"]),
            "new_companies_count": len(self.state["new_companies"]),
            "check_count": len(self.state["check_history"]),
            "new_companies": self.state["new_companies"],
            "leaderboard_url": self.leaderboard_url,
            "qualification_criteria": self.qualification_criteria
        }
    
    def print_report(self):
        """打印报告"""
        status = self.get_status()
        
        print("=" * 60)
        print("📊 Top Lean AI 榜单监控状态")
        print("=" * 60)
        print()
        print(f"📊 榜单 URL: {status['leaderboard_url']}")
        print(f"🕐 上次检查: {status['last_check'] or '从未检查'}")
        print(f"🏢 已知公司总数: {status['known_companies_count']}")
        print(f"🆕 历史新公司数: {status['new_companies_count']}")
        print(f"🔍 检查次数: {status['check_count']}")
        print()
        print("📋 资格标准:")
        print(f"   - 超过 {status['qualification_criteria']['min_arr']}")
        print(f"   - 少于 {status['qualification_criteria']['max_employees']} 名员工")
        print(f"   - 成立不到 {status['qualification_criteria']['max_age_years']} 年")
        print()
        print("👤 创建者:")
        print(f"   - LinkedIn: {status['qualification_criteria']['linkedin']}")
        print(f"   - X (Twitter): {status['qualification_criteria']['x_twitter']}")
        print()
        
        if status['new_companies']:
            print("🆕 最近发现的新公司:")
            for company in status['new_companies'][-5:]:  # 只显示最近 5 家
                print(f"  - {company['name']} ({company['discovered_at'][:10]})")
            print()
        
        print("=" * 60)


def main():
    """主函数"""
    import sys
    
    monitor = TopLeanAIMonitor()
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "status":
            monitor.print_report()
        elif command == "check":
            new_companies = monitor.check_for_updates()
            if new_companies:
                monitor.send_feishu_notification(new_companies)
        elif command == "list":
            print("🏢 已知公司列表:")
            print()
            for name, info in monitor.state["known_companies"].items():
                print(f"  - {name}")
                print(f"    类别: {info.get('category', 'N/A')}")
                print(f"    ARR: {info.get('arr', 'N/A')}")
                print(f"    备注: {info.get('notes', 'N/A')}")
                print()
        else:
            print(f"未知命令: {command}")
            print("使用:")
            print("  python top-lean-ai-monitor.py status   # 查看状态")
            print("  python top-lean-ai-monitor.py check    # 检查更新")
            print("  python top-lean-ai-monitor.py list     # 列出已知公司")
    else:
        monitor.print_report()


if __name__ == "__main__":
    main()
