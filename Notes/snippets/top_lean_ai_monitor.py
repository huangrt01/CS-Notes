#!/usr/bin/env python3
"""
Top Lean AI 榜单监控脚本 - OpenClaw 集成版

专注于核心功能：
1. 从 Google Sheets 获取榜单数据
2. 检测新公司
3. 状态管理
4. 提供简洁的 API 供 OpenClaw 调用

数据源：https://leanaileaderboard.com/
数据来源：Google Sheets CSV 导出
"""

import os
import json
import csv
import io
import requests
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


class TopLeanAIMonitor:
    """Top Lean AI 榜单监控器"""
    
    def __init__(self, workspace_path: Optional[str] = None):
        self.workspace_path = Path(workspace_path) if workspace_path else Path.cwd()
        self.state_file = self.workspace_path / ".top-lean-ai-state.json"
        self.google_sheets_csv_url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQ1vMlwTJ8_Lty161T73uwnMzVxb48XzHxz9aPNla5OgCjd2yJ0HMfxEHGSv1OsyGOarWUYDcsJZfmk/pub?output=csv"
        self.state = self._load_state()
    
    def _load_state(self) -> Dict:
        """加载状态"""
        if self.state_file.exists():
            with open(self.state_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {
            "last_check": None,
            "known_companies": {},
            "new_companies": [],
            "check_history": []
        }
    
    def _save_state(self) -> None:
        """保存状态"""
        with open(self.state_file, 'w', encoding='utf-8') as f:
            json.dump(self.state, f, ensure_ascii=False, indent=2)
    
    def fetch_companies(self) -> Dict[str, Dict]:
        """
        从 Google Sheets 获取最新榜单数据
        
        Returns:
            公司字典，key 为公司名，value 为公司信息
        """
        session = requests.Session()
        response = session.get(self.google_sheets_csv_url, allow_redirects=True, timeout=30)
        response.raise_for_status()
        
        reader = csv.reader(io.StringIO(response.text))
        rows = list(reader)
        
        if len(rows) < 2:
            return {}
        
        headers = rows[0]
        companies = {}
        
        for row in rows[1:]:
            company = self._parse_row(headers, row)
            if company and 'name' in company:
                companies[company['name']] = company
        
        return companies
    
    def _parse_row(self, headers: List[str], cells: List[str]) -> Optional[Dict]:
        """解析单行数据"""
        company = {
            "rank": None,
            "name": None,
            "description": None,
            "location": None,
            "annual_revenue": None,
            "num_employees": None,
            "revenue_per_employee": None,
            "profitable": None,
            "total_funding": None,
            "valuation": None,
            "valuation_per_employee": None,
            "founded": None,
            "last_updated": None,
            "source": None
        }
        
        for idx, header in enumerate(headers):
            if idx >= len(cells):
                continue
            
            value = cells[idx].strip()
            header_lower = header.lower()
            
            if idx == 0 and value.isdigit():
                company['rank'] = value
            elif 'company' in header_lower:
                company['name'] = value
            elif 'description' in header_lower:
                company['description'] = value
            elif 'location' in header_lower:
                company['location'] = value
            elif 'annual' in header_lower and 'revenue' in header_lower:
                company['annual_revenue'] = value
            elif '#' in header and 'employee' in header_lower:
                company['num_employees'] = value
            elif 'revenue' in header_lower and 'employee' in header_lower:
                company['revenue_per_employee'] = value
            elif 'profitable' in header_lower:
                company['profitable'] = value
            elif 'total' in header_lower and 'funding' in header_lower:
                company['total_funding'] = value
            elif 'valuation' in header_lower and 'employee' in header_lower:
                company['valuation_per_employee'] = value
            elif 'valuation' in header_lower:
                company['valuation'] = value
            elif 'founded' in header_lower:
                company['founded'] = value
            elif 'last' in header_lower and 'updated' in header_lower:
                company['last_updated'] = value
            elif 'source' in header_lower:
                company['source'] = value
        
        if not company['name'] and len(cells) > 2:
            company['name'] = cells[2]
        
        return company if company['name'] else None
    
    def check_updates(self) -> Dict:
        """
        检查榜单更新
        
        Returns:
            包含更新信息的字典：
            {
                "success": bool,
                "timestamp": str,
                "new_companies": List[Dict],
                "total_companies": int,
                "known_companies_count": int
            }
        """
        current_time = datetime.now().isoformat()
        
        try:
            latest_companies = self.fetch_companies()
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "timestamp": current_time
            }
        
        new_companies = []
        for name, info in latest_companies.items():
            if name not in self.state["known_companies"]:
                new_companies.append({
                    "name": name,
                    "info": info,
                    "discovered_at": current_time
                })
        
        if new_companies:
            self.state["new_companies"].extend(new_companies)
            self.state["known_companies"].update(latest_companies)
        
        self.state["last_check"] = current_time
        self.state["check_history"].append({
            "time": current_time,
            "new_companies_count": len(new_companies),
            "total_companies": len(latest_companies)
        })
        
        self._save_state()
        
        return {
            "success": True,
            "timestamp": current_time,
            "new_companies": new_companies,
            "total_companies": len(latest_companies),
            "known_companies_count": len(self.state["known_companies"])
        }
    
    def get_status(self) -> Dict:
        """
        获取当前监控状态
        
        Returns:
            状态字典
        """
        return {
            "last_check": self.state["last_check"],
            "known_companies_count": len(self.state["known_companies"]),
            "new_companies_count": len(self.state["new_companies"]),
            "check_count": len(self.state["check_history"]),
            "recent_new_companies": self.state["new_companies"][-10:] if self.state["new_companies"] else []
        }
    
    def get_all_companies(self) -> Dict[str, Dict]:
        """获取所有已知公司"""
        return self.state["known_companies"].copy()
    
    def format_company_message(self, company: Dict) -> str:
        """
        格式化公司信息为可读消息
        
        Args:
            company: 公司信息字典
        
        Returns:
            格式化的消息字符串
        """
        info = company.get("info", company)
        msg = f"🚀 {company.get('name', info.get('name', 'Unknown'))}\n"
        
        if info.get('rank'):
            msg += f"   排名: {info['rank']}\n"
        if info.get('description'):
            msg += f"   描述: {info['description']}\n"
        if info.get('location'):
            msg += f"   位置: {info['location']}\n"
        if info.get('annual_revenue'):
            msg += f"   年收入: {info['annual_revenue']}\n"
        if info.get('num_employees'):
            msg += f"   员工数: {info['num_employees']}\n"
        if info.get('revenue_per_employee'):
            msg += f"   人均收入: {info['revenue_per_employee']}\n"
        if info.get('profitable'):
            msg += f"   盈利: {info['profitable']}\n"
        if info.get('valuation'):
            msg += f"   估值: {info['valuation']}\n"
        if info.get('founded'):
            msg += f"   成立: {info['founded']}\n"
        
        return msg


def main():
    """命令行入口（保留用于调试）"""
    import sys
    
    monitor = TopLeanAIMonitor()
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "status":
            status = monitor.get_status()
            print(json.dumps(status, ensure_ascii=False, indent=2))
        elif command == "check":
            result = monitor.check_updates()
            print(json.dumps(result, ensure_ascii=False, indent=2))
        elif command == "list":
            companies = monitor.get_all_companies()
            print(json.dumps(companies, ensure_ascii=False, indent=2))
        else:
            print(f"Unknown command: {command}")
            print("Usage:")
            print("  python top-lean-ai-monitor.py status  # Get status")
            print("  python top-lean-ai-monitor.py check   # Check for updates")
            print("  python top-lean-ai-monitor.py list    # List all companies")
    else:
        status = monitor.get_status()
        print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
