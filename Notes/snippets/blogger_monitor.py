#!/usr/bin/env python3
"""
博主监控脚本 - 监控非技术知识.md里长期关注的博主更新
"""

import os
import json
import re
import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any


class BloggerMonitor:
    """博主监控器"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.workspace = Path("/root/.openclaw/workspace/CS-Notes")
        self.notes_file = self.workspace / "Notes" / "非技术知识.md"
        self.state_file = self.workspace / ".trae" / "logs" / "blogger_monitor_state.json"
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        
        # 博主状态
        self.state = self._load_state()
        
        # 支持的平台
        self.platforms = {
            "bilibili": self._check_bilibili,
            "youtube": self._check_youtube,
            "zhihu": self._check_zhihu,
            "github": self._check_github,
            "blog": self._check_blog,
        }
    
    def _load_state(self) -> Dict[str, Any]:
        """加载状态"""
        if self.state_file.exists():
            with open(self.state_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return {
            "last_check": None,
            "bloggers": {}
        }
    
    def _save_state(self):
        """保存状态"""
        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump(self.state, f, ensure_ascii=False, indent=2)
    
    def parse_bloggers_from_notes(self) -> List[Dict[str, str]]:
        """从非技术知识.md中解析博主列表"""
        bloggers = []
        
        with open(self.notes_file, "r", encoding="utf-8") as f:
            content = f.read()
        
        # 找到"### 持续关注"section，直到下一个"### "
        section_match = re.search(
            r"### 持续关注.*?(?=^### |\Z)",
            content,
            re.DOTALL | re.MULTILINE
        )
        
        if not section_match:
            print("⚠️ 没有找到'### 持续关注'section")
            return bloggers
        
        section_content = section_match.group(0)
        print(f"✅ 找到持续关注section，长度: {len(section_content)}")
        
        # 解析每个博主 - 支持多种格式
        # 格式1: * **博主名** 链接
        # 格式2: * **博主名**：链接 (中文冒号)
        # 格式3: * **博主名**: 链接 (英文冒号)
        # 格式4: * 博主名：链接 (没有加粗，中文冒号)
        # 格式5: * [博主名](链接) (Markdown链接格式)
        # 格式6: * 博主名 https://链接
        
        # 先处理 Markdown 链接格式: * [博主名](链接)
        markdown_link_pattern = re.compile(
            r"\*\s*\[(.*?)\]\((https?://[^\)]+)\)",
            re.MULTILINE
        )
        
        for match in markdown_link_pattern.finditer(section_content):
            name = match.group(1).strip()
            url = match.group(2).strip()
            
            if name and url:
                platform = self._detect_platform(url)
                bloggers.append({
                    "name": name,
                    "url": url,
                    "platform": platform
                })
                print(f"  找到博主 (Markdown链接): {name} ({platform}) - {url}")
        
        # 再处理其他格式
        # 格式: * **博主名** [：:]? 链接
        # 或者: * 博主名 [：:]? 链接
        other_pattern = re.compile(
            r"\*\s*(?:\*\*([^*]+)\*\*|([^*\[\]:]+))\s*(?:[:：]\s*)?(https?://[^\s\[\]]+)",
            re.MULTILINE
        )
        
        for match in other_pattern.finditer(section_content):
            name = match.group(1) or match.group(2)
            if name:
                name = name.strip()
            url = match.group(3).strip()
            
            # 检查是否已经通过 Markdown 链接格式添加过
            already_exists = any(b["url"] == url for b in bloggers)
            
            if name and url and not already_exists:
                platform = self._detect_platform(url)
                bloggers.append({
                    "name": name,
                    "url": url,
                    "platform": platform
                })
                print(f"  找到博主 (其他格式): {name} ({platform}) - {url}")
        
        return bloggers
    
    def _detect_platform(self, url: str) -> str:
        """检测平台类型"""
        if "bilibili.com" in url:
            return "bilibili"
        elif "youtube.com" in url or "youtu.be" in url:
            return "youtube"
        elif "zhihu.com" in url:
            return "zhihu"
        elif "github.com" in url:
            return "github"
        else:
            return "blog"
    
    def _check_bilibili(self, blogger: Dict[str, str]) -> Optional[Dict[str, Any]]:
        """检查B站更新"""
        # TODO: 实现B站更新检查
        # 需要处理B站的反爬机制
        return None
    
    def _check_youtube(self, blogger: Dict[str, str]) -> Optional[Dict[str, Any]]:
        """检查YouTube更新"""
        # TODO: 实现YouTube更新检查
        # 可以使用YouTube RSS feed
        return None
    
    def _check_zhihu(self, blogger: Dict[str, str]) -> Optional[Dict[str, Any]]:
        """检查知乎更新"""
        # TODO: 实现知乎更新检查
        return None
    
    def _check_github(self, blogger: Dict[str, str]) -> Optional[Dict[str, Any]]:
        """检查GitHub更新"""
        # TODO: 实现GitHub更新检查（需要安装 requests 库）
        # 暂时跳过，等待后续实现
        return None
    
    def _check_blog(self, blogger: Dict[str, str]) -> Optional[Dict[str, Any]]:
        """检查博客更新（通用）"""
        # TODO: 实现通用博客更新检查
        # 可以尝试查找RSS feed
        return None
    
    def check_updates(self) -> List[Dict[str, Any]]:
        """检查所有博主的更新"""
        updates = []
        bloggers = self.parse_bloggers_from_notes()
        
        for blogger in bloggers:
            platform = blogger["platform"]
            if platform in self.platforms:
                update = self.platforms[platform](blogger)
                if update:
                    updates.append(update)
        
        # 更新最后检查时间
        self.state["last_check"] = datetime.datetime.now().isoformat()
        self._save_state()
        
        return updates
    
    def get_status(self) -> Dict[str, Any]:
        """获取监控状态"""
        bloggers = self.parse_bloggers_from_notes()
        return {
            "last_check": self.state.get("last_check"),
            "total_bloggers": len(bloggers),
            "bloggers": bloggers,
            "state": self.state
        }


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="博主监控脚本")
    parser.add_argument("--check", action="store_true", help="检查更新")
    parser.add_argument("--status", action="store_true", help="查看状态")
    parser.add_argument("--list", action="store_true", help="列出所有博主")
    
    args = parser.parse_args()
    
    monitor = BloggerMonitor()
    
    if args.check:
        print("🔍 检查博主更新...")
        updates = monitor.check_updates()
        if updates:
            print(f"🎉 发现 {len(updates)} 个更新！")
            for update in updates:
                print(f"  - {update['blogger']}: {update['title']}")
        else:
            print("✅ 没有新更新")
    elif args.status:
        status = monitor.get_status()
        print("📊 博主监控状态")
        print(f"  最后检查: {status['last_check']}")
        print(f"  博主总数: {status['total_bloggers']}")
        print(f"  博主列表:")
        for blogger in status["bloggers"]:
            print(f"    - {blogger['name']} ({blogger['platform']}): {blogger['url']}")
    elif args.list:
        bloggers = monitor.parse_bloggers_from_notes()
        print("📋 博主列表")
        for blogger in bloggers:
            print(f"  - {blogger['name']} ({blogger['platform']}): {blogger['url']}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
