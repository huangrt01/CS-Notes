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
        import re
        from urllib.request import urlopen
        from urllib.parse import urlparse, parse_qs
        
        try:
            url = blogger["url"]
            name = blogger["name"]
            
            # 从URL中提取信息
            # 格式: https://space.bilibili.com/3546813525134159/upload/video
            # 或者: https://www.bilibili.com/video/BV11m421M7N4
            
            parsed = urlparse(url)
            path_parts = parsed.path.strip("/").split("/")
            
            # 检查是否是space URL
            if "space.bilibili.com" in url and len(path_parts) >= 1:
                uid = path_parts[0]
                if uid.isdigit():
                    # 使用B站RSS feed
                    feed_url = f"https://api.bilibili.com/x/space/article?mid={uid}"
                    
                    try:
                        with urlopen(feed_url, timeout=10) as response:
                            content = response.read().decode("utf-8", errors="ignore")
                            
                            # 简单解析JSON
                            # B站API返回JSON格式
                            import json
                            try:
                                data = json.loads(content)
                                if data.get("code") == 0 and data.get("data"):
                                    articles = data.get("data", {}).get("articles", [])
                                    if articles and len(articles) > 0:
                                        latest = articles[0]
                                        latest_title = latest.get("title", "")
                                        latest_id = latest.get("id", "")
                                        latest_url = f"https://www.bilibili.com/read/cv{latest_id}"
                                        
                                        # 检查是否有新更新
                                        state_key = f"bilibili_{uid}"
                                        last_id = self.state["bloggers"].get(state_key, {}).get("last_id")
                                        
                                        if last_id != str(latest_id):
                                            # 有新更新
                                            self.state["bloggers"][state_key] = {
                                                "last_id": str(latest_id),
                                                "last_checked": datetime.datetime.now().isoformat()
                                            }
                                            
                                            return {
                                                "blogger": name,
                                                "platform": "bilibili",
                                                "title": latest_title[:100],
                                                "url": latest_url,
                                                "date": latest.get("publish_time", "")
                                            }
                            except json.JSONDecodeError:
                                pass
                    except Exception:
                        pass
            
            return None
        except Exception as e:
            print(f"⚠️ 检查B站更新失败: {name} - {e}")
            return None
    
    def _check_youtube(self, blogger: Dict[str, str]) -> Optional[Dict[str, Any]]:
        """检查YouTube更新"""
        import re
        from urllib.request import urlopen
        
        try:
            url = blogger["url"]
            name = blogger["name"]
            
            # 从URL中提取channel ID或用户名
            # 格式: https://www.youtube.com/@PyTorch/videos
            # 或者: https://www.youtube.com/playlist?list=PLwAchVoh-4zNSI5UlKEkKCL5r_jJyrFeO
            
            channel_id = None
            playlist_id = None
            
            # 检查是否是playlist URL
            playlist_match = re.search(r'list=([a-zA-Z0-9_-]+)', url)
            if playlist_match:
                playlist_id = playlist_match.group(1)
            else:
                # 检查是否是@用户名格式
                at_match = re.search(r'@([a-zA-Z0-9_-]+)', url)
                if at_match:
                    username = at_match.group(1)
                    # 对于@用户名，我们需要先获取channel ID
                    # 暂时跳过，因为需要额外的API调用
                    return None
            
            # 如果有playlist ID，使用RSS feed
            if playlist_id:
                feed_url = f"https://www.youtube.com/feeds/videos.xml?playlist_id={playlist_id}"
            elif channel_id:
                feed_url = f"https://www.youtube.com/feeds/videos.xml?channel_id={channel_id}"
            else:
                return None
            
            with urlopen(feed_url, timeout=10) as response:
                content = response.read().decode("utf-8", errors="ignore")
                
                # 解析YouTube RSS feed
                # 查找最新的视频
                entry_match = re.search(
                    r'<entry[^>]*>.*?<title>([^<]+)</title>.*?<link[^>]*href=["\']([^"\']+)["\'].*?<published>([^<]+)</published>.*?</entry>',
                    content,
                    re.DOTALL | re.IGNORECASE
                )
                
                if entry_match:
                    latest_title = entry_match.group(1).strip()
                    latest_url = entry_match.group(2).strip()
                    latest_date = entry_match.group(3).strip()
                    
                    # 检查是否有新更新
                    state_key = f"youtube_{playlist_id or channel_id}"
                    last_title = self.state["bloggers"].get(state_key, {}).get("last_title")
                    
                    if last_title != latest_title:
                        # 有新更新
                        self.state["bloggers"][state_key] = {
                            "last_title": latest_title,
                            "last_checked": datetime.datetime.now().isoformat()
                        }
                        
                        return {
                            "blogger": name,
                            "platform": "youtube",
                            "title": latest_title[:100],
                            "url": latest_url,
                            "date": latest_date
                        }
            
            return None
        except Exception as e:
            print(f"⚠️ 检查YouTube更新失败: {name} - {e}")
            return None
    
    def _check_zhihu(self, blogger: Dict[str, str]) -> Optional[Dict[str, Any]]:
        """检查知乎更新"""
        import re
        from urllib.request import urlopen
        
        try:
            url = blogger["url"]
            name = blogger["name"]
            
            # 知乎的反爬机制比较严格，暂时只做简单的检查
            # 可以尝试获取页面内容，查找最新的文章
            
            state_key = f"zhihu_{url}"
            last_check = self.state["bloggers"].get(state_key, {}).get("last_check")
            
            # 暂时只记录检查时间，不做实际的更新检查
            # 知乎需要更复杂的反爬处理
            self.state["bloggers"][state_key] = {
                "last_check": datetime.datetime.now().isoformat()
            }
            
            return None
        except Exception as e:
            print(f"⚠️ 检查知乎更新失败: {name} - {e}")
            return None
    
    def _check_github(self, blogger: Dict[str, str]) -> Optional[Dict[str, Any]]:
        """检查GitHub更新"""
        import json
        from urllib.request import urlopen
        
        try:
            url = blogger["url"]
            name = blogger["name"]
            
            # 从URL中提取owner和repo
            # 格式: https://github.com/mli/paper-reading
            parts = url.rstrip("/").split("/")
            if len(parts) < 2:
                return None
            
            owner = parts[-2]
            repo = parts[-1]
            
            # 使用GitHub API获取最新的commit
            api_url = f"https://api.github.com/repos/{owner}/{repo}/commits?per_page=1"
            
            with urlopen(api_url, timeout=10) as response:
                data = json.loads(response.read().decode())
                
                if data and len(data) > 0:
                    latest_commit = data[0]
                    commit_sha = latest_commit["sha"]
                    commit_message = latest_commit["commit"]["message"]
                    commit_date = latest_commit["commit"]["author"]["date"]
                    
                    # 检查是否有新更新
                    state_key = f"github_{owner}_{repo}"
                    last_sha = self.state["bloggers"].get(state_key, {}).get("last_sha")
                    
                    if last_sha != commit_sha:
                        # 有新更新
                        self.state["bloggers"][state_key] = {
                            "last_sha": commit_sha,
                            "last_checked": datetime.datetime.now().isoformat()
                        }
                        
                        return {
                            "blogger": name,
                            "platform": "github",
                            "title": commit_message[:100],
                            "url": f"https://github.com/{owner}/{repo}/commit/{commit_sha}",
                            "date": commit_date
                        }
            
            return None
        except Exception as e:
            print(f"⚠️ 检查GitHub更新失败: {name} - {e}")
            return None
    
    def _check_blog(self, blogger: Dict[str, str]) -> Optional[Dict[str, Any]]:
        """检查博客更新（通用）"""
        import re
        from urllib.request import urlopen
        from urllib.parse import urljoin
        
        try:
            url = blogger["url"]
            name = blogger["name"]
            
            # 先尝试查找RSS feed
            feed_urls = [
                url.rstrip("/") + "/feed",
                url.rstrip("/") + "/rss",
                url.rstrip("/") + "/rss.xml",
                url.rstrip("/") + "/feed.xml",
                url.rstrip("/") + "/index.xml",
                url.rstrip("/") + "/atom.xml",
            ]
            
            # 也可以尝试从首页查找RSS链接
            try:
                with urlopen(url, timeout=10) as response:
                    html = response.read().decode("utf-8", errors="ignore")
                    
                    # 查找RSS链接
                    rss_links = re.findall(
                        r'<link[^>]*type=["\']application/(?:rss|atom)\+xml["\'][^>]*href=["\']([^"\']+)["\']',
                        html,
                        re.IGNORECASE
                    )
                    
                    if rss_links:
                        for link in rss_links:
                            feed_url = urljoin(url, link)
                            if feed_url not in feed_urls:
                                feed_urls.insert(0, feed_url)
            except Exception:
                pass
            
            # 尝试每个feed URL
            for feed_url in feed_urls:
                try:
                    with urlopen(feed_url, timeout=10) as response:
                        content = response.read().decode("utf-8", errors="ignore")
                        
                        # 简单解析RSS/Atom feed
                        # 查找最新的条目
                        title_match = re.search(
                            r'<item[^>]*>.*?<title>([^<]+)</title>.*?<link>([^<]+)</link>.*?<pubDate>([^<]+)</pubDate>.*?</item>',
                            content,
                            re.DOTALL | re.IGNORECASE
                        )
                        
                        if not title_match:
                            # 尝试Atom格式
                            title_match = re.search(
                                r'<entry[^>]*>.*?<title>([^<]+)</title>.*?<link[^>]*href=["\']([^"\']+)["\'].*?<updated>([^<]+)</updated>.*?</entry>',
                                content,
                                re.DOTALL | re.IGNORECASE
                            )
                        
                        if title_match:
                            latest_title = title_match.group(1).strip()
                            latest_url = title_match.group(2).strip()
                            latest_date = title_match.group(3).strip()
                            
                            # 检查是否有新更新
                            state_key = f"blog_{url}"
                            last_title = self.state["bloggers"].get(state_key, {}).get("last_title")
                            
                            if last_title != latest_title:
                                # 有新更新
                                self.state["bloggers"][state_key] = {
                                    "last_title": latest_title,
                                    "last_checked": datetime.datetime.now().isoformat()
                                }
                                
                                return {
                                    "blogger": name,
                                    "platform": "blog",
                                    "title": latest_title[:100],
                                    "url": latest_url if latest_url.startswith("http") else urljoin(url, latest_url),
                                    "date": latest_date
                                }
                    
                    break  # 找到一个feed就停止尝试
                except Exception:
                    continue
            
            return None
        except Exception as e:
            print(f"⚠️ 检查博客更新失败: {name} - {e}")
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
