#!/usr/bin/env python3
"""
简单的 Playwright 网页抓取脚本
基于 Playwright 的网页抓取，用于测试抓取知乎等网站
"""

import sys
import os
import json
import argparse
from pathlib import Path
from datetime import datetime


def scrape_with_playwright(url: str, headless: bool = True, wait_time: int = 5000) -> dict:
    """
    使用 Playwright 抓取网页
    
    Args:
        url: 要抓取的 URL
        headless: 是否使用无头模式
        wait_time: 等待时间（毫秒）
    
    Returns:
        包含抓取结果的字典
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("Error: Playwright not installed")
        print("Install with: pip install playwright")
        print("Then install browsers: playwright install chromium")
        sys.exit(1)
    
    result = {
        "url": url,
        "title": "",
        "content": "",
        "elapsed_seconds": 0.0,
        "success": False,
        "error": None
    }
    
    start_time = datetime.now()
    
    try:
        with sync_playwright() as p:
            # 启动浏览器
            browser = p.chromium.launch(headless=headless)
            page = browser.new_page()
            
            # 设置 User-Agent
            page.set_extra_http_headers({
                "User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1"
            })
            
            # 访问页面
            print(f"Accessing: {url}")
            page.goto(url, wait_until="networkidle", timeout=30000)
            
            # 等待一段时间
            print(f"Waiting {wait_time}ms...")
            page.wait_for_timeout(wait_time)
            
            # 获取标题
            result["title"] = page.title()
            
            # 获取内容
            result["content"] = page.content()
            
            result["success"] = True
            
            # 关闭浏览器
            browser.close()
            
    except Exception as e:
        result["error"] = str(e)
        print(f"Error: {e}")
    
    # 计算耗时
    elapsed = (datetime.now() - start_time).total_seconds()
    result["elapsed_seconds"] = round(elapsed, 2)
    
    return result


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Playwright 网页抓取工具")
    parser.add_argument("url", help="要抓取的 URL")
    parser.add_argument("--headless", action="store_true", default=True, help="使用无头模式（默认：True）")
    parser.add_argument("--no-headless", action="store_false", dest="headless", help="不使用无头模式（显示浏览器）")
    parser.add_argument("--wait-time", type=int, default=5000, help="等待时间（毫秒，默认：5000）")
    parser.add_argument("--output", help="输出文件路径（JSON 格式）")
    parser.add_argument("--save-html", help="保存 HTML 文件的路径")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🕷️ Playwright 网页抓取")
    print("=" * 60)
    print()
    
    # 抓取网页
    result = scrape_with_playwright(
        url=args.url,
        headless=args.headless,
        wait_time=args.wait_time
    )
    
    print()
    print("=" * 60)
    print("📊 抓取结果")
    print("=" * 60)
    print(f"URL: {result['url']}")
    print(f"标题: {result['title']}")
    print(f"耗时: {result['elapsed_seconds']} 秒")
    print(f"成功: {'✅' if result['success'] else '❌'}")
    
    if result.get('error'):
        print(f"错误: {result['error']}")
    
    print()
    
    # 保存 HTML
    if args.save_html and result.get('content'):
        html_path = Path(args.save_html)
        html_path.parent.mkdir(parents=True, exist_ok=True)
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(result['content'])
        print(f"✅ HTML 已保存到: {html_path}")
    
    # 保存 JSON
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"✅ 结果已保存到: {output_path}")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
