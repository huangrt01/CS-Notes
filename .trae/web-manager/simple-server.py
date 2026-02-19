#!/usr/bin/env python3
"""
简单的 HTTP 服务器 - 提供静态文件服务
"""

import http.server
import socketserver
import os
from pathlib import Path

# 配置
PORT = 5000
WEB_MANAGER_DIR = Path(__file__).parent

class MyHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_MANAGER_DIR), **kwargs)
    
    def end_headers(self):
        # 添加 CORS 头
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

def main():
    print("=" * 60)
    print("Todos Web Manager - 简单 HTTP 服务器")
    print("=" * 60)
    print(f"服务目录: {WEB_MANAGER_DIR}")
    print(f"访问地址: http://localhost:{PORT}")
    print("=" * 60)
    print("可用的文件:")
    print(f"  - index.html              (原版)")
    print(f"  - index-enhanced.html     (增强版 - 开发验证)")
    print(f"  - server.py               (Flask 后端 - 需要 flask)")
    print(f"  - simple-server.py        (简单 HTTP 服务器 - 当前)")
    print("=" * 60)
    print("提示: 用浏览器打开 index-enhanced.html")
    print("      然后使用 '加载 JSON 文件' 功能选择 .trae/todos/todos.json")
    print("=" * 60)
    
    with socketserver.TCPServer(("", PORT), MyHTTPRequestHandler) as httpd:
        print(f"服务器已启动，监听端口 {PORT}...")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n服务器已停止")
            httpd.shutdown()

if __name__ == "__main__":
    main()
