# Playwright Scraper Skill

使用 Playwright 进行网页抓取的 skill，支持模拟浏览器访问网页。

## 功能特性

- 使用 Playwright 模拟浏览器访问网页
- 支持无头模式和有头模式
- 支持等待特定元素出现
- 支持保存 HTML 和 JSON 结果
- 自动设置 User-Agent

## 使用方法

```bash
# 基本用法
python3 main.py <url>

# 不使用无头模式（显示浏览器）
python3 main.py <url> --no-headless

# 自定义等待时间
python3 main.py <url> --wait-time 10000

# 保存 HTML 文件
python3 main.py <url> --save-html output.html

# 保存 JSON 结果
python3 main.py <url> --output result.json
```

## 示例

```bash
# 抓取 GitHub 首页
python3 main.py https://github.com

# 抓取知乎文章
python3 main.py https://zhuanlan.zhihu.com/p/123456789

# 保存结果
python3 main.py https://example.com --save-html example.html --output example.json
```

## 依赖

- Playwright
- Chromium 浏览器

安装方法：

```bash
pip install playwright
playwright install chromium
```
