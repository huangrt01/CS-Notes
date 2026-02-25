# Agent-Browser 适用场合分析

## 概述

agent-browser 是一个快速的浏览器自动化 CLI 工具，专为 AI 代理设计。它基于 Rust 开发，提供了丰富的网页操作功能，可以极大地提升 AI 助手的网页交互能力。

## 核心功能

### 1. 网页导航和操作

- `open <url>`：导航到 URL
- `click <sel>`：点击元素（或 @ref）
- `dblclick <sel>`：双击元素
- `type <sel> <text>`：在元素中输入文本
- `fill <sel> <text>`：清除并填充
- `press <key>`：按键（Enter、Tab、Control+a）
- `keyboard type <text>`：使用真实按键输入文本（无选择器）
- `keyboard inserttext <text>`：插入文本，无按键事件
- `hover <sel>`：悬停元素
- `focus <sel>`：聚焦元素
- `check <sel>`：勾选复选框
- `uncheck <sel>`：取消勾选复选框
- `select <sel> <val...>`：选择下拉选项
- `drag <src> <dst>`：拖拽
- `upload <sel> <files...>`：上传文件
- `download <sel> <path>`：点击元素下载文件
- `scroll <dir> [px]`：滚动（上/下/左/右）
- `scrollintoview <sel>`：将元素滚动到视图中
- `wait <sel|ms>`：等待元素或时间

### 2. 网页信息获取

- `agent-browser get text <sel>`：获取文本
- `agent-browser get html <sel>`：获取 HTML
- `agent-browser get value <sel>`：获取值
- `agent-browser get attr <name> <sel>`：获取属性
- `agent-browser get title`：获取标题
- `agent-browser get url`：获取 URL
- `agent-browser get count <sel>`：获取数量
- `agent-browser get box <sel>`：获取边界框
- `agent-browser get styles <sel>`：获取样式

### 3. 网页状态检查

- `agent-browser is visible <sel>`：检查是否可见
- `agent-browser is enabled <sel>`：检查是否启用
- `agent-browser is checked <sel>`：检查是否勾选

### 4. 网页元素查找

- `agent-browser find role <value> <action> [text]`：按角色查找
- `agent-browser find text <value> <action> [text]`：按文本查找
- `agent-browser find label <value> <action> [text]`：按标签查找
- `agent-browser find placeholder <value> <action> [text]`：按占位符查找
- `agent-browser find alt <value> <action> [text]`：按替代文本查找
- `agent-browser find title <value> <action> [text]`：按标题查找
- `agent-browser find testid <value> <action> [text]`：按测试 ID 查找
- `agent-browser find first <action> [text]`：查找第一个
- `agent-browser find last <action> [text]`：查找最后一个
- `agent-browser find nth <n> <action> [text]`：查找第 n 个

### 5. 截图和 PDF

- `screenshot [path]`：截图
- `pdf <path>`：保存为 PDF
- `screenshot --full`：全屏截图
- `screenshot --annotate`：带编号标签和图例的注释截图

### 6. 可访问性树

- `snapshot`：带引用的可访问性树（用于 AI）
- `snapshot -i`：仅交互元素
- `snapshot -c`：移除空结构元素
- `snapshot -d <n>`：限制树深度
- `snapshot -s <sel>`：范围到 CSS 选择器

### 7. 网络操作

- `agent-browser network route <url> [--abort|--body <json>]`：路由 URL
- `agent-browser network unroute [url]`：取消路由
- `agent-browser network requests [--clear] [--filter <pattern>]`：请求

### 8. 存储管理

- `agent-browser cookies [get|set|clear]`：管理 Cookie（set 支持 --url、--domain、--path、--httpOnly、--secure、--sameSite、--expires）
- `agent-browser storage <local|session>`：管理 Web 存储

### 9. 标签页管理

- `agent-browser tab [new|list|close|<n>]`：管理标签页

### 10. 调试功能

- `agent-browser trace start|stop [path]`：录制 Playwright 跟踪
- `agent-browser profiler start|stop [path]`：录制 Chrome DevTools 性能分析
- `agent-browser record start <path> [url]`：开始视频录制（WebM）
- `agent-browser record stop`：停止并保存视频
- `agent-browser console [--clear]`：查看控制台日志
- `agent-browser errors [--clear]`：查看页面错误
- `agent-browser highlight <sel>`：高亮元素

### 11. 会话管理

- `agent-browser session`：显示当前会话名称
- `agent-browser session list`：列出活动会话

### 12. 浏览器设置

- `agent-browser set viewport <w> <h>`：设置视口
- `agent-browser set device <name>`：设置设备
- `agent-browser set geo <lat> <lng>`：设置地理位置
- `agent-browser set offline [on|off]`：设置离线模式
- `agent-browser set headers <json>`：设置请求头
- `agent-browser set credentials <user> <pass>`：设置凭证
- `agent-browser set media [dark|light] [reduced-motion]`：设置媒体偏好

### 13. 鼠标操作

- `agent-browser mouse move <x> <y>`：移动鼠标
- `agent-browser mouse down [btn]`：按下鼠标
- `agent-browser mouse up [btn]`：抬起鼠标
- `agent-browser mouse wheel <dy> [dx]`：滚轮

## 对本项目的具体优化场景

### 1. 笔记整理场景

**场景描述**：从网页、知乎文章、小红书文章、微信公众号文章等整理笔记。

**agent-browser 的价值**：
- 替代 Playwright，提供更简洁的 CLI 接口
- 支持 snapshot 功能，获取可访问性树，便于 AI 理解网页结构
- 支持截图和 PDF 保存，便于后续参考
- 支持获取网页文本、HTML、属性等信息，便于提取内容

**具体 Todo**：
- 用 agent-browser 替代现有的 Playwright 脚本
- 创建一个基于 agent-browser 的网页抓取 skill
- 测试 agent-browser 在整理笔记场景中的效果

### 2. 联网搜索场景

**场景描述**：使用联网搜索能力，搜索和下载资料。

**agent-browser 的价值**：
- 支持打开搜索页面，输入搜索关键词
- 支持点击搜索结果，浏览详细内容
- 支持获取搜索结果的文本、链接等信息
- 支持截图保存搜索结果

**具体 Todo**：
- 用 agent-browser 增强现有的 search-web skill
- 用 agent-browser 增强现有的 ask-echo skill
- 创建一个基于 agent-browser 的联网搜索 skill

### 3. Todo Web Manager 场景

**场景描述**：Todo Web Manager 需要访问公网，但是有安全隐患。

**agent-browser 的价值**：
- 支持通过 CDP 连接浏览器（--cdp <port>）
- 支持自动发现和连接正在运行的 Chrome（--auto-connect）
- 可以在用户的浏览器中操作，避免开公网端口的安全隐患

**具体 Todo**：
- 探索用 agent-browser 替代 Todo Web Manager 的公网访问
- 设计通过 agent-browser 连接用户浏览器的方案
- 测试这个方案的可行性和安全性

### 4. 网页测试场景

**场景描述**：测试网页功能、验证网页内容。

**agent-browser 的价值**：
- 支持检查网页元素的状态（可见、启用、勾选）
- 支持点击、输入、选择等操作
- 支持截图和 PDF 保存，便于验证
- 支持录制 trace 和视频，便于调试

**具体 Todo**：
- 用 agent-browser 测试 Todo Web Manager 的功能
- 用 agent-browser 测试其他网页功能
- 创建一个基于 agent-browser 的网页测试 skill

### 5. 数据采集场景

**场景描述**：从网页采集数据，例如 Top Lean AI 榜单、AI 新闻等。

**agent-browser 的价值**：
- 支持获取网页文本、HTML、属性等信息
- 支持查找网页元素（按角色、文本、标签等）
- 支持截图和 PDF 保存，便于后续参考
- 支持网络请求拦截，便于分析网络流量

**具体 Todo**：
- 用 agent-browser 替代现有的 Top Lean AI 榜单监控脚本
- 用 agent-browser 采集 AI 新闻
- 创建一个基于 agent-browser 的数据采集 skill

### 6. 文档生成场景

**场景描述**：从网页生成文档，例如技术文档、产品文档等。

**agent-browser 的价值**：
- 支持获取网页文本、HTML、属性等信息
- 支持截图和 PDF 保存，便于文档生成
- 支持可访问性树，便于理解网页结构
- 支持运行 JavaScript，便于交互

**具体 Todo**：
- 用 agent-browser 生成技术文档
- 用 agent-browser 生成产品文档
- 创建一个基于 agent-browser 的文档生成 skill

## 适用场合总结

agent-browser 适用于以下场景：

1. **需要网页交互的场景**：点击、输入、选择等
2. **需要网页信息提取的场景**：获取文本、HTML、属性等
3. **需要网页验证的场景**：检查元素状态、截图验证等
4. **需要网页调试的场景**：录制 trace、视频、查看控制台等
5. **需要避免公网端口的场景**：通过 CDP 连接用户浏览器

## 推荐的 Todo 优先级

### P0 优先级（最高）

1. 用 agent-browser 替代现有的 Playwright 脚本
2. 探索用 agent-browser 替代 Todo Web Manager 的公网访问

### P1 优先级

1. 创建一个基于 agent-browser 的网页抓取 skill
2. 用 agent-browser 增强现有的 search-web skill
3. 用 agent-browser 替代现有的 Top Lean AI 榜单监控脚本

### P2 优先级

1. 用 agent-browser 测试 Todo Web Manager 的功能
2. 创建一个基于 agent-browser 的联网搜索 skill
3. 创建一个基于 agent-browser 的数据采集 skill

### P3 优先级

1. 用 agent-browser 生成技术文档
2. 用 agent-browser 生成产品文档
3. 创建一个基于 agent-browser 的文档生成 skill
4. 创建一个基于 agent-browser 的网页测试 skill

## 总结

agent-browser 是一个非常强大的浏览器自动化 CLI 工具，它可以极大地提升 AI 助手的网页交互能力。对于本项目来说，agent-browser 可以在笔记整理、联网搜索、Todo Web Manager、网页测试、数据采集、文档生成等多个场景中产生优化。

建议优先推进 P0 和 P1 优先级的 Todo，尽快让 agent-browser 在本项目中发挥价值。

---

*最后更新：2026-02-26*
