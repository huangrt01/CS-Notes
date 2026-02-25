#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ask-Echo Web Search - 融合信息搜索 API 客户端

使用火山引擎 Ask-Echo 融合信息 API 进行 web 搜索、web 搜索总结版和图片搜索
"""

import os
import sys
import json
import requests
from datetime import datetime
from typing import Optional, Dict, Any, List


def _load_env_file():
    """加载 .env 文件"""
    env_paths = [
        os.path.expanduser('~/.clawdbot/.env'),
        os.path.join(os.path.dirname(__file__), '../../.env'),
        '.env'
    ]
    for env_path in env_paths:
        if os.path.exists(env_path):
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line and '=' in line and not line.startswith('#'):
                        key, value = line.split('=', 1)
                        if key not in os.environ:
                            os.environ[key] = value
            break

_load_env_file()


class AskEchoSearch:
    """Ask-Echo 融合信息搜索 API 客户端"""

    def __init__(self):
        # 使用 VOLCENGINE_ASK_ECHO 进行鉴权
        self.api_key = os.getenv("VOLCENGINE_ASK_ECHO_APIKEY")

        # API 端点
        self.api_url = "https://open.feedcoopapi.com/search_api/web_search"

        if not self.api_key:
            raise ValueError("VOLCENGINE_ASK_ECHO environment variable is required")

    def search(
        self,
        query: str,
        search_type: str = "web",
        count: int = 10,
        need_summary: bool = False,
        need_content: bool = False,
        need_url: bool = False,
        sites: Optional[str] = None,
        block_hosts: Optional[str] = None,
        auth_level: int = 0,
        time_range: Optional[str] = None,
        query_rewrite: bool = False,
        industry: Optional[str] = None,
        # 图片搜索参数
        image_width_min: Optional[int] = None,
        image_width_max: Optional[int] = None,
        image_height_min: Optional[int] = None,
        image_height_max: Optional[int] = None,
        image_shapes: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        执行搜索

        Args:
            query: 搜索查询（1-100 字符）
            search_type: 搜索类型（web/web_summary/image）
            count: 返回条数（web 最多 50，image 最多 5）
            need_summary: 是否需要精准摘要
            need_content: 是否仅返回有正文的结果
            need_url: 是否仅返回有 URL 的结果
            sites: 指定搜索站点范围（多个用 | 分隔）
            block_hosts: 屏蔽的站点（多个用 | 分隔）
            auth_level: 权威度等级（0=不限制，1=非常权威）
            time_range: 时间范围（OneDay/OneWeek/OneMonth/OneYear/日期范围）
            query_rewrite: 是否开启 Query 改写
            industry: 行业类型（finance/game）
            image_width_min: 图片最小宽度
            image_width_max: 图片最大宽度
            image_height_min: 图片最小高度
            image_height_max: 图片最大高度
            image_shapes: 允许的图片形状（横长方形/竖长方形/方形）

        Returns:
            API 响应结果
        """
        if not query:
            raise ValueError("Query cannot be empty")

        # 构造请求体
        payload = {
            "Query": query[:100],  # 限制 100 字符
            "SearchType": search_type,
            "Count": count,
        }

        # 过滤条件
        filter_obj = {}
        if need_content:
            filter_obj["NeedContent"] = True
        if need_url:
            filter_obj["NeedUrl"] = True
        if sites:
            filter_obj["Sites"] = sites
        if block_hosts:
            filter_obj["BlockHosts"] = block_hosts
        if auth_level > 0:
            filter_obj["AuthInfoLevel"] = auth_level

        # 图片搜索过滤条件
        if search_type == "image":
            if image_width_min:
                filter_obj["ImageWidthMin"] = image_width_min
            if image_width_max:
                filter_obj["ImageWidthMax"] = image_width_max
            if image_height_min:
                filter_obj["ImageHeightMin"] = image_height_min
            if image_height_max:
                filter_obj["ImageHeightMax"] = image_height_max
            if image_shapes:
                filter_obj["ImageShapes"] = image_shapes

        if filter_obj:
            payload["Filter"] = filter_obj

        # 其他参数
        if need_summary:
            payload["NeedSummary"] = True

        if time_range:
            payload["TimeRange"] = time_range

        if query_rewrite:
            payload["QueryControl"] = {"QueryRewrite": True}

        if industry:
            payload["Industry"] = industry

        # 构造请求头 - 使用 API Key + Bearer 鉴权
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        try:
            # web_summary 使用流式响应
            if search_type == "web_summary":
                return self._stream_request(self.api_url, headers, payload)
            else:
                return self._normal_request(self.api_url, headers, payload)

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Request failed: {e}")

    def _normal_request(self, url: str, headers: Dict, payload: Dict) -> Dict[str, Any]:
        """普通请求（非流式）"""
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()

        # 检查 API 错误
        if 'ResponseMetadata' in data and 'Error' in data['ResponseMetadata']:
            error = data['ResponseMetadata']['Error']
            error_code = error.get('Code', 'Unknown')
            error_msg = error.get('Message', 'Unknown error')
            raise RuntimeError(f"API Error [{error_code}]: {error_msg}")

        # 提取 Result 部分
        if 'Result' in data:
            if data['Result'] is None:
                raise RuntimeError("API returned null result")
            return data['Result']
        else:
            return data

    def _stream_request(self, url: str, headers: Dict, payload: Dict) -> Dict[str, Any]:
        """流式请求（用于 web_summary）"""
        response = requests.post(url, headers=headers, json=payload, stream=True, timeout=30)
        response.raise_for_status()

        result = None
        full_content = ""
        choices = None
        usage = None

        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data:'):
                    data_str = line[5:]
                    if data_str == '[DONE]':
                        break

                    try:
                        data = json.loads(data_str)

                        # 提取 Result
                        if 'Result' in data and data['Result'] is not None:
                            result = data['Result']

                        # 提取 Choices（LLM 总结）
                        if 'Result' in data and data['Result'] and 'Choices' in data['Result']:
                            choices = data['Result']['Choices']

                        # 提取 Usage
                        if 'Result' in data and data['Result'] and 'Usage' in data['Result']:
                            usage = data['Result']['Usage']

                    except json.JSONDecodeError:
                        continue

        # 构造返回结果
        return {
            'ResultCount': result.get('ResultCount', 0) if result else 0,
            'WebResults': result.get('WebResults', []) if result else [],
            'Choices': choices,
            'Usage': usage
        }

    def format_result(self, result: Dict[str, Any], search_type: str) -> str:
        """格式化搜索结果"""
        output = []

        # 检查 result 是否为空
        if not result:
            return "No results returned from API."

        # Web 搜索总结版 - 显示 LLM 总结
        if search_type == "web_summary" and 'Choices' in result and result['Choices']:
            output.append("=== LLM Summary ===\n")
            choice = result['Choices'][0]
            if 'Message' in choice and choice['Message']:
                message = choice['Message']
                if isinstance(message, dict) and 'content' in message:
                    output.append(f"{message['content']}\n")
            elif 'Delta' in choice and choice['Delta']:
                # 流式响应的 Delta
                delta = choice['Delta']
                if isinstance(delta, dict) and 'content' in delta:
                    output.append(f"{delta['content']}\n")

        # Web 搜索结果
        if 'WebResults' in result and result['WebResults']:
            output.append(f"=== Search Results ({len(result['WebResults'])} items) ===\n")
            for i, item in enumerate(result['WebResults'], 1):
                output.append(f"\n[{i}] {item.get('Title', 'N/A')}")
                if item.get('SiteName'):
                    output.append(f"    Site: {item['SiteName']}")
                if item.get('Url'):
                    output.append(f"    URL: {item['Url']}")
                if item.get('PublishTime'):
                    output.append(f"    Published: {item['PublishTime']}")
                if item.get('AuthInfoDes'):
                    auth_level = item.get('AuthInfoLevel', 0)
                    output.append(f"    Authority: {item['AuthInfoDes']} (Level: {auth_level})")

                # 摘要
                if item.get('Summary'):
                    output.append(f"\n    Summary:")
                    output.append(f"    {item['Summary'][:500]}...")  # 限制长度

                # 普通摘要
                elif item.get('Snippet'):
                    output.append(f"\n    Snippet:")
                    output.append(f"    {item['Snippet']}")

                # 正文
                if item.get('Content'):
                    content = item['Content']
                    if len(content) > 300:
                        content = content[:300] + "..."
                    output.append(f"\n    Content: {content}")

        # 图片搜索结果
        if 'ImageResults' in result and result['ImageResults']:
            output.append(f"=== Image Results ({len(result['ImageResults'])} items) ===\n")
            for i, item in enumerate(result['ImageResults'], 1):
                output.append(f"\n[{i}] {item.get('Title', 'N/A')}")
                if item.get('SiteName'):
                    output.append(f"    Site: {item['SiteName']}")
                if item.get('Url'):
                    output.append(f"    URL: {item['Url']}")
                if item.get('PublishTime'):
                    output.append(f"    Published: {item['PublishTime']}")

                # 图片信息
                if 'Image' in item:
                    img = item['Image']
                    output.append(f"\n    Image Info:")
                    output.append(f"      URL: {img.get('Url', 'N/A')}")
                    if img.get('Width') and img.get('Height'):
                        output.append(f"      Size: {img['Width']}x{img['Height']}")
                    if img.get('Shape'):
                        output.append(f"      Shape: {img['Shape']}")

        # 搜索上下文
        if 'SearchContext' in result:
            ctx = result['SearchContext']
            output.append("\n=== Search Info ===")
            output.append(f"  Query: {ctx.get('OriginQuery', 'N/A')}")
            output.append(f"  Type: {ctx.get('SearchType', 'N/A')}")

        # 耗时
        if 'TimeCost' in result:
            output.append(f"  Time Cost: {result['TimeCost']} ms")

        # Log ID
        if 'LogId' in result:
            output.append(f"  Log ID: {result['LogId']}")

        # Token 使用情况（仅 web_summary）
        if 'Usage' in result and result['Usage']:
            usage = result['Usage']
            output.append("\n=== Token Usage ===")
            output.append(f"  Prompt tokens: {usage.get('PromptTokens', 0)}")
            output.append(f"  Completion tokens: {usage.get('CompletionTokens', 0)}")
            output.append(f"  Total tokens: {usage.get('TotalTokens', 0)}")
            output.append(f"  Search time: {usage.get('SearchTimeCost', 0)} ms")
            output.append(f"  First token time: {usage.get('FirstTokenTimeCost', 0)} ms")
            output.append(f"  Total time: {usage.get('TotalTimeCost', 0)} ms")

        return "\n".join(output)


def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("Usage: python ask_echo.py <query> [options]")
        print("\nEnvironment variables:")
        print("  VOLCENGINE_ASK_ECHO      (required) - VolcEngine Ask-Echo API Key")
        print("\nOptions:")
        print("  --type <type>             Search type: web (default), web_summary, image")
        print("  --count <int>             Result count (web: max 50, image: max 5)")
        print("  --need-summary            Enable precise summary")
        print("  --need-content            Only return results with content")
        print("  --time-range <range>      Time range: OneDay/OneWeek/OneMonth/OneYear")
        print("  --sites <domains>         Specify sites (separate with |)")
        print("  --block-hosts <domains>   Block hosts (separate with |)")
        print("  --auth-level <int>        Authority level (0=all, 1=very authoritative)")
        print("  --query-rewrite           Enable query rewrite")
        print("  --industry <type>         Industry: finance/game")
        print("\nImage search options:")
        print("  --image-width-min <int>   Minimum image width")
        print("  --image-width-max <int>   Maximum image width")
        print("  --image-height-min <int>  Minimum image height")
        print("  --image-height-max <int>  Maximum image height")
        print("  --image-shapes <list>     Allowed shapes: 横长方形,竖长方形,方形")
        sys.exit(1)

    query = sys.argv[1]

    # 解析参数
    search_type = "web"
    count = 10
    need_summary = False
    need_content = False
    need_url = False
    sites = None
    block_hosts = None
    auth_level = 0
    time_range = None
    query_rewrite = False
    industry = None
    image_width_min = None
    image_width_max = None
    image_height_min = None
    image_height_max = None
    image_shapes = None

    i = 2
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == '--type' and i + 1 < len(sys.argv):
            search_type = sys.argv[i + 1]
            i += 2
        elif arg == '--count' and i + 1 < len(sys.argv):
            count = int(sys.argv[i + 1])
            i += 2
        elif arg == '--need-summary':
            need_summary = True
            i += 1
        elif arg == '--need-content':
            need_content = True
            i += 1
        elif arg == '--time-range' and i + 1 < len(sys.argv):
            time_range = sys.argv[i + 1]
            i += 2
        elif arg == '--sites' and i + 1 < len(sys.argv):
            sites = sys.argv[i + 1]
            i += 2
        elif arg == '--block-hosts' and i + 1 < len(sys.argv):
            block_hosts = sys.argv[i + 1]
            i += 2
        elif arg == '--auth-level' and i + 1 < len(sys.argv):
            auth_level = int(sys.argv[i + 1])
            i += 2
        elif arg == '--query-rewrite':
            query_rewrite = True
            i += 1
        elif arg == '--industry' and i + 1 < len(sys.argv):
            industry = sys.argv[i + 1]
            i += 2
        elif arg == '--image-width-min' and i + 1 < len(sys.argv):
            image_width_min = int(sys.argv[i + 1])
            i += 2
        elif arg == '--image-width-max' and i + 1 < len(sys.argv):
            image_width_max = int(sys.argv[i + 1])
            i += 2
        elif arg == '--image-height-min' and i + 1 < len(sys.argv):
            image_height_min = int(sys.argv[i + 1])
            i += 2
        elif arg == '--image-height-max' and i + 1 < len(sys.argv):
            image_height_max = int(sys.argv[i + 1])
            i += 2
        elif arg == '--image-shapes' and i + 1 < len(sys.argv):
            image_shapes = sys.argv[i + 1].split(',')
            i += 2
        else:
            i += 1

    # 验证搜索类型
    if search_type not in ['web', 'web_summary', 'image']:
        print(f"Error: Invalid search type '{search_type}'. Must be web, web_summary, or image.")
        sys.exit(1)

    # 验证 count
    if search_type == 'image' and count > 5:
        count = 5
        print("Warning: Image search max count is 5, using 5.")
    elif search_type in ['web', 'web_summary'] and count > 50:
        count = 50
        print("Warning: Web search max count is 50, using 50.")

    # web_summary 必须设置 need_summary
    if search_type == 'web_summary':
        need_summary = True

    try:
        client = AskEchoSearch()
        result = client.search(
            query=query,
            search_type=search_type,
            count=count,
            need_summary=need_summary,
            need_content=need_content,
            need_url=need_url,
            sites=sites,
            block_hosts=block_hosts,
            auth_level=auth_level,
            time_range=time_range,
            query_rewrite=query_rewrite,
            industry=industry,
            image_width_min=image_width_min,
            image_width_max=image_width_max,
            image_height_min=image_height_min,
            image_height_max=image_height_max,
            image_shapes=image_shapes,
        )

        print(client.format_result(result, search_type))

    except ValueError as e:
        print(f"\nConfiguration Error: {e}")
        sys.exit(1)
    except RuntimeError as e:
        print(f"\nError: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
