# Copyright (c) 2025 Beijing Volcano Engine Technology Co., Ltd. and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import asyncio
import json
from volcenginesdkarkruntime._models import BaseModel
from volcenginesdkarkruntime import AsyncArk
from veadk.utils.logger import get_logger
from httpx import Timeout
from veadk.config import getenv, settings

logger = get_logger(__name__)


async def link_reader(url_list: list[str]) -> dict:
    """
    Use this tool when you need to fetch content from web pages, PDFs, or Douyin videos.
    It retrieves the title and main content from the provided URLs.

    Examples: {"url_list": ["abc.com", "xyz.com"]}
    Args:
        url_list (list[str]): A list of URLs to parse (maximum 3).
    Returns:
        list[dict]: A list of dictionaries, each containing the title and content of the corresponding URL.
    """
    logger.debug(f"link_reader url_list: {url_list}")
    try:
        client = AsyncArk(
            api_key=getenv("MODEL_AGENT_API_KEY", settings.model.api_key),
            timeout=Timeout(connect=1.0, timeout=60.0),
        )
    except Exception as e:
        logger.error(f"link_reader client init failed:{e}")
        return []

    body = {
        "action_name": "LinkReader",
        "tool_name": "LinkReader",
        "parameters": {"url_list": url_list},
    }

    response = None
    try:
        response = await client.post(
            path="/tools/execute", body=body, cast_to=BaseModel
        )
        response = response.model_dump()
        logger.debug(f"link_reader response: {response}")

        if response["status_code"] != 200:
            logger.error(f"link_reader failed: {response}")
            return []
        else:
            return response["data"]["ark_web_data_list"]
    except Exception as e:
        logger.error(f"link_reader failed: {e}, response: {response}")
        return []


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python link_reader.py <url1> [url2] ...")
        sys.exit(1)

    urls = sys.argv[1:]
    # Run the async function
    result = asyncio.run(link_reader(urls))
    # Print result as JSON string
    print(json.dumps(result, ensure_ascii=False, indent=2))
