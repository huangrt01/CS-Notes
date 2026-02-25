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

import os
import sys
import json
import requests
from datetime import datetime

# API URL
API_URL = "https://open.feedcoopapi.com/agent_api/agent/chat/completion"


def search_web(question: str, stream: bool = False):
    """Search web and get AI agent response.

    Args:
        question: The question to search and answer.
        stream: Whether to use streaming response (default: False).
    """
    if not question:
        print("Question is empty.")
        return

    # Get API key
    api_key = os.getenv("MODEL_SEARCH_API_KEY") or os.getenv("ARK_API_KEY")

    # Get bot ID
    bot_id = os.getenv("SEARCH_BOT_ID")
    if not bot_id:
        print("SEARCH_BOT_ID environment variable is required.")
        print("Please set it to your bot ID from the console.")
        return

    # Prepare request
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "bot_id": bot_id,
        "stream": stream,
        "messages": [
            {
                "role": "user",
                "content": question
            }
        ]
    }

    try:
        if stream:
            # Streaming response
            print("\n=== Response (Streaming) ===\n")
            response = requests.post(API_URL, headers=headers, json=payload, stream=True)
            response.raise_for_status()

            full_content = ""
            references = None
            follow_ups = None
            usage = None

            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        data_str = line[6:]  # Remove 'data: ' prefix
                        if data_str == '[DONE]':
                            break

                        try:
                            data = json.loads(data_str)

                            # Handle references (first frame only)
                            if 'references' in data and data['references'] is not None:
                                references = data['references']

                            # Handle content
                            if 'choices' in data and data['choices']:
                                choice = data['choices'][0]
                                if 'delta' in choice and 'content' in choice['delta']:
                                    content = choice['delta']['content']
                                    print(content, end='', flush=True)
                                    full_content += content

                            # Handle follow_ups and usage (last frame)
                            if 'follow_ups' in data and data['follow_ups'] is not None:
                                follow_ups = data['follow_ups']
                            if 'usage' in data and data['usage'] is not None:
                                usage = data['usage']

                        except json.JSONDecodeError:
                            continue

            print("\n")
        else:
            # Non-streaming response
            response = requests.post(API_URL, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()

            # Extract content
            if 'choices' in data and data['choices']:
                message = data['choices'][0].get('message', {})
                full_content = message.get('content', '')
                print(f"\n=== Response ===\n{full_content}\n")

            # Extract metadata
            references = data.get('references')
            follow_ups = data.get('follow_ups')
            usage = data.get('usage')

        # Print references
        if references:
            print("\n=== References ===")
            for i, ref in enumerate(references, 1):
                print(f"\n[{i}] {ref.get('title', 'N/A')}")
                if ref.get('url'):
                    print(f"    URL: {ref['url']}")
                if ref.get('site_name'):
                    print(f"    Source: {ref['site_name']}")
                if ref.get('publish_time'):
                    publish_time = ref['publish_time']
                    if publish_time > 0:
                        dt = datetime.fromtimestamp(publish_time)
                        print(f"    Published: {dt.strftime('%Y-%m-%d %H:%M:%S')}")

        # Print follow-ups
        if follow_ups:
            print("\n=== Suggested Follow-up Questions ===")
            for i, item in enumerate(follow_ups, 1):
                print(f"{i}. {item.get('item', '')}")

        # Print usage
        if usage:
            print("\n=== Token Usage ===")
            print(f"  Prompt tokens: {usage.get('prompt_tokens', 0)}")
            print(f"  Completion tokens: {usage.get('completion_tokens', 0)}")
            print(f"  Total tokens: {usage.get('total_tokens', 0)}")

    except requests.exceptions.RequestException as e:
        print(f"\nError: {e}")
        if hasattr(e, 'response') and e.response is not None:
            try:
                error_data = e.response.json()
                print(f"Error details: {json.dumps(error_data, indent=2, ensure_ascii=False)}")
            except:
                print(f"Response: {e.response.text}")
    except Exception as e:
        print(f"\nError: {e}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python search_web.py <question> [--stream]")
        print("\nEnvironment variables:")
        print("  SEARCH_BOT_ID      (required) - Bot ID from console")
        print("  MODEL_SEARCH_API_KEY or ARK_API_KEY - API key for authentication")
        sys.exit(1)

    question = sys.argv[1]
    stream = '--stream' in sys.argv

    search_web(question, stream)
