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
import urllib
import time
import sys
from volcenginesdkarkruntime import Ark

# Default model
DEFAULT_MODEL = "doubao-seedream-4-5-251128"


def image_generate(prompt: str):
    """Generate image based on prompt.

    Args:
        prompt: The prompt to generate image.
    """
    if not prompt:
        print("Prompt is empty.")
        return

    api_key = os.getenv("MODEL_IMAGE_API_KEY") or os.getenv("ARK_API_KEY")

    client = Ark(api_key=api_key)

    try:
        response = client.images.generate(
            model=os.getenv("MODEL_IMAGE_NAME", DEFAULT_MODEL),
            prompt=prompt,
        )

        download_dir = os.getenv("IMAGE_DOWNLOAD_DIR", os.path.expanduser("./"))
        if not os.path.exists(download_dir):
            try:
                os.makedirs(download_dir, exist_ok=True)
            except Exception as e:
                print(f"Failed to create directory {download_dir}: {e}")
                return

        for i, image in enumerate(response.data):
            #  print(f"Image URL: {image.url}")
            try:
                timestamp = int(time.time())
                filename = f"generated_image_{timestamp}_{i}.png"
                filepath = os.path.join(download_dir, filename)
                urllib.request.urlretrieve(image.url, filepath)
                print(f"Downloaded to: {filepath}")
            except Exception as e:
                print(f"Failed to download image from {image.url}: {e}")
    except Exception as e:
        print(f"Error generating image: {e}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python image_generate.py <prompt>")
        sys.exit(1)
    prompt = sys.argv[1]
    image_generate(prompt)
