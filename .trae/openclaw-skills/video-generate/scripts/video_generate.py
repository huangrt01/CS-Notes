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
import time
import urllib.request
import base64
import mimetypes
from volcenginesdkarkruntime import Ark

# Try to import constants, with fallback

DEFAULT_VIDEO_MODEL_NAME = "doubao-seedance-1-5-pro-251215"


def get_image_content(image_input: str) -> str:
    """
    Process image input. If it's a local file, convert to base64 data URI.
    Otherwise, assume it's a URL and return as is.
    """
    if os.path.isfile(image_input):
        try:
            mime_type, _ = mimetypes.guess_type(image_input)
            if not mime_type:
                # Fallback or default
                mime_type = "image/png"

            with open(image_input, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
                return f"data:{mime_type};base64,{encoded_string}"
        except Exception as e:
            print(f"Failed to read or encode image file {image_input}: {e}")
            return None
    return image_input


def video_generate(filename: str, prompt: str, first_frame_image: str = None):
    """Generate video based on prompt.

    Args:
        filename: The filename to save the video as.
        prompt: The prompt to generate video.
        first_frame_image: Optional URL or local file path of the first frame image.
    """
    if not prompt:
        print("Prompt is empty.")
        return
    if not filename:
        print("Filename is empty.")
        return

    api_key = os.getenv("MODEL_VIDEO_API_KEY") or os.getenv("ARK_API_KEY")

    client = Ark(api_key=api_key)
    model_name = os.getenv("MODEL_VIDEO_NAME", DEFAULT_VIDEO_MODEL_NAME)

    print(f"Starting video generation with model: {model_name}")

    try:
        # Create task
        content = [{"type": "text", "text": prompt}]
        if first_frame_image:
            image_url_or_base64 = get_image_content(first_frame_image)
            if image_url_or_base64:
                content.append(
                    {"type": "image_url", "image_url": {"url": image_url_or_base64}}
                )
                log_msg = (
                    "base64 image"
                    if image_url_or_base64.startswith("data:")
                    else image_url_or_base64
                )
                print(f"Using first frame image: {log_msg[:50]}...")
            else:
                print(f"Could not process first frame image: {first_frame_image}")

        response = client.content_generation.tasks.create(
            model=model_name,
            content=content,
        )
        task_id = response.id
        print(f"Task created: {task_id}")

        # Poll status
        print("Polling for task completion...")
        while True:
            result = client.content_generation.tasks.get(task_id=task_id)
            status = result.status

            if status == "succeeded":
                video_url = result.content.video_url
                print(f"Video URL: {video_url}")

                # Download
                try:
                    # Ensure filename has extension if needed, though user might have provided it
                    # If user provided "myvideo", we might want "myvideo.mp4"
                    # But let's stick to what user provided exactly first

                    # Create directory if needed (based on filename path)
                    dirname = os.path.dirname(filename)
                    if dirname and not os.path.exists(dirname):
                        os.makedirs(dirname, exist_ok=True)

                    print(f"Downloading video to {filename}...")
                    urllib.request.urlretrieve(video_url, filename)
                    print(f"Downloaded to: {filename}")
                except Exception as e:
                    print(f"Failed to download video from {video_url}: {e}")

                break
            elif status == "failed":
                print(f"Video generation failed. Error: {result.error}")
                break
            elif status == "cancelled":
                print("Video generation cancelled.")
                break
            else:
                # running, queued, etc.
                print(f"Status: {status}. Waiting...")
                time.sleep(5)

    except Exception as e:
        print(f"Error generating video: {e}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(
            "Usage: python video_generate.py <filename> <prompt> [first_frame_image_path_or_url]"
        )
        sys.exit(1)

    filename = sys.argv[1]
    prompt = sys.argv[2]
    first_frame_image = sys.argv[3] if len(sys.argv) > 3 else None

    video_generate(filename, prompt, first_frame_image)
