---
name: video-generate
description: 使用 video_generate.py 脚本生成视频，需要提供文件名和 prompt，可选提供首帧图片（URL或本地路径）。
---

# Video Generate

## 适用场景

当需要根据文本描述生成视频时，使用该技能。支持通过首帧图片控制视频起始画面，首帧可以是 URL 或本地文件路径。

## 使用步骤

1. 准备目标文件名（如 `output.mp4`）和清晰具体的 `prompt`。
2. (可选) 准备首帧图片，可以是 HTTP URL，也可以是本地文件路径（脚本会自动转为 Base64）。
3. 运行脚本 `python scripts/video_generate.py <filename> "<prompt>" [first_frame]`。运行之前cd到对应的目录。
4. 脚本将输出视频的 TOS URL 并自动下载到指定文件。

## 认证与凭据来源

- 优先读取 `MODEL_VIDEO_API_KEY` 或 `ARK_API_KEY` 环境变量。
- 若未配置，将尝试使用 `VOLCENGINE_ACCESS_KEY` 与 `VOLCENGINE_SECRET_KEY` 获取 Ark API Key。

## 输出格式

- 控制台输出生成的视频 URL。
- 视频文件将被下载到指定路径。

## 示例

**纯文本生成：**

```bash
python scripts/video_generate.py "cat.mp4" "一只可爱的猫"
```

**带首帧图片生成（URL）：**

```bash
python scripts/video_generate.py "dog_run.mp4" "一只小狗在草地上奔跑" "https://example.com/dog_start.png"
```

**带首帧图片生成（本地文件）：**

```bash
python scripts/video_generate.py "my_video.mp4" "图片中的人物动起来" "/path/to/local/image.jpg"
```
