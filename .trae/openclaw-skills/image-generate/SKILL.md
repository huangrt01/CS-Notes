---
name: image-generate
description: 使用内置 image_generate.py 脚本生成图片, 准备清晰具体的 `prompt`。
---

# Image Generate

## 适用场景

当需要根据文本描述生成图片时，使用该技能调用 `image_generate` 函数。

## 使用步骤

1. 准备清晰具体的 `prompt`。
2. 运行脚本 `python scripts/image_generate.py "<prompt>"`。运行之前cd到对应的目录。
3. 脚本将返回生成图片的 URL。

## 认证与凭据来源

- 优先读取 `MODEL_IMAGE_API_KEY` 或 `ARK_API_KEY` 环境变量。
- 若未配置，将尝试使用 `VOLCENGINE_ACCESS_KEY` 与 `VOLCENGINE_SECRET_KEY` 获取 Ark API Key。

## 输出格式

- 输出生成的图片 URL。
- 若调用失败，将打印错误信息。

## 示例

```bash
python scripts/image_generate.py "一只可爱的猫"
```
