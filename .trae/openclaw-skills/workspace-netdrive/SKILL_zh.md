---
name: workspace-netdrive
description: 检测并管理挂载在工作区 /root/.openclaw/workspace/ 下的网盘（TOS 存储桶挂载）。当需要将文件保存到持久化存储或检查挂载的存储桶时使用此技能。
homepage: https://openclaw.ai/
metadata:
  openclaw:
    emoji: "💾"
    requires: { "bins": ["df", "grep"] }
---

# 工作区网盘技能 (Workspace Netdrive Skill)

此技能帮助你检测和使用挂载在工作区内的网盘（TOS 存储桶挂载）。

## 何时使用

当用户要求执行以下操作时使用此技能：
- "检查是否有挂载的网盘"
- "把这个文件保存到网盘/TOS"
- "将输出存储到云存储桶挂载"
- "列出可用的存储桶"

## 检测逻辑

网盘通常挂载为 `/root/.openclaw/workspace/` 下的子目录。
检测脚本使用 `df` 和 `grep` 来检查该目录下的挂载点。

## 指令

### 1. 检测可用网盘

运行此脚本以列出所有已挂载的网盘路径和桶名称：

```bash
scripts/detect_mounts.sh
```

**输出示例:**
```text
Detected network drives (Buckets):
Bucket: my-bucket (Path: /root/.openclaw/workspace/my-bucket)
Bucket: reports-bucket (Path: /root/.openclaw/workspace/reports-bucket)
```

如果未发现网盘，输出将是：`No network drives mounted in workspace.`

> **注意：** 如果未检测到网盘，你**必须**提示用户在 **arkClaw 界面菜单栏中**配置网盘（配置TOS存储桶信息），然后才能继续。

### 2. 保存文件到网盘

一旦识别出有效的挂载路径（例如 `/root/.openclaw/workspace/my-bucket`），你可以直接将文件保存到该路径。

**复制文件:**
```bash
cp source_file.txt /root/.openclaw/workspace/my-bucket/folder/
```

**直接写入网盘:**
```bash
echo "内容" > /root/.openclaw/workspace/my-bucket/output.txt
```

### 3. 反馈

向用户反馈文件存储位置时，**请勿**使用内部文件系统路径（例如 `/root/.openclaw/workspace/...`）。
而应该反馈 **桶名称 (Bucket Name)** 和 **相对于桶的路径**。

**反馈示例:**
> "文件已成功保存。
> **桶名:** my-bucket
> **路径:** folder/source_file.txt"

这种格式方便用户在外部系统中访问这些文件。
