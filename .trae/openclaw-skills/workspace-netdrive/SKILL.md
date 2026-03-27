---
name: workspace-netdrive
description: Detects and manages network drives (mounted TOS buckets) mounted in the workspace at /root/.openclaw/workspace/. Use when saving files to persistent storage or checking for mounted buckets.
homepage: https://openclaw.ai/
metadata:
  openclaw:
    emoji: "💾"
    requires: { "bins": ["df", "grep"] }
---

# Workspace Netdrive Skill

This skill helps you detect and utilize network drives (mounted TOS buckets) mounted within the workspace.

## When to Use

Use this skill when the user asks to:
- "Check if any network drive is mounted"
- "Save this file to the netdisk/TOS"
- "Save output to mounted cloud buckets"
- "List available storage buckets"

## Detection Logic

Network drives are mounted as subdirectories under `/root/.openclaw/workspace/`.
The detection script checks for mount points in this directory.

## Commands

### 1. Detect Available Drives

Run this script to list all mounted network drive paths and bucket names:

```bash
scripts/detect_mounts.sh
```

**Example Output:**
```text
Detected network drives (Buckets):
Bucket: my-bucket (Path: /root/.openclaw/workspace/my-bucket)
Bucket: reports-bucket (Path: /root/.openclaw/workspace/reports-bucket)
```

If no drives are found, it will output: `No network drives mounted in workspace.`

> **Important:** If no drives are detected, you **MUST** prompt the user to configure a network drive (configure TOS bucket information) via the **arkClaw interface menu bar** before proceeding.

### 2. Save Files to Drive

Once you have identified a valid mount path (e.g., `/root/.openclaw/workspace/my-bucket`), you can save files directly to it.

**Copy a file:**
```bash
cp source_file.txt /root/.openclaw/workspace/my-bucket/folder/
```

**Write directly to drive:**
```bash
echo "content" > /root/.openclaw/workspace/my-bucket/output.txt
```

### 3. Feedback

When reporting the saved file location to the user, **DO NOT** use the internal file system path (e.g., `/root/.openclaw/workspace/...`).
Instead, report the **Bucket Name** and the **Relative Path within the bucket**.

**Example Feedback:**
> "File saved successfully.
> **Bucket:** my-bucket
> **Path:** folder/source_file.txt"

This format makes it easier for users to access their files externally.
