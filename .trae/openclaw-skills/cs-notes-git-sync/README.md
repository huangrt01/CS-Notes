# CS-Notes Git Sync Skill

OpenClaw Skill，用于接收 Lark 消息并同步到 CS-Notes Git 仓库。

## 功能

- 接收 Lark 消息并解析为 todo 格式
- 克隆/拉取 CS-Notes 仓库
- 写入 INBOX.md 并自动 commit & push

## 安装

1. 将此目录复制到 `~/.openclaw/workspace/skills/cs-notes-git-sync/`
2. 修改 `main.py` 中的 `REPO_URL` 为实际的仓库地址
3. 确保已配置 Git 凭据（可以使用 SSH 或 Personal Access Token）

## 使用

```bash
python main.py "这是一条测试任务"
```

## 目录结构

```
cs-notes-git-sync/
├── skill.json    # Skill 配置文件
├── main.py       # 主程序
└── README.md     # 说明文档
```

