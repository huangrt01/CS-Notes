# 火山引擎 OpenClaw 部署指南

本指南说明如何在火山引擎 OpenClaw 云服务器上部署我们的 Skills。

## 前置条件

- 已登录到火山引擎 OpenClaw 云服务器的 shell
- 有 Git 仓库的访问权限（SSH 或 Personal Access Token）
- 已配置飞书机器人（可选，用于测试）
- 本地改动已 push 到 Git 仓库

## 推荐方案：直接从 Git 部署（最简单）

### 优化说明
**如果服务器上已经有 CS-Notes 目录**（例如在 `~/CS-Notes`），脚本会自动检测并使用它，无需重新克隆！

在火山引擎服务器 shell 上直接执行：

```bash
# 下载部署脚本
curl -fsSL https://raw.githubusercontent.com/你的用户名/CS-Notes/master/.trae/openclaw-skills/deploy-from-git.sh -o deploy-from-git.sh

# 或者如果仓库还没公开，直接在服务器上从 Git 克隆仓库后执行
cd ~
git clone https://github.com/你的用户名/CS-Notes.git temp-cs-notes
cd temp-cs-notes/.trae/openclaw-skills
chmod +x deploy-from-git.sh

# 运行部署脚本（替换为你的 Git 仓库地址）
bash deploy-from-git.sh https://github.com/你的用户名/CS-Notes.git
```

## 备选方案：本地打包上传

### 步骤 1: 上传 Skills 到服务器

在你的本地 Mac 上执行：

```bash
# 将本地 skills 目录打包
cd /Users/bytedance/CS-Notes/.trae/openclaw-skills
tar -czf cs-notes-skills.tar.gz cs-notes-git-sync cs-notes-todo-sync deploy-on-server.sh

# 使用 scp 上传到服务器（替换 username 和 server-ip）
scp cs-notes-skills.tar.gz deploy-on-server.sh username@server-ip:~/
```

### 步骤 2: 在服务器上解压并安装 Skills

在火山引擎服务器 shell 上执行：

```bash
# 解压
cd ~
tar -xzf cs-notes-skills.tar.gz

# 创建 OpenClaw skills 目录
mkdir -p ~/.openclaw/workspace/skills/

# 复制 skills
cp -r cs-notes-git-sync ~/.openclaw/workspace/skills/
cp -r cs-notes-todo-sync ~/.openclaw/workspace/skills/

# 验证
ls -la ~/.openclaw/workspace/skills/

# 运行部署脚本
chmod +x deploy-on-server.sh
bash deploy-on-server.sh https://github.com/你的用户名/CS-Notes.git
```

## 步骤 3: 克隆 CS-Notes 仓库

在火山引擎服务器上执行：

```bash
# 创建 workspace 目录
mkdir -p ~/.openclaw/workspace/
cd ~/.openclaw/workspace/

# 克隆仓库（替换为你的实际仓库地址）
git clone https://github.com/你的用户名/CS-Notes.git

# 或者使用 SSH
# git clone git@github.com:你的用户名/CS-Notes.git

# 验证
cd CS-Notes
git status
```

## 步骤 4: 配置 Git 凭据

在火山引擎服务器上执行：

```bash
# 配置用户信息
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# 如果使用 HTTPS，配置 credential helper 缓存密码
git config --global credential.helper cache

# 或者使用 Personal Access Token
# git remote set-url origin https://<your-token>@github.com/你的用户名/CS-Notes.git
```

## 步骤 5: 测试 Skills

### 测试 cs-notes-git-sync Skill

```bash
cd ~/.openclaw/workspace/skills/cs-notes-git-sync

# 编辑 main.py，配置正确的 REPO_URL
# 找到 REPO_URL = "https://github.com/username/CS-Notes.git"
# 替换为你的实际仓库地址

# 测试运行
python main.py "这是一条测试任务"
```

### 测试 cs-notes-todo-sync Skill

```bash
cd ~/.openclaw/workspace/skills/cs-notes-todo-sync

# 测试运行
python main.py
```

## 步骤 6: 配置飞书机器人集成

如果还没有配置飞书机器人：

1. 访问 OpenClaw WebChat 页面
2. 在对话框中输入：`帮我接飞书`
3. 按照 AI 指引完成配置

## 验证部署

完成以上步骤后，验证：

1. Skills 目录存在：`~/.openclaw/workspace/skills/`
2. CS-Notes 仓库已克隆：`~/.openclaw/workspace/CS-Notes/`
3. Git 可以正常 push/pull
4. Skills 可以正常运行

## 故障排查

### Git 权限问题

如果遇到 Git 权限问题：
- 检查 SSH key 是否配置：`ls -la ~/.ssh/`
- 生成新的 SSH key：`ssh-keygen -t ed25519 -C "your.email@example.com"`
- 将公钥添加到 GitHub/GitLab

### Skill 找不到仓库

检查 `main.py` 中的 `REPO_PATH` 和 `REPO_URL` 配置是否正确。

