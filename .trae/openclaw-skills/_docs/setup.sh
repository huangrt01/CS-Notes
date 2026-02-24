#!/bin/bash
# OpenClaw 统一设置脚本
# 在容器/服务器内直接运行此脚本

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_info "=========================================="
print_info "OpenClaw 统一设置脚本"
print_info "=========================================="
echo ""

# 步骤 1: 配置 Git
print_info "步骤 1/6: 配置 Git"
read -p "请输入 Git 用户名: " git_user
read -p "请输入 Git 邮箱: " git_email

git config --global user.name "$git_user"
git config --global user.email "$git_email"
git config --global credential.helper store
print_info "Git 配置完成"
echo ""

# 步骤 2: 检查仓库状态（优先使用本地已存在的仓库）
print_info "步骤 2/7: 检查仓库状态"
REPO_NAME="CS-Notes"
WORKSPACE_DIR="$HOME/.openclaw/workspace"
WORKSPACE_REPO_DIR="$WORKSPACE_DIR/$REPO_NAME"

# 检查常见的本地仓库位置
COMMON_PATHS=(
    "$HOME/$REPO_NAME"
    "$HOME/Desktop/$REPO_NAME"
    "$HOME/Documents/$REPO_NAME"
    "$PWD"
)

FOUND_REPO=""
for path in "${COMMON_PATHS[@]}"; do
    if [ -d "$path/.git" ]; then
        FOUND_REPO="$path"
        break
    fi
done

if [ -n "$FOUND_REPO" ]; then
    print_info "发现本地仓库: $FOUND_REPO"
    REPO_DIR="$FOUND_REPO"
else
    print_warn "未找到本地仓库"
    read -p "请输入 CS-Notes 仓库的完整路径（例如 /root/CS-Notes）: " REPO_DIR
    if [ ! -d "$REPO_DIR/.git" ]; then
        print_error "目录不是有效的 Git 仓库: $REPO_DIR"
        exit 1
    fi
fi

print_info "使用仓库目录: $REPO_DIR"
echo ""

# 步骤 3: 将仓库复制/移动到 OpenClaw 工作区
print_info "步骤 3/7: 准备工作区仓库"
mkdir -p "$WORKSPACE_DIR"

if [ -d "$WORKSPACE_REPO_DIR/.git" ]; then
    print_warn "工作区仓库已存在，尝试拉取最新代码"
    cd "$WORKSPACE_REPO_DIR"
    git pull || print_warn "Git pull 失败，请手动处理"
else
    print_info "将本地仓库复制到工作区..."
    cp -r "$REPO_DIR" "$WORKSPACE_DIR/"
    print_info "复制完成"
fi

cd "$WORKSPACE_REPO_DIR"
print_info "工作区仓库: $WORKSPACE_REPO_DIR"
echo ""

# 步骤 4: 复制 Skills
print_info "步骤 4/7: 安装 Skills"
SKILLS_SOURCE="$REPO_DIR/.trae/openclaw-skills"
OPENCLAW_SKILLS_DIR="$HOME/.openclaw/workspace/skills"

mkdir -p "$OPENCLAW_SKILLS_DIR"

if [ -d "$SKILLS_SOURCE" ]; then
    for skill_dir in "$SKILLS_SOURCE"/*/; do
        skill_name=$(basename "$skill_dir")
        if [ "$skill_name" != "." ] && [ "$skill_name" != ".." ] && [ -d "$skill_dir" ]; then
            print_info "安装 Skill: $skill_name"
            cp -r "$skill_dir" "$OPENCLAW_SKILLS_DIR/"
        fi
    done
    print_info "Skills 安装完成"
else
    print_error "未找到 Skills 目录: $SKILLS_SOURCE"
fi
echo ""

# 步骤 5: 配置 cs-notes-git-sync 的 REPO_URL
print_info "步骤 5/7: 配置 Skill 参数"
REPO_URL=$(git remote get-url origin 2>/dev/null || echo "")
if [ -z "$REPO_URL" ]; then
    read -p "请输入 Git 仓库 URL: " REPO_URL
fi

if [ -f "$OPENCLAW_SKILLS_DIR/cs-notes-git-sync/main.py" ]; then
    sed -i.bak "s|REPO_URL = \".*\"|REPO_URL = \"$REPO_URL\"|" "$OPENCLAW_SKILLS_DIR/cs-notes-git-sync/main.py" 2>/dev/null || true
    rm -f "$OPENCLAW_SKILLS_DIR/cs-notes-git-sync/main.py.bak" 2>/dev/null || true
    print_info "已更新 cs-notes-git-sync 的 REPO_URL: $REPO_URL"
fi
echo ""

# 步骤 6: 配置 Git 推送权限
print_info "步骤 6/7: 配置 Git 推送权限"
if [[ "$REPO_URL" == https://* ]]; then
    print_warn "检测到使用 HTTPS 协议"
    print_info "为了能 git push，你需要："
    echo "1. 生成 Personal Access Token (PAT)"
    echo "2. 使用以下格式设置远程 URL:"
    echo "   git remote set-url origin https://<your-token>@github.com/huangrt01/CS-Notes.git"
    echo ""
    read -p "是否现在配置？(y/n): " setup_now
    if [ "$setup_now" = "y" ] || [ "$setup_now" = "Y" ]; then
        read -p "请输入你的 Personal Access Token: " git_token
        # 从 URL 中提取路径部分
        if [[ "$REPO_URL" =~ ^https://github.com/(.*)\.git$ ]]; then
            repo_path="${BASH_REMATCH[1]}"
            new_url="https://$git_token@github.com/$repo_path.git"
            git remote set-url origin "$new_url"
            print_info "已更新远程 URL"
        else
            print_warn "无法解析仓库 URL，请手动设置"
        fi
    fi
else
    print_info "检测到使用 SSH 协议"
    print_info "请确保你的 SSH key 已添加到 GitHub"
    echo ""
    if [ ! -f "$HOME/.ssh/id_ed25519" ] && [ ! -f "$HOME/.ssh/id_rsa" ]; then
        print_warn "未找到 SSH key"
        read -p "是否生成新的 SSH key？(y/n): " gen_ssh
        if [ "$gen_ssh" = "y" ] || [ "$gen_ssh" = "Y" ]; then
            ssh-keygen -t ed25519 -C "$git_email" -f "$HOME/.ssh/id_ed25519" -N ""
            print_info "SSH key 已生成"
            echo ""
            print_info "请将以下公钥添加到 GitHub:"
            cat "$HOME/.ssh/id_ed25519.pub"
        fi
    fi
fi
echo ""

# 步骤 7: 验证
print_info "步骤 7/7: 验证"
echo ""
print_info "检查 Skills:"
ls -la "$OPENCLAW_SKILLS_DIR/" 2>/dev/null || print_warn "Skills 目录为空"
echo ""
print_info "检查 Git 仓库:"
git status
echo ""
print_info "尝试测试推送权限..."
read -p "是否进行测试推送？(y/n, 会创建一个测试提交): " test_push
if [ "$test_push" = "y" ] || [ "$test_push" = "Y" ]; then
    touch .test-push-$(date +%s)
    git add .test-push-* 2>/dev/null || true
    git commit -m "test: 验证推送权限" 2>/dev/null || true
    if git push 2>&1; then
        print_info "测试推送成功！"
        git reset --mixed HEAD~1 2>/dev/null || true
        rm -f .test-push-* 2>/dev/null || true
    else
        print_error "测试推送失败，请检查权限配置"
    fi
fi
echo ""

print_info "=========================================="
print_info "设置完成！"
print_info "=========================================="
echo ""
print_info "下一步："
echo "1. 测试 cs-notes-git-sync Skill:"
echo "   cd $OPENCLAW_SKILLS_DIR/cs-notes-git-sync"
echo "   python main.py \"测试任务\""
echo ""
echo "2. 访问 OpenClaw WebChat，开始与 bot 对话"
echo ""

