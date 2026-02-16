#!/bin/bash
# 火山引擎 OpenClaw 服务器一键部署脚本
# 使用方法：bash deploy-on-server.sh <git-repo-url>

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查参数
if [ $# -lt 1 ]; then
    print_error "请提供 Git 仓库 URL"
    echo "使用方法: $0 <git-repo-url>"
    echo "示例: $0 https://github.com/你的用户名/CS-Notes.git"
    exit 1
fi

GIT_REPO_URL="$1"
REPO_NAME="CS-Notes"
WORKSPACE_DIR="$HOME/.openclaw/workspace"
SKILLS_DIR="$WORKSPACE_DIR/skills"

print_info "=========================================="
print_info "火山引擎 OpenClaw 部署脚本"
print_info "=========================================="
print_info "Git 仓库: $GIT_REPO_URL"
print_info "工作目录: $WORKSPACE_DIR"
echo ""

# 步骤 1: 创建目录
print_info "步骤 1/6: 创建必要的目录"
mkdir -p "$WORKSPACE_DIR"
mkdir -p "$SKILLS_DIR"
print_info "目录创建完成"
echo ""

# 步骤 2: 检查并克隆仓库
print_info "步骤 2/6: 克隆/更新 CS-Notes 仓库"
cd "$WORKSPACE_DIR"
if [ -d "$REPO_NAME" ]; then
    print_warn "仓库已存在，尝试拉取最新代码"
    cd "$REPO_NAME"
    git pull || {
        print_error "Git pull 失败"
        exit 1
    }
else
    print_info "克隆仓库..."
    git clone "$GIT_REPO_URL" "$REPO_NAME" || {
        print_error "Git clone 失败"
        exit 1
    }
    cd "$REPO_NAME"
fi
print_info "仓库准备完成"
echo ""

# 步骤 3: 检查并安装 Skills
print_info "步骤 3/6: 检查 Skills"
if [ -d "$HOME/cs-notes-git-sync" ] && [ -d "$HOME/cs-notes-todo-sync" ]; then
    print_info "发现 Skills 目录，正在安装..."
    cp -r "$HOME/cs-notes-git-sync" "$SKILLS_DIR/"
    cp -r "$HOME/cs-notes-todo-sync" "$SKILLS_DIR/"
    
    # 更新 REPO_URL 配置
    print_info "更新 Skill 配置..."
    if [ -f "$SKILLS_DIR/cs-notes-git-sync/main.py" ]; then
        sed -i.bak "s|REPO_URL = \".*\"|REPO_URL = \"$GIT_REPO_URL\"|" "$SKILLS_DIR/cs-notes-git-sync/main.py"
        rm -f "$SKILLS_DIR/cs-notes-git-sync/main.py.bak"
        print_info "已更新 cs-notes-git-sync 的 REPO_URL"
    fi
    
    print_info "Skills 安装完成"
else
    print_warn "未找到 Skills 目录，请先上传 cs-notes-skills.tar.gz 并解压"
fi
echo ""

# 步骤 4: 配置 Git 用户信息
print_info "步骤 4/6: 配置 Git 用户信息"
read -p "请输入 Git 用户名: " git_user
read -p "请输入 Git 邮箱: " git_email

git config --global user.name "$git_user"
git config --global user.email "$git_email"
git config --global credential.helper cache
print_info "Git 配置完成"
echo ""

# 步骤 5: 验证
print_info "步骤 5/6: 验证部署"
echo ""
print_info "检查目录结构:"
ls -la "$WORKSPACE_DIR/"
echo ""
print_info "检查 Skills:"
ls -la "$SKILLS_DIR/" 2>/dev/null || print_warn "Skills 目录为空"
echo ""
print_info "检查 Git 仓库:"
cd "$WORKSPACE_DIR/$REPO_NAME"
git status
echo ""

# 步骤 6: 测试提示
print_info "步骤 6/6: 下一步操作"
echo ""
print_info "部署完成！下一步："
echo "1. 如果已上传 Skills，测试运行："
echo "   cd $SKILLS_DIR/cs-notes-git-sync"
echo "   python main.py \"测试任务\""
echo ""
echo "2. 配置飞书机器人（如需要）："
echo "   访问 OpenClaw WebChat，输入 \"帮我接飞书\""
echo ""
print_info "=========================================="
print_info "部署完成！"
print_info "=========================================="

