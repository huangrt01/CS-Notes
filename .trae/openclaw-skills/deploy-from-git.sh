#!/bin/bash
# 火山引擎 OpenClaw 从 Git 直接部署脚本
# 使用方法：bash deploy-from-git.sh <git-repo-url>

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
TEMP_DIR="$HOME/cs-notes-temp"

print_info "=========================================="
print_info "火山引擎 OpenClaw 从 Git 部署脚本"
print_info "=========================================="
print_info "Git 仓库: $GIT_REPO_URL"
print_info "工作目录: $WORKSPACE_DIR"
echo ""

# 步骤 1: 创建目录
print_info "步骤 1/7: 创建必要的目录"
mkdir -p "$WORKSPACE_DIR"
mkdir -p "$SKILLS_DIR"
mkdir -p "$TEMP_DIR"
print_info "目录创建完成"
echo ""

# 步骤 2: 临时克隆仓库来获取 Skills
print_info "步骤 2/7: 临时克隆仓库获取 Skills"
cd "$TEMP_DIR"
if [ -d "$REPO_NAME" ]; then
    print_warn "临时仓库已存在，删除后重新克隆"
    rm -rf "$REPO_NAME"
fi
git clone --depth 1 "$GIT_REPO_URL" "$REPO_NAME"
print_info "临时克隆完成"
echo ""

# 步骤 3: 复制 Skills
print_info "步骤 3/7: 复制 Skills 到 OpenClaw 目录"
SKILLS_SOURCE="$TEMP_DIR/$REPO_NAME/.trae/openclaw-skills"
if [ -d "$SKILLS_SOURCE" ]; then
    for skill_dir in "$SKILLS_SOURCE"/*/; do
        skill_name=$(basename "$skill_dir")
        if [ "$skill_name" != "." ] && [ "$skill_name" != ".." ]; then
            print_info "安装 Skill: $skill_name"
            cp -r "$skill_dir" "$SKILLS_DIR/"
        fi
    done
    print_info "Skills 复制完成"
else
    print_error "未找到 Skills 目录: $SKILLS_SOURCE"
fi
echo ""

# 步骤 4: 克隆/更新主仓库到 workspace
print_info "步骤 4/7: 克隆/更新主仓库"
cd "$WORKSPACE_DIR"
if [ -d "$REPO_NAME" ]; then
    print_warn "主仓库已存在，尝试拉取最新代码"
    cd "$REPO_NAME"
    git pull || {
        print_error "Git pull 失败"
        exit 1
    }
else
    print_info "克隆主仓库..."
    git clone "$GIT_REPO_URL" "$REPO_NAME" || {
        print_error "Git clone 失败"
        exit 1
    }
    cd "$REPO_NAME"
fi
print_info "主仓库准备完成"
echo ""

# 步骤 5: 更新 cs-notes-git-sync 的 REPO_URL 配置
print_info "步骤 5/7: 更新 Skill 配置"
if [ -f "$SKILLS_DIR/cs-notes-git-sync/main.py" ]; then
    sed -i.bak "s|REPO_URL = \".*\"|REPO_URL = \"$GIT_REPO_URL\"|" "$SKILLS_DIR/cs-notes-git-sync/main.py"
    rm -f "$SKILLS_DIR/cs-notes-git-sync/main.py.bak"
    print_info "已更新 cs-notes-git-sync 的 REPO_URL"
fi
echo ""

# 步骤 6: 配置 Git 用户信息
print_info "步骤 6/7: 配置 Git 用户信息"
read -p "请输入 Git 用户名: " git_user
read -p "请输入 Git 邮箱: " git_email

git config --global user.name "$git_user"
git config --global user.email "$git_email"
git config --global credential.helper cache
print_info "Git 配置完成"
echo ""

# 步骤 7: 验证
print_info "步骤 7/7: 验证部署"
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

# 清理临时目录
print_info "清理临时目录..."
rm -rf "$TEMP_DIR"
print_info "清理完成"
echo ""

# 完成提示
print_info "=========================================="
print_info "部署完成！下一步："
echo ""
echo "1. 测试 cs-notes-git-sync Skill:"
echo "   cd $SKILLS_DIR/cs-notes-git-sync"
echo "   python main.py \"测试任务\""
echo ""
echo "2. 配置飞书机器人（如需要）："
echo "   访问 OpenClaw WebChat，输入 \"帮我接飞书\""
echo ""
print_info "=========================================="

