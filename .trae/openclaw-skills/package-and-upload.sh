#!/bin/bash
# 本地打包并上传脚本
# 使用方法：bash package-and-upload.sh <server-username> <server-ip>

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
if [ $# -lt 2 ]; then
    print_error "请提供服务器用户名和 IP"
    echo "使用方法: $0 <server-username> <server-ip>"
    echo "示例: $0 root 123.45.67.89"
    exit 1
fi

SERVER_USER="$1"
SERVER_IP="$2"
SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

print_info "=========================================="
print_info "本地打包并上传脚本"
print_info "=========================================="
print_info "服务器: $SERVER_USER@$SERVER_IP"
print_info "脚本目录: $SCRIPTS_DIR"
echo ""

# 步骤 1: 打包
print_info "步骤 1/3: 打包 Skills"
cd "$SCRIPTS_DIR"
TAR_FILE="cs-notes-skills.tar.gz"
tar -czf "$TAR_FILE" cs-notes-git-sync cs-notes-todo-sync deploy-on-server.sh
print_info "打包完成: $TAR_FILE"
echo ""

# 步骤 2: 上传
print_info "步骤 2/3: 上传到服务器"
scp "$TAR_FILE" deploy-on-server.sh "$SERVER_USER@$SERVER_IP:~/"
print_info "上传完成"
echo ""

# 步骤 3: 服务器端指令
print_info "步骤 3/3: 服务器端操作"
echo ""
print_info "请登录到服务器并执行以下命令："
echo ""
echo "  ssh $SERVER_USER@$SERVER_IP"
echo ""
echo "  # 解压文件"
echo "  cd ~"
echo "  tar -xzf cs-notes-skills.tar.gz"
echo ""
echo "  # 运行部署脚本（替换为你的 Git 仓库地址）"
echo "  chmod +x deploy-on-server.sh"
echo "  bash deploy-on-server.sh https://github.com/你的用户名/CS-Notes.git"
echo ""
print_info "=========================================="
print_info "打包上传完成！"
print_info "=========================================="

