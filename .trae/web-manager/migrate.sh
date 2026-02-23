#!/bin/bash
# Todo Web Manager - 一键迁移脚本
# 自动检查同步状态，然后构建

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Todo Web Manager - 一键迁移${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# 步骤 1：检查模板同步状态
echo -e "${BLUE}[步骤 1/4]${NC} 检查模板同步状态..."
cd "$SCRIPT_DIR"
./sync-templates.sh
echo ""

# 步骤 2：询问是否继续
echo -e "${BLUE}[步骤 2/4]${NC} 准备构建..."
read -p "是否继续构建可迁移包？(y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo -e "${YELLOW}已取消构建。${NC}"
    exit 0
fi
echo ""

# 步骤 3：运行构建脚本
echo -e "${BLUE}[步骤 3/4]${NC} 运行构建脚本..."
cd "$SCRIPT_DIR"
./build.sh
echo ""

# 步骤 4：完成
echo -e "${BLUE}[步骤 4/4]${NC} 迁移完成！"
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}迁移成功！${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "下一步："
echo "  1. 查看构建输出目录"
echo "  2. 将压缩包复制到新项目"
echo "  3. 根据需要定制项目规则"
echo ""
