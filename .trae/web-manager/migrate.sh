#!/bin/bash
# Todo Web Manager - 一键迁移脚本
# 自动检查同步状态，智能同步模板，然后构建

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

# 定义需要同步的文件
FILES=(
    ".openclaw-memory/MEMORY.md|.trae/web-manager/templates/MEMORY-generic.md"
    ".openclaw-memory/AGENTS.md|.trae/web-manager/templates/AGENTS-generic.md"
    ".trae/rules/project_rules.md|.trae/web-manager/templates/project_rules-generic.md"
)

# 步骤 1：检查模板同步状态并智能同步
echo -e "${BLUE}[步骤 1/4]${NC} 检查模板同步状态并智能同步..."
cd "$SCRIPT_DIR"

NEEDS_UPDATE=()
for pair in "${FILES[@]}"; do
    IFS='|' read -r ORIGINAL TEMPLATE <<< "$pair"
    
    ORIGINAL_PATH="$PROJECT_ROOT/$ORIGINAL"
    TEMPLATE_PATH="$PROJECT_ROOT/$TEMPLATE"
    
    echo -e "${BLUE}[处理]${NC} $(basename "$ORIGINAL")"
    
    if [ ! -f "$ORIGINAL_PATH" ]; then
        echo -e "  ${RED}✗ 原始文件不存在: $ORIGINAL_PATH${NC}"
        continue
    fi
    
    if [ ! -f "$TEMPLATE_PATH" ]; then
        echo -e "  ${RED}✗ 模板文件不存在: $TEMPLATE_PATH${NC}"
        continue
    fi
    
    # 使用 Python 智能同步工具（自动应用变更）
    python3 "$SCRIPT_DIR/auto_sync.py" "$ORIGINAL_PATH" "$TEMPLATE_PATH"
    echo ""
done

echo ""

# 步骤 2：询问是否继续（除非传入 --yes 参数）
echo -e "${BLUE}[步骤 2/4]${NC} 准备构建..."
if [ "$1" == "--yes" ]; then
    echo "自动确认构建..."
else
    echo "自动确认构建..."
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
