#!/bin/bash
# Todo Web Manager - 模板同步检查脚本
# 检查原始文件是否有更新，提示用户手动处理

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
echo -e "${BLUE}Todo Web Manager - 模板同步检查${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# 定义需要同步的文件
FILES=(
    ".openclaw-memory/MEMORY.md|.trae/web-manager/templates/MEMORY-generic.md"
    ".openclaw-memory/AGENTS.md|.trae/web-manager/templates/AGENTS-generic.md"
    ".trae/rules/project_rules.md|.trae/web-manager/templates/project_rules-generic.md"
)

NEEDS_UPDATE=()

for pair in "${FILES[@]}"; do
    IFS='|' read -r ORIGINAL TEMPLATE <<< "$pair"
    
    ORIGINAL_PATH="$PROJECT_ROOT/$ORIGINAL"
    TEMPLATE_PATH="$PROJECT_ROOT/$TEMPLATE"
    
    echo -e "${BLUE}[检查]${NC} $(basename "$ORIGINAL")"
    
    if [ ! -f "$ORIGINAL_PATH" ]; then
        echo -e "  ${RED}✗ 原始文件不存在: $ORIGINAL_PATH${NC}"
        continue
    fi
    
    if [ ! -f "$TEMPLATE_PATH" ]; then
        echo -e "  ${RED}✗ 模板文件不存在: $TEMPLATE_PATH${NC}"
        continue
    fi
    
    # 获取修改时间
    ORIGINAL_MTIME=$(stat -f "%m" "$ORIGINAL_PATH" 2>/dev/null || stat -c "%Y" "$ORIGINAL_PATH")
    TEMPLATE_MTIME=$(stat -f "%m" "$TEMPLATE_PATH" 2>/dev/null || stat -c "%Y" "$TEMPLATE_PATH")
    
    ORIGINAL_DATE=$(date -r "$ORIGINAL_MTIME" "+%Y年%m月%d日 %H时%M分%S秒")
    TEMPLATE_DATE=$(date -r "$TEMPLATE_MTIME" "+%Y年%m月%d日 %H时%M分%S秒")
    
    if [ "$ORIGINAL_MTIME" -gt "$TEMPLATE_MTIME" ]; then
        echo -e "  ${YELLOW}⚠ 原始文件已更新${NC}"
        echo -e "     原始: $ORIGINAL_DATE"
        echo -e "     模板: $TEMPLATE_DATE"
        NEEDS_UPDATE+=("$ORIGINAL|$TEMPLATE")
    else
        echo -e "  ${GREEN}✓ 模板是最新的${NC}"
    fi
    echo ""
done

if [ ${#NEEDS_UPDATE[@]} -eq 0 ]; then
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}所有模板都是最新的！${NC}"
    echo -e "${GREEN}========================================${NC}"
    exit 0
fi

echo -e "${YELLOW}========================================${NC}"
echo -e "${YELLOW}检测到原始文件有更新！${NC}"
echo -e "${YELLOW}========================================${NC}"
echo ""

echo -e "${BLUE}请手动检查以下模板文件（如果更新是通用的）：${NC}"
for pair in "${NEEDS_UPDATE[@]}"; do
    IFS='|' read -r ORIGINAL TEMPLATE <<< "$pair"
    echo -e "  $YELLOW→$NC $(basename "$TEMPLATE")"
done
echo ""

echo -e "${BLUE}提示：${NC}"
echo -e "  - 如果更新是 CS-Notes 特定的，不需要更新模板"
echo -e "  - 如果更新是通用的，请编辑模板文件移除项目特定内容"
echo -e "  - 模板文件位置: $SCRIPT_DIR/templates/"
echo ""
