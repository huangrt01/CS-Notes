#!/bin/bash

# Todo Web Manager - 一键迁移脚本
# 使用方法：./migrate.sh
#
# 工作流：
# 1. 检查原始文件和模板的同步状态
# 2. 如果有更新，提示用户手动编辑模板
# 3. 自动运行 build.sh 打包

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 项目根目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WEB_MANAGER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATES_DIR="$WEB_MANAGER_DIR/templates"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Todo Web Manager - 一键迁移${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# 步骤 1: 检查同步状态
echo -e "${BLUE}[步骤 1/4] 检查模板同步状态...${NC}"
echo ""

NEEDS_SYNC=false

check_file() {
    local original_file="$1"
    local template_file="$2"
    local name="$3"
    
    if [ -f "$original_file" ] && [ -f "$template_file" ]; then
        ORIGINAL_MTIME=$(stat -f "%m" "$original_file" 2>/dev/null || echo 0)
        TEMPLATE_MTIME=$(stat -f "%m" "$template_file" 2>/dev/null || echo 0)
        
        if [ "$ORIGINAL_MTIME" -gt "$TEMPLATE_MTIME" ]; then
            echo -e "  ${YELLOW}⚠️  ${name}: 原始文件已更新${NC}"
            echo -e "     原始: $(date -r "$ORIGINAL_MTIME")"
            echo -e "     模板: $(date -r "$TEMPLATE_MTIME" 2>/dev/null || echo "未创建")"
            NEEDS_SYNC=true
        else
            echo -e "  ${GREEN}✓ ${name}: 模板是最新的${NC}"
        fi
    fi
}

check_file "$PROJECT_ROOT/.openclaw-memory/MEMORY.md" "$TEMPLATES_DIR/MEMORY-generic.md" "MEMORY.md"
check_file "$PROJECT_ROOT/.openclaw-memory/AGENTS.md" "$TEMPLATES_DIR/AGENTS-generic.md" "AGENTS.md"
check_file "$PROJECT_ROOT/.trae/rules/project_rules.md" "$TEMPLATES_DIR/project_rules-generic.md" "project_rules.md"

echo ""

# 步骤 2: 如果需要同步，提示用户编辑
if [ "$NEEDS_SYNC" = true ]; then
    echo -e "${YELLOW}检测到原始文件有更新！${NC}"
    echo ""
    echo "请手动更新以下模板文件（如果更新是通用的）："
    echo ""
    echo "  1. $TEMPLATES_DIR/MEMORY-generic.md"
    echo "  2. $TEMPLATES_DIR/AGENTS-generic.md"
    echo "  3. $TEMPLATES_DIR/project_rules-generic.md"
    echo ""
    echo "提示："
    echo "  - 如果更新是 CS-Notes 特定的，不需要更新模板"
    echo "  - 如果更新是通用的，请编辑模板文件移除项目特定内容"
    echo ""
    
    read -p "是否现在打开模板文件进行编辑？(y/n): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "正在打开模板文件..."
        if command -v open &> /dev/null; then
            open "$TEMPLATES_DIR"
        elif command -v xdg-open &> /dev/null; then
            xdg-open "$TEMPLATES_DIR"
        else
            echo "请手动打开目录: $TEMPLATES_DIR"
        fi
        echo ""
        read -p "编辑完成后按 Enter 继续..."
    fi
else
    echo -e "${GREEN}✓ 所有模板都是最新的！${NC}"
fi
echo ""

# 步骤 3: 确认是否继续构建
echo -e "${BLUE}[步骤 2/4] 准备构建...${NC}"
echo ""
read -p "是否继续构建可迁移包？(y/n): " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}已取消构建${NC}"
    exit 0
fi
echo ""

# 步骤 4: 运行 build.sh
echo -e "${BLUE}[步骤 3/4] 运行构建脚本...${NC}"
echo ""
cd "$WEB_MANAGER_DIR"
./build.sh
echo ""

# 完成
echo -e "${BLUE}[步骤 4/4] 迁移完成！${NC}"
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
