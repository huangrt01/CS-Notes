#!/bin/bash

# 同步模板脚本 - 将 CS-Notes 中的通用更新同步到模板文件
# 使用方法：./sync-templates.sh

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 项目根目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WEB_MANAGER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATES_DIR="$WEB_MANAGER_DIR/templates"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}同步通用化模板${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# 检查是否需要同步
echo -e "${YELLOW}检查文件变更...${NC}"
echo ""

# MEMORY.md 同步
echo "📄 MEMORY.md:"
if [ -f "$PROJECT_ROOT/.openclaw-memory/MEMORY.md" ] && [ -f "$TEMPLATES_DIR/MEMORY-generic.md" ]; then
    # 比较最后修改时间
    ORIGINAL_MTIME=$(stat -f "%m" "$PROJECT_ROOT/.openclaw-memory/MEMORY.md")
    TEMPLATE_MTIME=$(stat -f "%m" "$TEMPLATES_DIR/MEMORY-generic.md" 2>/dev/null || echo 0)
    
    if [ "$ORIGINAL_MTIME" -gt "$TEMPLATE_MTIME" ]; then
        echo -e "  ${YELLOW}原始文件已更新，建议重新生成模板${NC}"
        echo -e "  原始: $(date -r "$ORIGINAL_MTIME")"
        echo -e "  模板: $(date -r "$TEMPLATE_MTIME" 2>/dev/null || echo "未创建")"
    else
        echo -e "  ${GREEN}模板是最新的${NC}"
    fi
fi
echo ""

# AGENTS.md 同步
echo "📄 AGENTS.md:"
if [ -f "$PROJECT_ROOT/.openclaw-memory/AGENTS.md" ] && [ -f "$TEMPLATES_DIR/AGENTS-generic.md" ]; then
    ORIGINAL_MTIME=$(stat -f "%m" "$PROJECT_ROOT/.openclaw-memory/AGENTS.md")
    TEMPLATE_MTIME=$(stat -f "%m" "$TEMPLATES_DIR/AGENTS-generic.md" 2>/dev/null || echo 0)
    
    if [ "$ORIGINAL_MTIME" -gt "$TEMPLATE_MTIME" ]; then
        echo -e "  ${YELLOW}原始文件已更新，建议重新生成模板${NC}"
        echo -e "  原始: $(date -r "$ORIGINAL_MTIME")"
        echo -e "  模板: $(date -r "$TEMPLATE_MTIME" 2>/dev/null || echo "未创建")"
    else
        echo -e "  ${GREEN}模板是最新的${NC}"
    fi
fi
echo ""

# project_rules.md 同步
echo "📄 project_rules.md:"
if [ -f "$PROJECT_ROOT/.trae/rules/project_rules.md" ] && [ -f "$TEMPLATES_DIR/project_rules-generic.md" ]; then
    ORIGINAL_MTIME=$(stat -f "%m" "$PROJECT_ROOT/.trae/rules/project_rules.md")
    TEMPLATE_MTIME=$(stat -f "%m" "$TEMPLATES_DIR/project_rules-generic.md" 2>/dev/null || echo 0)
    
    if [ "$ORIGINAL_MTIME" -gt "$TEMPLATE_MTIME" ]; then
        echo -e "  ${YELLOW}原始文件已更新，建议重新生成模板${NC}"
        echo -e "  原始: $(date -r "$ORIGINAL_MTIME")"
        echo -e "  模板: $(date -r "$TEMPLATE_MTIME" 2>/dev/null || echo "未创建")"
    else
        echo -e "  ${GREEN}模板是最新的${NC}"
    fi
fi
echo ""

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}使用说明${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "当你在 CS-Notes 中更新了记忆/规则文件后："
echo ""
echo "1. 如果更新是**通用的**（适用于所有项目）："
echo "   - 手动更新对应的模板文件（MEMORY-generic.md 等）"
echo "   - 然后运行 build.sh 重新打包"
echo ""
echo "2. 如果更新是**CS-Notes 特定的**："
echo "   - 不需要更新模板"
echo "   - 模板保持原样即可"
echo ""
echo "3. 分层策略："
echo "   - 原始文件：CS-Notes 项目专用"
echo "   - 模板文件：通用化版本，用于迁移"
echo ""
