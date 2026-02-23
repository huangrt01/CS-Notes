#!/bin/bash

# Todo Web Manager 解耦方案 - 自动化构建脚本
# 用于打包可迁移的 Todo Web Manager
#
# 使用方法：
#   ./build.sh [输出目录]
#
# 默认输出到：./todos-web-manager-package/

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 项目根目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WEB_MANAGER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 默认输出目录
OUTPUT_DIR="${1:-$WEB_MANAGER_DIR/todos-web-manager-package}"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Todo Web Manager 解耦构建脚本${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "项目根目录: $PROJECT_ROOT"
echo "Web Manager 目录: $WEB_MANAGER_DIR"
echo "输出目录: $OUTPUT_DIR"
echo ""

# 清理并创建输出目录
echo -e "${YELLOW}[1/8] 清理并创建输出目录...${NC}"
rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"
echo -e "${GREEN}✓ 输出目录已创建${NC}"
echo ""

# ============ 步骤 1: 复制 .trae/web-manager/ ============
echo -e "${YELLOW}[2/8] 复制 Web Manager 核心文件...${NC}"
mkdir -p "$OUTPUT_DIR/.trae/web-manager"
cp "$WEB_MANAGER_DIR/server.py" "$OUTPUT_DIR/.trae/web-manager/"
cp "$WEB_MANAGER_DIR/index-enhanced.html" "$OUTPUT_DIR/.trae/web-manager/"
cp "$WEB_MANAGER_DIR/config.json" "$OUTPUT_DIR/.trae/web-manager/"
cp "$WEB_MANAGER_DIR/requirements.txt" "$OUTPUT_DIR/.trae/web-manager/" 2>/dev/null || true
cp "$WEB_MANAGER_DIR/build.sh" "$OUTPUT_DIR/.trae/web-manager/" 2>/dev/null || true
echo -e "${GREEN}✓ Web Manager 核心文件已复制${NC}"
echo ""

# ============ 步骤 2: 复制 .trae/todos/ ============
echo -e "${YELLOW}[3/8] 复制 Todos 数据目录结构...${NC}"
mkdir -p "$OUTPUT_DIR/.trae/todos/archive"
# 不复制实际的 todos.json 和归档数据，只创建空目录结构
touch "$OUTPUT_DIR/.trae/todos/.gitkeep"
touch "$OUTPUT_DIR/.trae/todos/archive/.gitkeep"
echo -e "${GREEN}✓ Todos 数据目录结构已创建${NC}"
echo ""

# ============ 步骤 3: 复制 .trae/documents/ ============
echo -e "${YELLOW}[4/8] 复制 Documents 目录结构...${NC}"
mkdir -p "$OUTPUT_DIR/.trae/documents"
# 创建 INBOX.md 模板
cat > "$OUTPUT_DIR/.trae/documents/INBOX.md" << 'EOF'
# INBOX - 收件箱

这是新项目的收件箱，用于捕获新想法和待办事项。

EOF
echo -e "${GREEN}✓ Documents 目录结构已创建${NC}"
echo ""

# ============ 步骤 4: 复制 .trae/rules/ ============
echo -e "${YELLOW}[5/8] 复制 Rules 目录...${NC}"
mkdir -p "$OUTPUT_DIR/.trae/rules/core"
mkdir -p "$OUTPUT_DIR/.trae/rules/templates"
mkdir -p "$OUTPUT_DIR/.trae/rules/project"

# 使用通用化模板
if [ -f "$WEB_MANAGER_DIR/templates/project_rules-generic.md" ]; then
    cp "$WEB_MANAGER_DIR/templates/project_rules-generic.md" "$OUTPUT_DIR/.trae/rules/project/project_rules.md"
    echo -e "${GREEN}✓ 使用通用化项目规则模板${NC}"
else
    # 创建简单的项目规则模板
    cat > "$OUTPUT_DIR/.trae/rules/project/project_rules.md" << 'EOF'
# 项目规则

这是新项目的规则，请根据项目需求进行定制。

EOF
fi
echo -e "${GREEN}✓ Rules 目录结构已创建${NC}"
echo ""

# ============ 步骤 5: 复制 .openclaw-memory/ ============
echo -e "${YELLOW}[6/8] 复制 OpenClaw 记忆体系...${NC}"
mkdir -p "$OUTPUT_DIR/.openclaw-memory/memory"

# 优先使用通用化模板
if [ -f "$WEB_MANAGER_DIR/templates/MEMORY-generic.md" ]; then
    cp "$WEB_MANAGER_DIR/templates/MEMORY-generic.md" "$OUTPUT_DIR/.openclaw-memory/MEMORY.md"
    echo -e "${GREEN}✓ 使用通用化 MEMORY.md 模板${NC}"
elif [ -f "$PROJECT_ROOT/.openclaw-memory/MEMORY.md" ]; then
    cp "$PROJECT_ROOT/.openclaw-memory/MEMORY.md" "$OUTPUT_DIR/.openclaw-memory/"
    echo -e "${YELLOW}⚠  使用原始 MEMORY.md，建议根据新项目定制${NC}"
fi

if [ -f "$WEB_MANAGER_DIR/templates/AGENTS-generic.md" ]; then
    cp "$WEB_MANAGER_DIR/templates/AGENTS-generic.md" "$OUTPUT_DIR/.openclaw-memory/AGENTS.md"
    echo -e "${GREEN}✓ 使用通用化 AGENTS.md 模板${NC}"
elif [ -f "$PROJECT_ROOT/.openclaw-memory/AGENTS.md" ]; then
    cp "$PROJECT_ROOT/.openclaw-memory/AGENTS.md" "$OUTPUT_DIR/.openclaw-memory/"
    echo -e "${YELLOW}⚠  使用原始 AGENTS.md，建议根据新项目定制${NC}"
fi

# SOUL.md 是通用的，可以直接复制
if [ -f "$PROJECT_ROOT/.openclaw-memory/SOUL.md" ]; then
    cp "$PROJECT_ROOT/.openclaw-memory/SOUL.md" "$OUTPUT_DIR/.openclaw-memory/"
fi
if [ -f "$PROJECT_ROOT/.openclaw-memory/HEARTBEAT.md" ]; then
    cp "$PROJECT_ROOT/.openclaw-memory/HEARTBEAT.md" "$OUTPUT_DIR/.openclaw-memory/"
fi
if [ -f "$PROJECT_ROOT/.openclaw-memory/TOOLS.md" ]; then
    cp "$PROJECT_ROOT/.openclaw-memory/TOOLS.md" "$OUTPUT_DIR/.openclaw-memory/"
fi
if [ -f "$PROJECT_ROOT/.openclaw-memory/IDENTITY.md" ]; then
    cp "$PROJECT_ROOT/.openclaw-memory/IDENTITY.md" "$OUTPUT_DIR/.openclaw-memory/"
fi
    
# 创建 USER.md 模板
cat > "$OUTPUT_DIR/.openclaw-memory/USER.md" << 'EOF'
# USER.md - About Your Human

*Learn about the person you're helping. Update this as you go.*

- **Name:** [Your Name]
- **What to call them:** [Your Name]
- **Pronouns:** 
- **Timezone:** [Your Timezone]
- **Notes:** 

## Context

*(What do they care about? What projects are they working on? What annoys them? What makes them laugh? Build this over time.)*

---

The more you know, the better you can help. But remember — you're learning about a person, not building a dossier. Respect the difference.

EOF
    
# 创建记忆目录占位符
touch "$OUTPUT_DIR/.openclaw-memory/memory/.gitkeep"
echo -e "${GREEN}✓ OpenClaw 记忆体系已复制${NC}"
echo ""

# ============ 步骤 6: 创建 README ============
echo -e "${YELLOW}[7/8] 创建 README 文档...${NC}"
cat > "$OUTPUT_DIR/README.md" << 'EOF'
# Todo Web Manager - 可迁移包

这是 Todo Web Manager 的可迁移包，可以轻松部署到任意新项目中。

## 快速开始

### 1. 解压包

将此包的内容复制到你的新项目根目录：

```bash
cp -r .trae/ /path/to/your/new/project/
cp -r .openclaw-memory/ /path/to/your/new/project/
```

### 2. 配置

编辑 `.trae/web-manager/config.json`，配置你的项目名称和路径：

```json
{
  "project": {
    "name": "你的项目名称",
    "title": "Todos Web Manager"
  }
}
```

### 3. 定制

1. 编辑 `.openclaw-memory/USER.md`，填入你的个人信息
2. 根据需要修改 `.openclaw-memory/MEMORY.md`，移除 CS-Notes 特定内容
3. 在 `.trae/rules/project/` 下创建你的项目特定规则

### 4. 启动服务

```bash
cd .trae/web-manager
pip install -r requirements.txt
python server.py
```

然后在浏览器中访问：http://localhost:5000

## 目录结构

```
项目根目录/
├── .trae/
│   ├── todos/              # Todo 数据
│   ├── documents/          # 文档
│   ├── rules/              # 规则
│   └── web-manager/        # Web Manager
└── .openclaw-memory/      # OpenClaw 记忆体系
```

## 特性

- ✅ 完全解耦，可迁移到任意新项目
- ✅ 配置文件驱动，易于定制
- ✅ 包含完整的 OpenClaw 记忆体系
- ✅ P0-P9 优先级体系支持
- ✅ Plan 机制本质化重构
- ✅ 规则分层架构（核心/模板/项目）

## 更新

当 CS-Notes 仓库有更新时，可以重新运行 build.sh 来获取最新版本。

EOF
echo -e "${GREEN}✓ README 文档已创建${NC}"
echo ""

# ============ 步骤 7: 创建压缩包 ============
echo -e "${YELLOW}[8/8] 创建压缩包...${NC}"
cd "$OUTPUT_DIR/.."
PACKAGE_NAME="todos-web-manager-$(date +%Y%m%d-%H%M%S).tar.gz"
tar -czf "$PACKAGE_NAME" -C "$OUTPUT_DIR" .
echo -e "${GREEN}✓ 压缩包已创建: $PACKAGE_NAME${NC}"
echo ""

# 完成
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}构建完成！${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "输出目录: $OUTPUT_DIR"
echo "压缩包: $(dirname "$OUTPUT_DIR")/$PACKAGE_NAME"
echo ""
echo "下一步："
echo "  1. 查看 $OUTPUT_DIR/README.md"
echo "  2. 将内容复制到你的新项目"
echo "  3. 根据项目需求进行定制"
echo ""
