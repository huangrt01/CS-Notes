#!/bin/bash
# Todo 同步脚本 - 一键拉取 git + 扫描新任务
# 支持日志记录功能

# 配置
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG_DIR="$REPO_ROOT/.trae/logs"
LOG_FILE="$LOG_DIR/todo-sync-$(date +%Y%m%d).log"
TODO_MANAGER="$REPO_ROOT/Notes/snippets/todo-manager.py"

# 创建日志目录
mkdir -p "$LOG_DIR"

# 日志函数
log() {
    local level="$1"
    shift
    local message="$*"
    local timestamp="$(date '+%Y-%m-%d %H:%M:%S')"
    echo "[$timestamp] [$level] $message" | tee -a "$LOG_FILE"
}

# 错误处理函数
error_exit() {
    log "ERROR" "$1"
    exit 1
}

# 主函数
main() {
    log "INFO" "========================================"
    log "INFO" "Todo 同步脚本开始执行"
    log "INFO" "========================================"

    # 1. 拉取 git 最新代码
    log "INFO" "步骤 1/3: 拉取 git 最新代码"
    cd "$REPO_ROOT" || error_exit "无法进入仓库目录"
    
    if git pull origin master >> "$LOG_FILE" 2>&1; then
        log "INFO" "Git 拉取成功"
    else
        log "WARN" "Git 拉取可能有问题，请检查日志"
    fi

    # 2. 扫描 Inbox 中的新任务
    log "INFO" "步骤 2/3: 扫描 Inbox 中的新任务"
    
    if [ ! -f "$TODO_MANAGER" ]; then
        error_exit "找不到 todo-manager.py: $TODO_MANAGER"
    fi

    if [ ! -x "$TODO_MANAGER" ]; then
        chmod +x "$TODO_MANAGER"
    fi

    # 运行 todo-manager.py 并记录输出
    log "INFO" "运行 todo-manager.py..."
    "$TODO_MANAGER" 2>&1 | tee -a "$LOG_FILE"
    local todo_exit_code=${PIPESTATUS[0]}
    
    if [ $todo_exit_code -eq 0 ]; then
        log "INFO" "Todo 管理器执行成功"
    else
        log "WARN" "Todo 管理器执行完成，退出码: $todo_exit_code"
    fi

    # 3. 完成
    log "INFO" "步骤 3/3: 同步完成"
    log "INFO" "========================================"
    log "INFO" "Todo 同步脚本执行完毕"
    log "INFO" "日志文件: $LOG_FILE"
    log "INFO" "========================================"
}

# 运行主函数
main
