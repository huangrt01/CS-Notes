#!/bin/bash
# Todo 同步脚本 - 一键拉取 git + 扫描新任务 + 生成执行提示 + 自动提交推送
# 支持日志记录、冲突检测、自动 commit & push

# 配置
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG_DIR="$REPO_ROOT/.trae/logs"
LOG_FILE="$LOG_DIR/todo-sync-$(date +%Y%m%d).log"
TODO_MANAGER="$REPO_ROOT/Notes/snippets/todo-manager.py"
TODO_PROMPT="$REPO_ROOT/Notes/snippets/todo-prompt.py"
DEFAULT_BRANCH="master"

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

# 检查 git 仓库状态
check_git_status() {
    cd "$REPO_ROOT" || return 1
    
    # 检查是否有未提交的更改
    if git status --porcelain | grep -q .; then
        log "WARN" "检测到未提交的更改"
        return 1
    fi
    return 0
}

# 检查冲突
check_conflicts() {
    cd "$REPO_ROOT" || return 1
    
    # 检查是否有合并冲突
    if git ls-files -u | grep -q .; then
        log "ERROR" "检测到 Git 合并冲突！请手动解决后再继续"
        echo "=== 冲突文件 ==="
        git ls-files -u
        return 1
    fi
    return 0
}

# 增强的 git 拉取
git_pull_enhanced() {
    cd "$REPO_ROOT" || error_exit "无法进入仓库目录"
    
    log "INFO" "步骤 1/5: 拉取 git 最新代码"
    
    # 先检查当前分支
    local current_branch=$(git rev-parse --abbrev-ref HEAD)
    log "INFO" "当前分支: $current_branch"
    
    # 如果不是默认分支，提示用户
    if [ "$current_branch" != "$DEFAULT_BRANCH" ]; then
        log "WARN" "当前分支不是 $DEFAULT_BRANCH，是否切换？(y/n, 默认: n)"
        read -t 10 -r response || response="n"
        if [ "$response" = "y" ] || [ "$response" = "Y" ]; then
            git checkout "$DEFAULT_BRANCH" >> "$LOG_FILE" 2>&1 || {
                log "ERROR" "切换分支失败"
                return 1
            }
            log "INFO" "已切换到 $DEFAULT_BRANCH 分支"
        fi
    fi
    
    # 先获取远程更新
    log "INFO" "获取远程更新..."
    git fetch origin >> "$LOG_FILE" 2>&1 || {
        log "WARN" "获取远程更新失败"
        return 1
    }
    
    # 检查是否有本地未提交的更改
    if ! check_git_status; then
        log "WARN" "有未提交的更改，是否先 stash？(y/n, 默认: y)"
        read -t 10 -r response || response="y"
        if [ "$response" = "y" ] || [ "$response" = "Y" ]; then
            git stash push -m "todo-sync: auto stash before pull" >> "$LOG_FILE" 2>&1
            log "INFO" "已 stash 本地更改"
            local stash_saved=1
        else
            log "WARN" "保留本地更改，可能会有冲突"
        fi
    fi
    
    # 执行 pull
    if git pull origin "$DEFAULT_BRANCH" >> "$LOG_FILE" 2>&1; then
        log "INFO" "Git 拉取成功"
        
        # 如果之前有 stash，尝试恢复
        if [ "$stash_saved" = 1 ]; then
            log "INFO" "尝试恢复 stash..."
            if git stash pop >> "$LOG_FILE" 2>&1; then
                log "INFO" "Stash 恢复成功"
            else
                log "WARN" "Stash 恢复可能有冲突，请检查"
                check_conflicts
            fi
        fi
        return 0
    else
        log "ERROR" "Git 拉取失败，请检查日志"
        check_conflicts
        return 1
    fi
}

# 自动 commit & push
git_commit_push() {
    cd "$REPO_ROOT" || error_exit "无法进入仓库目录"
    
    log "INFO" "步骤 5/5: 检查是否需要提交更改"
    
    # 检查是否有更改
    if ! git status --porcelain | grep -q .; then
        log "INFO" "没有需要提交的更改"
        return 0
    fi
    
    # 显示更改
    log "INFO" "检测到以下更改："
    git status --short | while read -r line; do
        log "INFO" "  $line"
    done
    
    # 询问是否提交
    echo
    log "INFO" "是否提交并推送这些更改？(y/n, 默认: y)"
    read -t 30 -r response || response="y"
    
    if [ "$response" != "y" ] && [ "$response" != "Y" ]; then
        log "INFO" "跳过提交"
        return 0
    fi
    
    # 输入提交信息
    echo
    log "INFO" "请输入提交信息 (默认: 'update: auto commit by todo-sync.sh'):"
    read -r commit_msg
    if [ -z "$commit_msg" ]; then
        commit_msg="update: auto commit by todo-sync.sh"
    fi
    
    # 执行 commit
    log "INFO" "执行 git add..."
    git add . >> "$LOG_FILE" 2>&1
    
    log "INFO" "执行 git commit..."
    if git commit -m "$commit_msg" >> "$LOG_FILE" 2>&1; then
        log "INFO" "Commit 成功"
    else
        log "WARN" "Commit 可能失败，请检查"
        return 1
    fi
    
    # 执行 push
    log "INFO" "执行 git push..."
    if git push origin "$DEFAULT_BRANCH" >> "$LOG_FILE" 2>&1; then
        log "INFO" "Push 成功！"
        return 0
    else
        log "ERROR" "Push 失败，请检查日志"
        return 1
    fi
}

# 主函数
main() {
    log "INFO" "========================================"
    log "INFO" "Todo 同步脚本开始执行"
    log "INFO" "========================================"
    
    # 1. 拉取 git 最新代码（增强版）
    git_pull_enhanced
    local pull_result=$?
    
    # 2. 扫描 Inbox 中的新任务
    log "INFO" "步骤 2/5: 扫描 Inbox 中的新任务"
    
    if [ ! -f "$TODO_MANAGER" ]; then
        error_exit "找不到 todo-manager.py: $TODO_MANAGER"
    fi

    if [ ! -x "$TODO_MANAGER" ]; then
        chmod +x "$TODO_MANAGER"
    fi

    log "INFO" "运行 todo-manager.py..."
    "$TODO_MANAGER" 2>&1 | tee -a "$LOG_FILE"
    local todo_exit_code=${PIPESTATUS[0]}
    
    if [ $todo_exit_code -eq 0 ]; then
        log "INFO" "Todo 管理器执行成功"
    else
        log "WARN" "Todo 管理器执行完成，退出码: $todo_exit_code"
    fi

    # 3. 生成待执行任务提示清单
    log "INFO" "步骤 3/5: 生成待执行任务提示清单"
    
    if [ ! -f "$TODO_PROMPT" ]; then
        error_exit "找不到 todo-prompt.py: $TODO_PROMPT"
    fi

    if [ ! -x "$TODO_PROMPT" ]; then
        chmod +x "$TODO_PROMPT"
    fi

    log "INFO" "运行 todo-prompt.py..."
    "$TODO_PROMPT" 2>&1 | tee -a "$LOG_FILE"
    local prompt_exit_code=${PIPESTATUS[0]}
    
    if [ $prompt_exit_code -eq 0 ]; then
        log "INFO" "任务提示生成器执行成功"
    else
        log "WARN" "任务提示生成器执行完成，退出码: $prompt_exit_code"
    fi

    # 4. 完成中间步骤
    log "INFO" "步骤 4/5: 同步完成"
    
    # 5. 自动 commit & push（可选）
    git_commit_push

    # 完成
    log "INFO" "========================================"
    log "INFO" "Todo 同步脚本执行完毕"
    log "INFO" "日志文件: $LOG_FILE"
    log "INFO" "========================================"
}

# 运行主函数
main
