#!/bin/bash
# Top Lean AI 榜单监控 Cron 配置脚本

CS_NOTES_DIR="/root/.openclaw/workspace/CS-Notes"
LOG_DIR="/var/log/openclaw"

# 创建日志目录
mkdir -p "$LOG_DIR"

# 配置 cron job
# 每天早上 9 点运行 Top Lean AI 榜单监控
# 每天每小时运行 Session 检查
(
  crontab -l 2>/dev/null
  echo ""
  echo "# Top Lean AI 榜单监控 - 每天早上 9 点运行"
  echo "0 9 * * * cd $CS_NOTES_DIR && python3 Notes/snippets/top_lean_ai_monitor.py check >> $LOG_DIR/top-lean-ai.log 2>&1"
  echo ""
  echo "# Session 检查 - 每小时运行"
  echo "0 * * * * cd $CS_NOTES_DIR && python3 Notes/snippets/session-optimizer.py check >> $LOG_DIR/session-optimizer.log 2>&1"
) | crontab -

echo "✅ Cron 配置完成！"
echo ""
echo "📝 Cron 配置内容："
crontab -l
echo ""
echo "📁 日志文件位置："
echo "   - Top Lean AI: $LOG_DIR/top-lean-ai.log"
echo "   - Session: $LOG_DIR/session-optimizer.log"
