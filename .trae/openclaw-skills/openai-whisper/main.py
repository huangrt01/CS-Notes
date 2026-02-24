#!/usr/bin/env python3
"""
OpenAI Whisper Skill - Local speech-to-text with the Whisper CLI (no API key)
"""

import sys
import os
import argparse
from pathlib import Path

# 添加 snippets 目录到 sys.path，以便导入 task_execution_logger
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "Notes" / "snippets"))

try:
    from task_execution_logger import (
        TaskExecutionLogger,
        TaskStage,
        LogLevel,
        TaskArtifact,
        create_logger
    )
    TASK_LOGGER_AVAILABLE = True
except ImportError:
    TASK_LOGGER_AVAILABLE = False

# 配置路径 - 支持多路径检测
REPO_ROOT_CANDIDATES = [
    Path("/root/.openclaw/workspace/CS-Notes"),
    Path("/Users/bytedance/CS-Notes"),
    Path(__file__).parent.parent.parent
]

REPO_ROOT = None
for candidate in REPO_ROOT_CANDIDATES:
    if candidate.exists():
        REPO_ROOT = candidate
        break

if REPO_ROOT is None:
    REPO_ROOT = Path.cwd()

# 初始化任务日志系统
task_logger = None
if TASK_LOGGER_AVAILABLE:
    try:
        task_logger = create_logger(REPO_ROOT)
    except Exception as e:
        print(f"⚠️ 初始化任务日志系统失败: {e}")

# 导入 speech_to_text
sys.path.insert(0, str(REPO_ROOT / "Notes" / "snippets"))
try:
    from speech_to_text import transcribe_audio
    SPEECH_TO_TEXT_AVAILABLE = True
except ImportError:
    SPEECH_TO_TEXT_AVAILABLE = False


class OpenAIWhisperSkill:
    """OpenAI Whisper Skill"""
    
    def __init__(self):
        self.repo_root = REPO_ROOT
    
    def transcribe(self, audio_path: str, model: str = "medium") -> dict:
        """
        转录音频文件
        
        参数:
            audio_path: 音频文件路径
            model: Whisper 模型 (base, small, medium, large, large-v2, large-v3, large-v3-turbo, turbo)
        
        返回:
            转录结果字典
        """
        audio_path_obj = Path(audio_path)
        if not audio_path_obj.exists():
            return {
                "success": False,
                "error": f"音频文件不存在: {audio_path}"
            }
        
        if not SPEECH_TO_TEXT_AVAILABLE:
            return {
                "success": False,
                "error": "speech_to_text 模块不可用"
            }
        
        try:
            result = transcribe_audio(str(audio_path_obj), model)
            
            return {
                "success": True,
                "audio_path": str(audio_path_obj),
                "model": model,
                "text": result
            }
        
        except Exception as e:
            return {
                "success": False,
                "error": f"转录失败: {type(e).__name__}: {e}"
            }


def handle_transcribe():
    """处理转录音频命令"""
    from datetime import datetime
    task_id = f"openai-whisper-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    agent = "openclaw"
    
    # 记录任务开始
    if TASK_LOGGER_AVAILABLE and task_logger:
        try:
            task_logger.start_task(task_id, agent=agent)
            task_logger.log_info(
                task_id,
                TaskStage.PLANNING,
                "开始转录音频",
                {},
                agent=agent
            )
        except Exception as e:
            print(f"⚠️ 记录任务日志失败: {e}")
    
    parser = argparse.ArgumentParser(description="OpenAI Whisper Skill")
    parser.add_argument("audio", help="Path to audio file")
    parser.add_argument("--model", default="medium", 
                       choices=["base", "small", "medium", "large", "large-v2", "large-v3", "large-v3-turbo", "turbo"],
                       help="Whisper model (default: medium)")
    
    # 解析参数
    if len(sys.argv) < 2:
        error_msg = "Missing audio file path"
        if TASK_LOGGER_AVAILABLE and task_logger:
            try:
                task_logger.fail_task(task_id, error_msg)
            except Exception as e:
                print(f"⚠️ 记录任务失败日志失败: {e}")
        print("Usage: python main.py transcribe <audio_path> [--model model_name]")
        return
    
    # 移除 "transcribe" 命令
    if sys.argv[1] == "transcribe":
        sys.argv = sys.argv[:1] + sys.argv[2:]
    
    try:
        args = parser.parse_args()
        
        if TASK_LOGGER_AVAILABLE and task_logger:
            task_logger.log_info(
                task_id,
                TaskStage.EXECUTING,
                "转录音频文件",
                {"audio_path": args.audio, "model": args.model},
                agent=agent
            )
        
        skill = OpenAIWhisperSkill()
        result = skill.transcribe(args.audio, model=args.model)
        
        if TASK_LOGGER_AVAILABLE and task_logger:
            if result.get("success"):
                task_logger.log_success(
                    task_id,
                    TaskStage.COMPLETED,
                    "音频转录成功",
                    {"model": args.model},
                    agent=agent
                )
                task_logger.complete_task(task_id, agent=agent)
            else:
                task_logger.fail_task(task_id, result.get("error", "Unknown error"))
        
        import json
        return json.dumps(result, ensure_ascii=False)
    
    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        if TASK_LOGGER_AVAILABLE and task_logger:
            try:
                task_logger.fail_task(task_id, error_msg)
            except Exception as log_e:
                print(f"⚠️ 记录任务失败日志失败: {log_e}")
        import json
        return json.dumps({"success": False, "error": error_msg}, ensure_ascii=False)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "transcribe":
        print(handle_transcribe())
    else:
        print("Usage: python main.py transcribe <audio_path> [--model model_name]")
