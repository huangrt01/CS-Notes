#!/usr/bin/env python3
"""
Speech-to-Text Tool
Uses OpenAI Whisper API or local Whisper model
"""

import sys
import os
import argparse

# Try to import openai-whisper
try:
    import whisper
    HAS_WHISPER = True
except ImportError:
    HAS_WHISPER = False
    print("Warning: openai-whisper not installed, trying OpenAI API...")

# Try to import openai
try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    print("Warning: openai not installed")


def transcribe_audio(file_path: str, model: str = "large-v3") -> str:
    """
    Transcribe audio file to text
    """
    if HAS_WHISPER:
        print(f"Using local Whisper model: {model}")
        model = whisper.load_model(model)
        result = model.transcribe(file_path)
        return result["text"]
    elif HAS_OPENAI:
        print("Using OpenAI Whisper API")
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        with open(file_path, "rb") as f:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=f
            )
        return transcript.text
    else:
        raise ImportError("Neither openai-whisper nor openai installed")


def main():
    parser = argparse.ArgumentParser(description="Speech-to-Text Tool")
    parser.add_argument("file", help="Audio file path")
    parser.add_argument("--model", default="large-v3", help="Whisper model (base, small, medium, large, large-v2, large-v3)")
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"Error: File {args.file} not found")
        sys.exit(1)

    try:
        text = transcribe_audio(args.file, args.model)
        print("\n--- Transcription ---\n")
        print(text)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
