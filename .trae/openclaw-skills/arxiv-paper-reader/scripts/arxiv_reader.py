#!/usr/bin/env python3
"""Fetch and extract text from arxiv paper HTML pages.

Usage:
    python arxiv_reader.py <arxiv_id_or_url> [--output FILE] [--raw] [--sections]

Examples:
    python arxiv_reader.py 2507.02259
    python arxiv_reader.py https://arxiv.org/abs/2507.02259
    python arxiv_reader.py 2507.02259 --output /tmp/paper.txt
    python arxiv_reader.py 2507.02259 --sections
    python arxiv_reader.py 2507.02259 --raw
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile


def parse_arxiv_id(input_str: str) -> str:
    input_str = input_str.strip()
    m = re.match(r'(?:https?://arxiv\.org/(?:abs|html|pdf)/)?(\d{4}\.\d{4,5})(?:v\d+)?', input_str)
    if m:
        return m.group(1)
    if re.match(r'\d{4}\.\d{4,5}$', input_str):
        return input_str
    raise ValueError(f"Invalid arxiv ID or URL: {input_str}")


def fetch_html(arxiv_id: str) -> str:
    url = f"https://arxiv.org/html/{arxiv_id}v1"
    try:
        result = subprocess.run(
            ["curl", "-s", "-L", "--max-time", "60", url],
            capture_output=True, text=True, check=True
        )
        html = result.stdout
        if not html or len(html) < 100:
            raise RuntimeError(f"Empty or too short response from {url}")
        return html
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"curl failed: {e.stderr}")


def html_to_text(html: str) -> str:
    html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL)
    html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL)
    html = re.sub(r'<nav[^>]*>.*?</nav>', '', html, flags=re.DOTALL)
    html = re.sub(r'<header[^>]*>.*?</header>', '', html, flags=re.DOTALL)
    html = re.sub(r'<footer[^>]*>.*?</footer>', '', html, flags=re.DOTALL)
    html = re.sub(r'<sup[^>]*>.*?</sup>', '', html, flags=re.DOTALL)
    for tag in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
        html = re.sub(rf'<{tag}[^>]*>', '\n\n', html)
        html = re.sub(rf'</{tag}>', '\n\n', html)
    html = re.sub(r'<br\s*/?>', '\n', html)
    html = re.sub(r'<p[^>]*>', '\n', html)
    html = re.sub(r'</p>', '\n', html)
    html = re.sub(r'<li[^>]*>', '\n- ', html)
    html = re.sub(r'<[^>]+>', '', html)
    html = re.sub(r'&lt;', '<', html)
    html = re.sub(r'&gt;', '>', html)
    html = re.sub(r'&amp;', '&', html)
    html = re.sub(r'&nbsp;', ' ', html)
    html = re.sub(r'&#\d+;', '', html)
    html = re.sub(r'\n{3,}', '\n\n', html)
    lines = []
    for line in html.split('\n'):
        stripped = line.strip()
        if stripped:
            lines.append(stripped)
    text = '\n'.join(lines)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def extract_sections(text: str) -> str:
    section_pattern = re.compile(
        r'^((?:\d+\.)*\d+(?:\.\d+)*\s+[\w][^\n]{0,80})$', re.MULTILINE
    )
    parts = section_pattern.split(text)
    if len(parts) <= 1:
        return text
    result = []
    for i, part in enumerate(parts):
        part = part.strip()
        if not part:
            continue
        if section_pattern.match(part):
            result.append(f"\n{'='*60}\n{part}\n{'='*60}")
        else:
            result.append(part)
    return '\n'.join(result)


def clean_math_artifacts(text: str) -> str:
    text = re.sub(r'italic_[a-z_]+\s*', '', text)
    text = re.sub(r'start_POSTSUBSCRIPT\s*', '_', text)
    text = re.sub(r'end_POSTSUBSCRIPT\s*', '', text)
    text = re.sub(r'start_POSTSUBSCRIPT\s*([a-zA-Z]+)\s*end_POSTSUBSCRIPT', r'_\1', text)
    text = re.sub(r'over\^?\s*start_ARG\s*([^}]+)\s*end_ARG', r'\1^', text)
    text = re.sub(r'bold_[a-z]+\s*', '', text)
    text = re.sub(r'caligraphic_[A-Z]\s*', '', text)
    text = re.sub(r'blackboard_[A-Z]\s*', '', text)
    text = re.sub(r'roman_(\w+)', r'\1', text)
    text = re.sub(r'\s{2,}', ' ', text)
    return text


def main():
    parser = argparse.ArgumentParser(description="Fetch and extract text from arxiv paper HTML")
    parser.add_argument("input", help="arxiv ID (e.g. 2507.02259) or URL")
    parser.add_argument("--output", "-o", help="Output file path (default: stdout)")
    parser.add_argument("--raw", action="store_true", help="Skip math cleanup")
    parser.add_argument("--sections", action="store_true", help="Add section separators")
    args = parser.parse_args()

    try:
        arxiv_id = parse_arxiv_id(args.input)
        print(f"Fetching https://arxiv.org/html/{arxiv_id}v1 ...", file=sys.stderr)
        html = fetch_html(arxiv_id)
        text = html_to_text(html)
        if not args.raw:
            text = clean_math_artifacts(text)
        if args.sections:
            text = extract_sections(text)
        if args.output:
            os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(text)
            print(f"Written to {args.output} ({len(text)} chars)", file=sys.stderr)
        else:
            print(text)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
