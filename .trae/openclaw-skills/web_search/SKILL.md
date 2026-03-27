---
name: web_search
description: Privacy-respecting metasearch using your local SearXNG instance. Search the web, images, news, and more without external API dependencies.
author: Avinash Venkatswamy
version: 1.0.1
homepage: https://searxng.org
triggers:
  - "search for"
  - "search web"
  - "find information"
  - "look up"
metadata: {"clawdbot":{"emoji":"🔍","requires":{"bins":["python3"]},"config":{"env":{"SEARXNG_URL":{"description":"SearXNG instance URL","default":"http://localhost:8080","required":true}}}}}
---

# SearXNG Search

Search the web using your local SearXNG instance - a privacy-respecting metasearch engine.

## Commands

### Web Search
```bash
uv run {baseDir}/scripts/searxng.py search "query"              # Top 10 results
uv run {baseDir}/scripts/searxng.py search "query" -n 20        # Top 20 results
uv run {baseDir}/scripts/searxng.py search "query" --format json # JSON output
```

### Category Search
```bash
uv run {baseDir}/scripts/searxng.py search "query" --category images
uv run {baseDir}/scripts/searxng.py search "query" --category news
uv run {baseDir}/scripts/searxng.py search "query" --category videos
```

### Advanced Options
```bash
uv run {baseDir}/scripts/searxng.py search "query" --language en
uv run {baseDir}/scripts/searxng.py search "query" --time-range day
```

## Configuration

**Required:** Set the `SEARXNG_URL` environment variable to your SearXNG instance:

```bash
export SEARXNG_URL=https://your-searxng-instance.com
```

Or configure in your Clawdbot config:
```json
{
  "env": {
    "SEARXNG_URL": "https://your-searxng-instance.com"
  }
}
```

Default (if not set): `http://localhost:8080`

## Features

- 🔒 Privacy-focused (uses your local instance)
- 🌐 Multi-engine aggregation
- 📰 Multiple search categories
- 🎨 Rich formatted output
- 🚀 Fast JSON mode for programmatic use

## API

Uses your local SearXNG JSON API endpoint (no authentication required by default).
