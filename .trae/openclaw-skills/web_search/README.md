# SearXNG Search Skill for Clawdbot

Privacy-respecting web search using your local SearXNG instance.

## Prerequisites

**This skill requires a running SearXNG instance.**

If you don't have SearXNG set up yet:

1. **Docker (easiest)**:
   ```bash
   docker run -d -p 8080:8080 searxng/searxng
   ```

2. **Manual installation**: Follow the [official guide](https://docs.searxng.org/admin/installation.html)

3. **Public instances**: Use any public SearXNG instance (less private)

## Features

- 🔒 **Privacy-focused**: Uses your local SearXNG instance
- 🌐 **Multi-engine**: Aggregates results from multiple search engines
- 📰 **Multiple categories**: Web, images, news, videos, and more
- 🎨 **Rich output**: Beautiful table formatting with result snippets
- 🚀 **Fast JSON mode**: Programmatic access for scripts and integrations

## Quick Start

### Basic Search
```
Search "python asyncio tutorial"
```

### Advanced Usage
```
Search "climate change" with 20 results
Search "cute cats" in images category
Search "breaking news" in news category from last day
```

## Configuration

**You must configure your SearXNG instance URL before using this skill.**

### Set Your SearXNG Instance

Configure the `SEARXNG_URL` environment variable in your Clawdbot config:

```json
{
  "env": {
    "SEARXNG_URL": "https://your-searxng-instance.com"
  }
}
```

Or export it in your shell:
```bash
export SEARXNG_URL=https://your-searxng-instance.com
```

## Direct CLI Usage

You can also use the skill directly from the command line:

```bash
# Basic search
uv run ~/clawd/skills/searxng/scripts/searxng.py search "query"

# More results
uv run ~/clawd/skills/searxng/scripts/searxng.py search "query" -n 20

# Category search
uv run ~/clawd/skills/searxng/scripts/searxng.py search "query" --category images

# JSON output (for scripts)
uv run ~/clawd/skills/searxng/scripts/searxng.py search "query" --format json

# Time-filtered news
uv run ~/clawd/skills/searxng/scripts/searxng.py search "latest AI news" --category news --time-range day
```

## Available Categories

- `general` - General web search (default)
- `images` - Image search
- `videos` - Video search
- `news` - News articles
- `map` - Maps and locations
- `music` - Music and audio
- `files` - File downloads
- `it` - IT and programming
- `science` - Scientific papers and resources

## Time Ranges

Filter results by recency:
- `day` - Last 24 hours
- `week` - Last 7 days
- `month` - Last 30 days
- `year` - Last year

## Examples

### Web Search
```bash
uv run ~/clawd/skills/searxng/scripts/searxng.py search "rust programming language"
```

### Image Search
```bash
uv run ~/clawd/skills/searxng/scripts/searxng.py search "sunset photography" --category images -n 10
```

### Recent News
```bash
uv run ~/clawd/skills/searxng/scripts/searxng.py search "tech news" --category news --time-range day
```

### JSON Output for Scripts
```bash
uv run ~/clawd/skills/searxng/scripts/searxng.py search "python tips" --format json | jq '.results[0]'
```

## SSL/TLS Notes

The skill is configured to work with self-signed certificates (common for local SearXNG instances). If you need strict SSL verification, edit the script and change `verify=False` to `verify=True` in the httpx request.

## Troubleshooting

### Connection Issues

If you get connection errors:

1. **Check your SearXNG instance is running:**
   ```bash
   curl -k $SEARXNG_URL
   # Or: curl -k http://localhost:8080 (default)
   ```

2. **Verify the URL in your config**
3. **Check SSL certificate issues**

### No Results

If searches return no results:

1. Check your SearXNG instance configuration
2. Ensure search engines are enabled in SearXNG settings
3. Try different search categories

## Privacy Benefits

- **No tracking**: All searches go through your local instance
- **No data collection**: Results are aggregated locally
- **Engine diversity**: Combines results from multiple search providers
- **Full control**: You manage the SearXNG instance

## About SearXNG

SearXNG is a free, open-source metasearch engine that respects your privacy. It aggregates results from multiple search engines while not storing your search data.

Learn more: https://docs.searxng.org/

## License

This skill is part of the Clawdbot ecosystem and follows the same license terms.
