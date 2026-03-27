# Changelog

All notable changes to the SearXNG skill will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.1] - 2026-01-26

### Changed
- **Security:** Changed default SEARXNG_URL from hardcoded private URL to generic `http://localhost:8080`
- **Configuration:** Made SEARXNG_URL required configuration (no private default)
- Updated all documentation to emphasize configuration requirement
- Removed hardcoded private URL from all documentation

### Security
- Eliminated exposure of private SearXNG instance URL in published code

## [1.0.0] - 2026-01-26

### Added
- Initial release
- Web search via local SearXNG instance
- Multiple search categories (general, images, videos, news, map, music, files, it, science)
- Time range filters (day, week, month, year)
- Rich table output with result snippets
- JSON output mode for programmatic use
- SSL self-signed certificate support
- Configurable SearXNG instance URL via SEARXNG_URL env var
- Comprehensive error handling
- Rich CLI with argparse

### Features
- Privacy-focused (all searches local)
- No API keys required
- Multi-engine result aggregation
- Beautiful formatted output
- Language selection support
