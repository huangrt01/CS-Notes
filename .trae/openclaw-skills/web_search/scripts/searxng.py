#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["httpx", "rich"]
# ///
"""SearXNG CLI - Privacy-respecting metasearch via your local instance."""

import argparse
import os
import sys
import json
import warnings
import httpx
from rich.console import Console
from rich.table import Table
from rich import print as rprint
from urllib.parse import urlencode

# Suppress SSL warnings for local self-signed certificates
warnings.filterwarnings('ignore', message='Unverified HTTPS request')

console = Console()
SEARXNG_URL = os.getenv("SEARXNG_URL", "http://localhost:8080")

def search_searxng(
    query: str,
    limit: int = 10,
    category: str = "general",
    language: str = "auto",
    time_range: str = None,
    output_format: str = "table"
) -> dict:
    """
    Search using SearXNG instance.
    
    Args:
        query: Search query string
        limit: Number of results to return
        category: Search category (general, images, news, videos, etc.)
        language: Language code (auto, en, de, fr, etc.)
        time_range: Time range filter (day, week, month, year)
        output_format: Output format (table, json)
    
    Returns:
        Dict with search results
    """
    params = {
        "q": query,
        "format": "json",
        "categories": category,
    }
    
    if language != "auto":
        params["language"] = language
    
    if time_range:
        params["time_range"] = time_range
    
    try:
        # Disable SSL verification for local self-signed certs
        response = httpx.get(
            f"{SEARXNG_URL}/search",
            params=params,
            timeout=30,
            verify=False  # For local self-signed certs
        )
        response.raise_for_status()
        
        data = response.json()
        
        # Limit results
        if "results" in data:
            data["results"] = data["results"][:limit]
        
        return data
        
    except httpx.HTTPError as e:
        console.print(f"[red]Error connecting to SearXNG:[/red] {e}", file=sys.stderr)
        return {"error": str(e), "results": []}
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {e}", file=sys.stderr)
        return {"error": str(e), "results": []}


def display_results_table(data: dict, query: str):
    """Display search results in a rich table."""
    results = data.get("results", [])
    
    if not results:
        rprint(f"[yellow]No results found for:[/yellow] {query}")
        return
    
    table = Table(title=f"SearXNG Search: {query}", show_lines=False)
    table.add_column("#", style="dim", width=3)
    table.add_column("Title", style="bold")
    table.add_column("URL", style="blue", width=50)
    table.add_column("Engines", style="green", width=20)
    
    for i, result in enumerate(results, 1):
        title = result.get("title", "No title")[:70]
        url = result.get("url", "")[:45] + "..."
        engines = ", ".join(result.get("engines", []))[:18]
        
        table.add_row(
            str(i),
            title,
            url,
            engines
        )
    
    console.print(table)
    
    # Show additional info
    if data.get("number_of_results"):
        rprint(f"\n[dim]Total results available: {data['number_of_results']}[/dim]")
    
    # Show content snippets for top 3
    rprint("\n[bold]Top results:[/bold]")
    for i, result in enumerate(results[:3], 1):
        title = result.get("title", "No title")
        url = result.get("url", "")
        content = result.get("content", "")[:200]
        
        rprint(f"\n[bold cyan]{i}. {title}[/bold cyan]")
        rprint(f"   [blue]{url}[/blue]")
        if content:
            rprint(f"   [dim]{content}...[/dim]")


def display_results_json(data: dict):
    """Display results in JSON format for programmatic use."""
    print(json.dumps(data, indent=2))


def main():
    parser = argparse.ArgumentParser(
        description="SearXNG CLI - Search the web via your local SearXNG instance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  %(prog)s search "python asyncio"
  %(prog)s search "climate change" -n 20
  %(prog)s search "cute cats" --category images
  %(prog)s search "breaking news" --category news --time-range day
  %(prog)s search "rust tutorial" --format json

Environment:
  SEARXNG_URL: SearXNG instance URL (default: {SEARXNG_URL})
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Search command
    search_parser = subparsers.add_parser("search", help="Search the web")
    search_parser.add_argument("query", nargs="+", help="Search query")
    search_parser.add_argument(
        "-n", "--limit",
        type=int,
        default=10,
        help="Number of results (default: 10)"
    )
    search_parser.add_argument(
        "-c", "--category",
        default="general",
        choices=["general", "images", "videos", "news", "map", "music", "files", "it", "science"],
        help="Search category (default: general)"
    )
    search_parser.add_argument(
        "-l", "--language",
        default="auto",
        help="Language code (auto, en, de, fr, etc.)"
    )
    search_parser.add_argument(
        "-t", "--time-range",
        choices=["day", "week", "month", "year"],
        help="Time range filter"
    )
    search_parser.add_argument(
        "-f", "--format",
        choices=["table", "json"],
        default="table",
        help="Output format (default: table)"
    )
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    if args.command == "search":
        query = " ".join(args.query)
        
        data = search_searxng(
            query=query,
            limit=args.limit,
            category=args.category,
            language=args.language,
            time_range=args.time_range,
            output_format=args.format
        )
        
        if args.format == "json":
            display_results_json(data)
        else:
            display_results_table(data, query)


if __name__ == "__main__":
    main()
