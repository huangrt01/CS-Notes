---
name: github-gh-cli
description: Inspect GitHub pull requests, issues, workflow runs, and API data with the gh CLI. Use when the user wants repository facts from GitHub without opening the browser, especially for PR status, CI failures, issue lists, or structured JSON output.
---

# GitHub GH CLI

## Use this skill when

Use this skill when the task is about GitHub data and `gh` is the fastest path:
- pull request status or checks;
- recent workflow runs or failed jobs;
- issue or PR lists;
- structured JSON output for scripting; or
- GitHub API queries not covered by simpler commands.

## Working rules

- Prefer `gh` over browser navigation when command-line access is enough.
- Prefer `--json` and `--jq` when you need structured results.
- When outside the target repository, pass `--repo owner/repo` explicitly.
- If auth is missing, ask the user to handle `gh auth login`; do not attempt interactive credential setup yourself.

### Contributor scope and permission advice

- Inventory a stated time window before recommending a contributor's scope; disclose date bounds and result caps. Sample across modules and work types, including CI/mainline recovery, build tooling, architecture upkeep and regression follow-through. A few recent feature PRs do not establish the person's full role.
- Inspect substantive PR bodies, selected diffs and follow-up records. Separate public author/test claims from independently checked behavior; PR volume alone proves neither quality nor review qualification.
- Check both formal cross-author reviews and ordinary PR comments. A `reviewed-by:` search misses comment-only validation and responsibility for regressions; missing search results are not proof of no review work.
- Distinguish repository management permissions, maintainer responsibility, CODEOWNERS eligibility and bypass authority. Apply each appointment's actual rules without treating a subsystem checklist as the whole assessment of repository-wide maintenance.

## Common commands

### Pull requests

```bash
gh pr view 123 --repo owner/repo
gh pr checks 123 --repo owner/repo
gh pr list --repo owner/repo --limit 20
```

### Actions / CI

```bash
gh run list --repo owner/repo --limit 10
gh run view <run-id> --repo owner/repo
gh run view <run-id> --repo owner/repo --log-failed
```

### Issues

```bash
gh issue list --repo owner/repo --limit 20
gh issue list --repo owner/repo --json number,title,state --jq '.[]'
```

### API

```bash
gh api repos/owner/repo/pulls/123
gh api repos/owner/repo/issues --jq '.[].title'
```
