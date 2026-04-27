# AGENTS.md

This file is for Codex CLI/App working inside this repository. It complements rather than replaces `.trae/rules/project_rules.md` and `.openclaw-memory/AGENTS.md`.

## 1. What this repository really is

`CS-Notes` is not just a notes repo. It is a combined system for:

- evergreen knowledge management (`Notes/`)
- writing and opinionated output (`创作/`)
- confidential work-in-progress for company topics (`公司项目/`)
- task management and execution control plane (`.trae/todos/`, `.trae/web-manager/`)
- agent / workflow / skill experimentation (`.trae/openclaw-skills/`, `Notes/snippets/`, `.openclaw-memory/`)

When you work here, optimize for the whole system instead of treating files as isolated documents.

## 2. The user's likely goals and working style

Assume the user usually wants one or more of the following:

1. **Answer the question, then land the result in the right place.**
   They do not just want a chat answer; they often want the result integrated into the repo.
2. **Push tasks forward autonomously.**
   Only stop when a real user decision or manual user-side action is required.
3. **End-to-end usefulness over toy completion.**
   "Implemented" is not enough; the result should be actually usable in this repo's workflow.
4. **Real edits, not theater.**
   Avoid generating piles of temp files or pseudo-work. Prefer directly improving the real target files.
5. **Preserve the user's voice and structure.**
   Especially in notes and writing, fit the existing style instead of imposing generic AI prose.
6. **Strong traceability.**
   Important conclusions should be attributable to source links, file paths, or concrete artifacts.

## 3. Session bootstrap for Codex

### Shell environment

Before the first real shell command in a session, use:

```bash
zsh -lc 'source ~/.zshrc; <command>'
```

This repository already persists that preference in:

- `.codex/environments/environment.toml`

which currently contains:

- `script = "source ~/.zshrc"`

Still, if you invoke shell commands manually, preserve this convention.

### Minimal context loading

Do not read the whole repo blindly. Start with the smallest relevant context.

- Always useful:
  - `README.md`
  - `.trae/rules/project_rules.md`
  - `.trae/documents/PROJECT_CONTEXT.md`
- If the task is about workflow / memory / agent behavior:
  - `.openclaw-memory/MEMORY.md`
  - `.openclaw-memory/AGENTS.md`
  - recent files under `.openclaw-memory/memory/` and `memory/`
- If the task is about todos / execution:
  - `.trae/todos/todos.json`
  - relevant skills under `.trae/openclaw-skills/`
  - `.trae/web-manager/WORKFLOW.md` when migration / packaging / template sync is involved
- If the task is about writing:
  - read 2-3 representative pieces under `创作/`
- If the task is about note integration:
  - search broadly in `Notes/` first, then inspect candidate files' structure before editing

## 4. Task playbooks

### A. Note curation / knowledge integration

This is a primary use case of the repo.

Follow this order:

1. Search broadly for the best existing destination in `Notes/`.
2. Before editing a Markdown target, inspect its structure first.
   - Prefer the existing `markdown-toc` skill or `Notes/snippets/markdown_toc.py`.
3. Insert into the most appropriate existing section whenever possible.
4. Create a new subsection only if there is truly no good fit.
5. Keep the wording compact.
6. Add source links for externally derived material.
7. Do not delete existing user content just to make the structure cleaner.
8. If one source spans multiple themes, split it across multiple files/sections rather than forcing it into one place.

Additional intent split:

- **Article / paper / post**: usually refine and integrate the knowledge.
- **Video / course / collected material**: often better treated as reference collection, quote block, or source pointer inside the right section.

### B. Writing / essay drafting

Another primary use case is producing high-signal writing in `创作/` and sometimes `公司项目/`.

Required style tendencies:

- plain, condensed, not over-decorated
- opinionated, not blandly neutral
- structured and analytical
- good at comparison and abstraction
- minimal AI fluff, no forced enthusiasm, no emoji spam unless the file already clearly wants it

Before substantial writing edits, read a few representative pieces in `创作/` and align with that voice.

If the task touches `公司项目/`, first read:

- `公司项目/01-公司项目创作pipeline.md`

and follow that pipeline strictly.

### C. Todo-driven execution

This repo has a real task operating system. Respect it.

Core rules:

- Single source of truth: `.trae/todos/todos.json`
- When adding a todo, prefer the `todo-adder` skill instead of hand-editing JSON.
- When executing todos, prefer the `priority-task-reader` flow.
- Before starting a pending todo for real work, mark it `in-progress` and add `started_at`.
- Distinguish clearly between:
  - tasks AI can finish alone
  - tasks blocked on explicit user action or decision

The user's deeper preference is: push each task as far as possible until the next real dependency on them is clear.

Do not "advance" a todo by only changing status text. Advance it with actual work.

### D. Tooling / agent system / web manager work

A large part of this repo is an agent-workflow laboratory.

High-value directories:

- `.trae/openclaw-skills/`
- `.trae/web-manager/`
- `.trae/web-manager/templates/`
- `.openclaw-memory/`
- `Notes/snippets/`
- `.codex/`

When editing these areas:

1. Preserve interoperability across Codex, Trae, and OpenClaw.
2. Favor changes that improve the default working path, not just a demo path.
3. If a feature is also packaged / migrated elsewhere, check whether templates or build scripts must be updated too.
4. Keep an eye on whether a project-specific change is accidentally being written into a generic template, or vice versa.

## 5. Git, safety, and boundaries

### Never leak or publish the wrong things

Absolutely avoid committing or exposing:

- anything under `公司项目/` to the public repo
- secrets, tokens, passwords, AK/SK, private URLs that should not be public
- accidental environment files

`.gitignore` and helper scripts already encode part of this policy; still verify manually.

### Preferred repo Git workflow

When the user asks for pull / push / sync behavior, prefer the repo's standard scripts:

- `Notes/snippets/todo-pull.sh`
- `Notes/snippets/todo-push.sh`
- `Notes/snippets/todo-push-commit.sh`

Before any commit/push:

1. inspect `git status`
2. inspect `git diff`
3. if using `todo-push.sh`, read the generated `git-diff-summary-*.md`
4. confirm no forbidden files are included
5. confirm no meaningful user content was accidentally deleted

Never use force push or similarly destructive Git operations unless the user explicitly asks.

### Symlink awareness

Files under `.openclaw-memory/` may be symlink targets for another workspace setup. Modify them in place; do not delete-and-recreate them casually.

## 6. Communication style inside Codex

Default to Chinese unless the user asks otherwise.

Preferred interaction style:

- direct, competent, low-filler
- concise for simple tasks, thorough for complex ones
- report real progress on long tasks
- when blocked, explain the exact blocker and the next irreversible user action needed
- mention concrete file paths and concrete outputs

## 7. What good work looks like in this repo

A strong result in this repository usually has these properties:

- placed in the right file, not just any file
- aligned with existing structure and style
- connected to source links or artifacts
- useful in the actual workflow, not just theoretically correct
- safe to commit
- easy for the user or another agent to continue from

If you are unsure between a quick local answer and a durable repo improvement, bias toward the durable repo improvement when it matches the user's request.
