# AGENTS.md

本文件适用于 Codex CLI/App 在本仓库内工作，补充而不替代 `.trae/rules/project_rules.md` 与 `.openclaw-memory/AGENTS.md`。

## 1. 仓库定位

`CS-Notes` 是一个复合工作系统，不只是笔记仓库：

- `Notes/`：长期知识管理
- `创作/`：写作与观点表达
- `公司项目/`：公司相关 WIP，默认视为私密内容
- `.trae/`：todo、规则、web 管理、执行控制面
- `.openclaw-memory/`、`.trae/openclaw-skills/`、`Notes/snippets/`、`.codex/`：agent / workflow / skill 实验区

工作时优先优化整个系统，而不是把文件当成彼此孤立的文档。

## 2. 默认原则

1. 先回答问题，再把结果落到最合适的位置。
2. 能自主推进就推进；只有在用户决策或手动操作不可替代时才停下。
3. 优先做真实改动，少做临时文件、演示性产物或伪完成。
4. 尊重现有结构与文风，尤其是 `Notes/`、`创作/`、`公司项目/`。
5. 重要结论尽量附上来源链接、文件路径或具体产物，保证可追溯。
6. 如果在“快速回复”和“可持续落盘”之间犹豫，优先后者。
7. 笔记整理类任务默认直接执行；高复杂度的代码/端到端任务可以先给简短 plan 供 review，或只问一两个关键问题。

## 3. Shell 与上下文

### Shell 策略

默认使用独立命令，保留并发能力：

```bash
zsh -lc 'source ~/.zshrc; <command>'
```

仓库已有 `.codex/environments/environment.toml` 配置 `script = "source ~/.zshrc"`，手动执行时也保持一致。

只有一串命令明确依赖共享 shell 状态、且反复加载 `~/.zshrc` 成本明显时，才使用：

```bash
/Users/bytedance/CS-Notes/Notes/snippets/codex-persistent-shell.sh
```

使用规则：

- 该脚本会在单个 TTY session 内只加载一次 `~/.zshrc`
- 适用于共享 cwd、env、alias、shell function、virtualenv / conda / nvm 状态
- 不要把 persistent shell 当全局默认，否则会削弱并发并增加状态污染风险

### 最小上下文

不要先通读整个仓库。按需加载：

- 常规必读：`README.md`、`.trae/rules/project_rules.md`、`.trae/documents/PROJECT_CONTEXT.md`
- workflow / memory：`.openclaw-memory/MEMORY.md`、`.openclaw-memory/AGENTS.md`、相关 memory 文件
- todo / 执行：`.trae/todos/todos.json`、相关 skill、`.trae/web-manager/WORKFLOW.md`
- 写作：先读 `创作/` 下 2-3 篇代表文章
- 笔记整合：先在 `Notes/` 广搜，再检查候选文件结构

### Markdown 文件打开

当需要在本机为用户打开 Markdown 文件时，默认使用 Typora：

```bash
open -a Typora <file.md>
```

## 4. 工作流

### A. 笔记整合

1. 先在 `Notes/` 中广搜最佳落点。
2. 修改 Markdown 前先看结构，优先用 `Notes/snippets/markdown_toc.py`。
3. 优先插入现有 section；确实没有合适位置再新增小节。
4. 语言尽量压缩，不为“更整洁”而删除用户原内容。
5. 外部材料必须附来源链接。
6. 一份材料跨多个主题时，拆分落到多个位置，不强塞进一个文件。

### B. 写作与公司项目

- 文风要求：平实、凝练、有立场、结构清晰、重分析与比较，避免 AI 套话。
- 做较大写作前，先读 `创作/` 下 2-3 篇文章对齐语气。
- 涉及 `公司项目/` 时，先读 `公司项目/01-公司项目创作pipeline.md`，并按该 pipeline 执行。

### C. Todo 驱动执行

- 单一数据源：`.trae/todos/todos.json`
- 新增 todo：优先使用 `todo-adder`
- 执行 todo：优先走 `priority-task-reader`
- 真正开工前，先把任务改成 `in-progress` 并写入 `started_at`
- 推进任务必须带来真实产物，不要只改状态文本
- 明确区分：AI 可独立完成的任务 vs 必须等待用户动作的任务

### D. Tooling / agent / web-manager

重点目录：`.trae/openclaw-skills/`、`.trae/web-manager/`、`.trae/web-manager/templates/`、`.openclaw-memory/`、`Notes/snippets/`、`.codex/`

处理这些目录时：

1. 保持 Codex、Trae、OpenClaw 的互通性。
2. 优先改善默认工作路径，而不是只做 demo。
3. 涉及模板、迁移、打包时，检查模板和脚本是否需要同步更新。
4. 注意区分“项目定制改动”与“通用模板改动”，避免写错层级。

## 5. Git、安全与边界

### 禁止外泄

绝不提交或暴露：

- `公司项目/` 下任何内容
- 密钥、token、密码、AK/SK、私有链接
- 环境文件或其他敏感配置

### Git 工作流

优先使用：

- `Notes/snippets/todo-pull.sh`
- `Notes/snippets/todo-push.sh`
- `Notes/snippets/todo-push-commit.sh`

commit / push 前必须：

1. 看 `git status`
2. 看 `git diff`
3. 如果使用 `todo-push.sh`，还要看 `git-diff-summary-*.md`
4. 确认没有带上禁推文件
5. 确认没有误删有价值的用户内容

禁止使用 force push 或其他高风险破坏性 git 操作，除非用户明确要求。

### Symlink 注意事项

`.openclaw-memory/` 下文件可能是其他工作区的 symlink 目标。修改时原地编辑，不要轻易删除重建。

## 6. 沟通风格与完成标准

- 默认中文，直接、简洁、少废话。
- 简单任务简答，复杂任务给清晰进展和关键决策。
- 卡住时说明具体 blocker，以及下一步需要用户做的不可替代动作。
- 尽量给出明确文件路径、产物路径和结论。

一个好的结果通常应满足：

- 放在正确文件，而不是随便找个地方
- 与现有结构和文风一致
- 有来源或路径支撑，可追溯
- 真正能接入当前工作流
- 安全、可继续、可提交
