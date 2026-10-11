# Integration with research workflows

This skill owns retrieval and interpretation technique. A research orchestrator
owns the question decomposition, evidence state, budgets and completion rules.
Do not create a parallel ledger or silently replace an active research question.

## Reuse source tools before building adapters

For infrastructure selection, inventory components already mentioned in project
notes, skills and the managed material catalog. A recorded review or card is a
lead, not a current full-source read. Follow the original repository and inspect
its implementation, setup requirements and known failures. Extend this pool only
when a named retrieval gap remains.

Keep three decisions separate: which component is useful, whether it is usable
on this host now, and whether its evidence was admitted downstream. Popularity,
installation and a doctor check do not collapse those decisions.

| Need | Reuse first | Boundary to verify |
| --- | --- | --- |
| GitHub source, issues, versions | Existing GitHub connector or `gh`; pin code citations to inspected commits | Public access does not prove private-repository scope; do not duplicate the GitHub client |
| Signed-in X or other dynamic pages | Existing native platform reader or installed Ego browser | Read visible authorized content and preserve TaskSpace ownership; add another browser bridge only for a demonstrated missing operation |
| Public URL to text | Jina Reader as an optional extraction path | Test the actual URL; preserve original provenance, wall/truncation status and content digest; HTTP success alone is insufficient |
| Additional broad web recall | Exa hosted MCP through an existing MCP client, such as mcporter | Inspect live schemas and run a real bounded query; anonymous access has limits and does not enable authenticated Exa Agent |
| Video subtitles / feeds / papers | Existing `yt-dlp`, feed reader, paper explorer and source-specific skills | Inspect available subtitles and supported source APIs; no general browser replacement is needed |
| Repeated site operations | OpenCLI adapters; Browser Harness or Browserbase Skills when their runtime fits | Verify browser bridge/account setup and the concrete adapter before installing another control stack |
| Batch website extraction | Crawl4AI or Firecrawl when volume justifies it | Separate self-hosted library from hosted search/API features, credentials, maintenance and licence scope |

[Agent-Reach's component map](https://github.com/Panniantong/Agent-Reach/blob/94f06c1969dfc1834001269d79d3ad0972d9dee6/README.md)
is useful for upstream selection. Its role is installation/diagnosis/routing;
the agent calls upstream readers directly. Its
[web check at this commit](https://github.com/Panniantong/Agent-Reach/blob/94f06c1969dfc1834001269d79d3ad0972d9dee6/agent_reach/channels/web.py)
returns `ok` without a network probe, so do not translate that check into a
LoopX ready provider. Reuse an already-working upstream tool where possible.

[Jina's MCP tool table](https://github.com/jina-ai/MCP/blob/5d6eb191a75d8e67b6e01ce427f0cc5c05c800aa/README.md#available-tools)
distinguishes optional-key URL reading from key-required search and PDF-layout
extraction. [Exa's MCP guide](https://github.com/exa-labs/exa-mcp-server/blob/e9c3b0126a3373eb1aeaca162e84d0791ff4c7f7/README.md#authentication)
documents anonymous search/fetch with rate limits. These are dated source
references, not permanent promises of availability; prefer current live schema
and invocation results over old command examples.

Do not install an entire toolkit or copy login cookies merely because a channel
is listed. Choose the smallest component that adds real coverage within current
authorization. Keep raw observations and host configuration private; public
guidance records reusable boundaries rather than account context or run dumps.

## LoopX deep research

When the user invokes `loopx-deepresearch`, or this skill is used for an expedition
inside an existing LoopX research run, read the available `loopx-deepresearch`
skill and follow its current CLI packet. Do not start a LoopX run merely because
the user asks for ordinary web research or asks to create this skill.

- Read status before selecting the expedition. Respect `research_contract`,
  `next_expedition`, `stop_conditions` and the packet's `evidence_commands`.
- Use external-research to formulate queries, inspect originals, distinguish
  claims from observations and decide whether another source removes a real gap.
- Record durable sources/findings through the current typed evidence commands.
  Map observation strength, source-family dependence and uncertainty into the
  supported fields; do not invent CLI flags or a second persisted schema. Use
  help or the packet for exact syntax. Do not write canonical state files directly.
- Resolve subquestions only under the research owner's evidence rules. Re-read
  status after the expedition; stop/report when its contract says to do so.
- If the user changes scope, reconcile it through that owner's supported flow;
  do not quietly close another active run or keep searching the superseded target.

The installed LoopX command skill may be generated. Do not patch a managed copy
as the integration mechanism: use this companion skill during retrieval. Changes
to the generated entrypoint belong in its canonical owner and normal repository
workflow, only when such a change is part of the task.

## LoopX external evidence providers

When the user explicitly asks to use LoopX research methods or connectors, use
the existing `external-evidence` capability. Consult current CLI help and the
[capability-owned guide](https://github.com/loopx-project/loopx/blob/main/loopx/capabilities/external_research/README.md)
for schemas and commands; do not copy a generated LoopX skill or add a second
research ledger.

- Distinguish a host research **method** from a concrete tool **connector**.
  Connector registry presence and an upstream doctor result are discovery
  evidence, not readiness. Observe a bounded real read or search on this host;
  inventory records installation, enablement, readiness and unavailable reasons
  without granting credentials or authority.
- Create the native plan before the selected execution. Invoke the existing
  source tool under its own permission boundary. The CLI's bundled `execute`
  path is for the public GitHub method; it does not dispatch arbitrary host
  tools. Other methods/connectors return a caller-authored receipt bound to
  the exact plan, request, provider id and kind.
- Preserve original URLs, source families, observation strength, dates,
  inspected-content digests and read limitations. A rendered or extracted
  body is not proof that linked sources, images or an entire article were read.
  Failed/no-evidence receipts carry no sources; a failure on one optional
  reader does not invalidate another successfully inspected source.
- Run native receipt validation, then explicitly admit/reject source refs.
  For durable projection use the existing deepresearch ledger with the exact
  plan objective as its question, and inspect `readback` for blockers and
  matching source lineage. Retire admitted evidence only after actual downstream
  coverage. Caller receipts are not authenticated execution attestations;
  retirement is not a research-completeness or product-qualification claim.

Retain the caller's registry/runtime and current Turn identity. Neither a new
provider nor a corrected research target authorizes a hidden Goal, another Turn,
fresh quota accounting or a replacement for an already-bound Todo.

## Other research tasks

Reuse the task's existing outline, source table, RFC or report. Keep one compact
claim-to-source map where it helps auditing; do not require a database, new goal,
subagents or a published artifact for a short comparison. If the orchestrator is
unavailable, external retrieval can still be useful, but do not claim that its
state, report or qualification was updated.
