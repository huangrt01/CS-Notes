---
name: browser
description: "Browser automation. When you need to visit a webpage, open a website, take a screenshot, or perform browser automation, **you MUST read this SKILL.md first** — contains required parameters and operating procedures. **Calling the browser tool without reading may result in failed operations or unexpected behavior.**"
metadata:
  {
    "openclaw":
      {
        "emoji": "🌐",
        "requires": { "config": ["browser.enabled"] }
      }
  }
---

# Browser Tool Guide

The browser tool controls the default browser inside the instance via CDP.
All calls use `action` as the top-level discriminator; interaction goes through `action: "act"` with a `request` object.

**Profile:** Use `"chromebrowser"` unless the user specifies otherwise.

---

## ⚠️ CRITICAL: Human-in-the-loop & `<browser-handoff />`

**STOP AND HAND OFF** when any step requires human judgment: login, CAPTCHA, QR code, payment, SMS/email verification, cookie consent, confirmation prompts, or when the user explicitly asks to take over.

**You MUST include `<browser-handoff />` at the very end of your reply.** The system uses this tag to unlock browser control for the user — without it, the browser remains locked to the agent. Then take a screenshot (it will render automatically), explain what the user needs to do, and **WAIT** for confirmation.

```
The page requires login. Please complete it and let me know when done.

<browser-handoff />
```

Never automate credentials or bypass verification. If in doubt, emit the tag — a false positive is harmless; a missing tag blocks the user.

---

## Core Loop

**⚠️ Serial calls only. NEVER call browser tools in parallel.** Each call depends on the previous result.

Basic sequence: `open` → `snapshot` → `act` → *(snapshot if needed)* → `act` → …

- **After `open` → snapshot immediately.**
- **After `snapshot` → act on the refs you got.** Don't re-snapshot unless the result was empty.
- **After `act` → snapshot only if the page may have changed** (navigation, submit, dynamic load).
- **Snapshot mode:** default returns full page. Use `interactive: true` for targeted operations once layout is clear, or `compact: true` on large pages. Never use `interactive: true` on first load.
- **Screenshot** only on task completion or user request. Use `screenshot` for visual interpretation when DOM alone is ambiguous.

---

## Discipline Rules

- **`<browser-handoff />` is mandatory on any handoff.** Login, CAPTCHA, payment, verification → emit the tag at the end of your reply. No tag = user is blocked.
- **Tab cleanup on task completion.** See section below.
- **One call at a time.** Read each response before the next.
- **Snapshot → Act.** Get refs from snapshot before acting. Re-snapshot only after navigation, submit, or dynamic DOM change — not between consecutive actions on a stable page.
- **Screenshot only on task completion or user request.** DOM parsing covers page state during the task.
- **2-error abort.** Two consecutive failures → stop and report.
- **No repeats.** Don't re-open an already-open URL. Don't retry without reading the error.
- **Manual login only.** Never automate credential entry.
- **Type appends.** To replace: `Control+a` first, then `type`. Use `fill` for batch fields.
- **Required fields first.** `*` means required — fill before optional fields.
- **SSO redirects.** Don't loop on logout/refresh. Hand off.
- **No fabricated queries.** Always trigger searches through page interactions.

---

## ⚠️ Tab Cleanup on Task Completion

**When to check:** call `{ "action": "tabs" }` when the task finishes (best-effort — if it fails, skip and finish normally). **Especially important after long-running tasks (5+ browser actions)** where tabs are likely to have accumulated.

| Tabs open | Action |
|-----------|--------|
| ≤ 5 | Silent. No mention needed. |
| 6 – 10 | Brief note: _"Opened N tabs — want me to close them?"_ |
| 11 – 19 | Recommend: _"N tabs open, suggest closing task tabs to free resources."_ |
| ≥ 20 | **⚠️ Warn:** _"N tabs open — will cause slowdown/crashes. Close task tabs?"_ |

**Rules:**
- Only close tabs opened during the current task. If unsure which are yours, list candidates and ask.
- If user has set auto-cleanup preference ("以后直接关" / "always close when done"), close without asking and confirm: _"Closed N tabs."_

---

## Last-resort actions

Only when the core loop has failed.

- **`wait`** — use only if snapshot comes back empty.
- **`evaluate`** — only for viewport scrolling or custom JS with no alternative.
- **`status` / `start`** — only on explicit browser error.

---

## Cookbook

### Open & inspect

```
{ "action": "open", "url": "https://example.com" }
{ "action": "snapshot" }
```

### Login-gated sites (大众点评, 知乎, etc.)

1. Prompt login first — hand off with `<browser-handoff />`.
2. If user declines, attempt as unauthenticated.
3. If blocked, prompt login again.

### Click / type / fill

```
{ "action": "snapshot", "interactive": true }
// button "Submit" [ref=e5]
{ "action": "act", "request": { "kind": "click", "ref": "e5" }}

// textbox "Search" [ref=e3] — type appends; Control+a to clear first
{ "action": "act", "request": { "kind": "type", "ref": "e3", "text": "OpenClaw", "submit": true }}

// batch fill
{ "action": "act", "request": { "kind": "fill", "fields": [
  { "ref": "e1", "value": "Ada Lovelace" },
  { "ref": "e2", "value": "[email protected]" }
]}}
```

### Other interactions

```
{ "action": "act", "request": { "kind": "click", "ref": "e1", "doubleClick": true }}
{ "action": "act", "request": { "kind": "click", "ref": "e1", "button": "right" }}
{ "action": "act", "request": { "kind": "select", "ref": "e9", "values": ["OptionA"] }}
{ "action": "act", "request": { "kind": "press", "key": "Control+a" }}
{ "action": "act", "request": { "kind": "hover", "ref": "e4" }}
{ "action": "act", "request": { "kind": "drag", "startRef": "e10", "endRef": "e11" }}
{ "action": "act", "request": { "kind": "resize", "width": 375, "height": 812 }}
```

### Screenshot

```
{ "action": "screenshot" }                              // viewport
{ "action": "screenshot", "fullPage": true }             // full page
{ "action": "screenshot", "ref": "12" }                  // element by ref
{ "action": "screenshot", "element": ".hero" }           // element by CSS
```

⚠️ Always use `MEDIA:<path>` to show screenshots to the user — don't just describe them.

```
MEDIA:/tmp/openclaw/screenshots/page_17100.png
```

### Dialog / Upload / PDF

```
// Confirm dialog — arm first, then trigger
{ "action": "dialog", "accept": true }
{ "action": "act", "request": { "kind": "click", "ref": "e8" }}

// Upload — files must be in /tmp/openclaw/uploads/
{ "action": "upload", "paths": ["/tmp/openclaw/uploads/report.pdf"], "inputRef": "e5" }

// Export page as PDF
{ "action": "pdf" }
```

---

## Advanced: Tab Management

For multi-tab workflows only.

```
{ "action": "tabs" }                                    // list tabs
{ "action": "open", "url": "https://other.com" }        // new tab
{ "action": "focus", "targetId": "ABCD1234" }           // switch tab
{ "action": "close", "targetId": "ABCD1234" }           // close tab
```

---

## Troubleshooting

```
// Snapshot empty → wait then retry
{ "action": "act", "request": { "kind": "wait", "loadState": "networkidle" }}
{ "action": "snapshot", "interactive": true }

// Browser not responding
{ "action": "status" }
{ "action": "start" }

// Check JS errors
{ "action": "console", "level": "error" }
```
