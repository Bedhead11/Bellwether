# Design Decision #3 — Drift vs. Intended-Change Disambiguation

> Status: **Decided** (design). The fingerprint field lands in the `AgentRun` schema in
> Phase 0; the quarantine/accept-new-normal flow lands with the Baseline Manager (Phase 1).

## Problem

A deliberate prompt edit, model swap, or config change shifts the behavioral distribution —
and must **not** be reported as drift. Naively, every release fires every alert → alert
fatigue → muted tool → dead project. We must distinguish *intended change* (re-baseline)
from *drift* (alert) on streaming, non-deterministic signals, with a low false-positive rate.

## Options considered

### Option A — Pure statistical change-point detection
Detect step-changes (ADWIN / Page-Hinkley / BOCPD); treat all as drift.
- **Pros:** zero integration work; framework-agnostic.
- **Cons:** *cannot* tell intended from unintended — a deploy looks identical to drift.
  Guarantees alert fatigue. Rejected.

### Option C — Fully automatic auto-rebaseline
Any sustained step-change → silently adopt as the new normal.
- **Pros:** zero alerts on deploys.
- **Cons:** **masks real drift as "new normal."** A genuine slow degradation that crosses
  into a step gets absorbed. Dangerous; inverts the product's purpose. Rejected.

### Option B — Fingerprint + deploy markers + quarantine + change-point correlation + human accept-new-normal (RECOMMENDED)
Layered: use *declared* change signals where available, fall back to *correlation* where not,
and keep a human-confirmable, fully-audited "accept new normal" path.

## Recommendation — Option B

### 1. Config fingerprint (carried on every `AgentRun`)
Every run carries a **config fingerprint**: a stable hash over the things whose change is
*intended* to change behavior —

- model id / provider / model version string,
- prompt-template version (hash or explicit version),
- tool schema set (names + arg schemas),
- sampling params (temperature, top_p, max_tokens),
- framework + key library versions,
- declared `agent_version` / deploy id if the user provides one.

Baselines are keyed by **(agent, task_class, config_fingerprint)**. A new fingerprint is, by
definition, a candidate intended change — it gets its own baseline lineage rather than
polluting the old one.

### 2. Deploy markers (explicit, optional but encouraged)
The SDK exposes `bellwether.mark_deploy(version=..., note=...)`. An explicit marker is the
strongest signal: it timestamps an intended change and carries human rationale into the audit
log. When present, it removes ambiguity entirely.

### 3. Quarantine / learning window
On a **new fingerprint** OR an **explicit deploy marker**, open a quarantine window for that
(agent, task_class):

- Detectors run in **observe mode** — they compute scores but emit alerts tagged
  `post-change · unconfirmed`, *not* fired as drift.
- A fresh baseline lineage accumulates from the new runs.
- The window closes after enough runs to establish the new baseline (cold-start rules from
  `04`) or a max duration.

This prevents the release-fires-every-alert failure while still *recording* everything for
audit and for the case below.

### 4. Change-point correlation (the discriminator)
A change-point detector runs continuously per feature. When it flags a sustained step-change:

- **Step-change aligns (within window) with a fingerprint change or deploy marker**
  → label **intended change**. Propose a new baseline version; route to accept-new-normal.
- **Step-change with NO corresponding fingerprint change / marker**
  → label **drift**. Fire the alert. (This is the case that matters — silent degradation.)

The correlation window is asymmetric and conservative: unexplained step-changes default to
**drift**, because a missed deploy marker (false alarm, annoying) is far cheaper than a
missed real degradation (the product failing at its one job).

### 5. Human-confirmable "accept new normal" (audited)
When a change is labeled intended, the new baseline is *proposed*, not silently adopted. A
human (or an explicit auto-accept policy for declared deploys) confirms "accept new normal,"
which:

- promotes the new baseline lineage to active,
- writes a full audit record: before/after baseline summary, the fingerprint diff, the
  deploy marker/rationale, and who/what accepted it,
- clears the `post-change · unconfirmed` tags.

This is also the L3 metamorphic invariant "re-baseline after declared change clears the
alert" (`02`).

## Decision table

| Step-change? | Fingerprint changed / deploy marker? | Action |
|---|---|---|
| no | — | normal monitoring |
| yes | yes (aligned) | intended change → quarantine → propose new baseline → accept-new-normal |
| yes | no | **drift → alert** |
| no | yes | new lineage opens; observe until baseline established |

## Riskiest unknown

**Silent upstream config changes the fingerprint can't see.** A provider can change the model
behind a stable name (`gpt-4o`, `claude-…`) with no version string change, so the fingerprint
is identical but behavior shifts — this looks exactly like drift and *should* alert, which is
arguably correct, but it muddies "intended vs. unintended." Mitigations:

1. Treat unexplained step-changes conservatively as drift (already the default) — so we err
   toward flagging, which is the safe direction.
2. Optional **behavioral canary probes**: periodically send a fixed probe input and
   fingerprint the *output behavior*, catching silent model swaps that the static config
   fingerprint misses.
3. Surface the ambiguity in the triage narrative ("step-change with no declared config
   change — possible silent upstream model update or genuine drift") rather than asserting a
   cause we can't prove.
