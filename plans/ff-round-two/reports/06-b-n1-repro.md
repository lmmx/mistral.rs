# 06 follow-up -- focused reproduction of the B/differing/N=1 ON-vs-OFF divergence (2026-09-10, sixth session)

Status: **reproducible and FF-specific. Evidence points to an actual fast-forward correctness bug
(item 5), not batch-shape, not ordinary sampling noise, not a harness/request-construction or
server-state artifact.** This is the human-checkpoint follow-up to the anomaly flagged in
`06-batch-shape.md`'s "Correctness / equality" section: workload B (differing schemas), N=1,
`nested.schema.json`, seed 42.

No Rust source touched. One minimal, additive change to `ff_harness.py` (diff below) to capture
full response text and `usage`, which the `concurrency` mode's report previously omitted. No
rebuild. `/workspace` stayed on `grammar-fast-forward` at `cab8cd3ae` throughout; this report, the
harness change, and the raw data were produced and committed from the separate `git worktree` at
`/tmp/ff-artifacts-wt` (branch `ff-demo-artifacts`).

## What the committed Plan 06 report and raw JSON already showed, re-inspected first

Before running anything new, `06-batch-shape.md` and its two raw JSON files
(`06-concurrency-on-on-B-differing-n1-*.json`, `06-concurrency-off-off-B-differing-n1-*.json`) were
re-read. Confirmed directly from the JSON (not re-derived from the earlier summary):

- Identical request construction on both sides: same `prompt`, same `schema_files` list, same
  `schema_index=2` (`nested.schema.json`), same `seed=42`, same `max_tokens=64`,
  `stagger_seconds=0.0`, `num_requests=1`. Rules out harness/request-construction difference
  (investigation item 3) for this pair directly -- the bodies sent were byte-identical modulo the
  flag.
- `metrics_after` in the ON report already showed `mistralrs_grammar_ff_attempts_total{outcome=
  "staged"}` and `splices_staged_total` both non-zero for this run and `tokens_fed_total: 8`
  (cumulative across the whole ON sweep at that point; the *delta* for this one request, computed
  against `metrics_before`, was staged=1, fed=1 -- reported correctly in `06-batch-shape.md`).
- Neither JSON file contained the response text or `usage` block -- `ff_harness.py`'s
  `concurrency` mode only ever captured `finish_reason` and latency, not content. This is the gap
  the harness change below closes; nothing about the earlier finding was mis-recorded, it was just
  incomplete for root-causing.

## Harness change (smallest necessary, artifact-side only)

`ff_harness.py`, `concurrency` mode only (`equality`/`routing-log`/`arms` modes untouched):
`ConcurrencyRequestResult` gained two fields (`content`, `usage`), populated from the existing
JSON response body's `choices[0].message.content` and top-level `usage` in `fire_one`, and
threaded into the report's `requests[]` entries. No new request is sent, no existing field
changed shape or meaning; purely additive.

```diff
--- a/ff_harness.py
+++ b/ff_harness.py
@@ class ConcurrencyRequestResult:
     finish_reason: str | None = None
     error: str | None = None
+    content: str | None = None
+    usage: dict[str, Any] | None = None
@@ def fire_one(
-        finish_reason = payload.get("choices", [{}])[0].get("finish_reason")
+        choice = payload.get("choices", [{}])[0]
+        finish_reason = choice.get("finish_reason")
+        content = choice.get("message", {}).get("content")
+        usage = payload.get("usage")
         return ConcurrencyRequestResult(
-            ok=True, status=resp.status, latency_s=latency, finish_reason=finish_reason, **common
+            ok=True, status=resp.status, latency_s=latency, finish_reason=finish_reason,
+            content=content, usage=usage, **common
         )
@@ report["requests"] entries
                 "finish_reason": r.finish_reason,
                 "error": r.error,
+                "content": r.content,
+                "usage": r.usage,
```

## Experiment design

Smallest experiment that can distinguish the five candidate explanations: repeat the *exact same*
single-request invocation (workload B, differing schema set, N=1, seed 42 -- which deterministically
selects `nested.schema.json` via the harness's seeded shuffle) five times against a fresh FF-OFF
process, then five times against a fresh FF-ON process, capturing full content and `usage` each
time.

- **Fresh restarts, both legs**, confirmed via `/metrics` immediately before sending any sweep
  traffic (all `mistralrs_grammar_ff_*` counters at 0, `mistralrs_grammar_ff_support_total` showing
  the expected `supported=false,reason=flag_disabled` / `supported=true,reason=enabled`). One
  restart claimed by the user turned out not to be a genuine new process (stale cumulative
  counters, ~1752 decode tokens matching the prior full sweep's total) -- caught before any
  measurement was taken, and a corrected fresh restart was confirmed before proceeding. This
  matters for investigation item 4 (server-state/configuration difference): every trial reported
  below ran against a verified-fresh process, so residual state from prior requests is ruled out
  as a factor in the results below.
- One small unconstrained warmup request (`max_tokens=16`, discarded) before each leg's 5 repeats,
  matching the original sweep's methodology.
- No full 24-cell sweep re-run. Only this one workload/N/schema cell, 5+5 repeats.

Command (identical across all 10 repeats except `--flag` and `--label`):
```
python3 ff_harness.py run --mode concurrency \
  --model-id unsloth/Qwen3.5-4B-GGUF --flag {on|off} \
  --server-cmd "sleep 300" \
  --server-host host.docker.internal --server-port 1234 \
  --startup-timeout-seconds 5 --max-tokens 64 --plan 06-b-n1-repro \
  --metric-prefix mistralrs_ --out-dir plans/ff-round-two/reports/06-b-n1-repro \
  --label {off|on}-rep{1..5} --num-requests 1 --schema-set differing \
  --unconstrained-fraction 0.0 --seed 42
```

## Results

**FF OFF, 5/5 repeats: fully reproducible, correct.**

| rep | finish_reason | completion_tokens | content |
|---|---|---|---|
| 1-5 (identical) | `stop` | 47 | `{"run": {"release_channel_name": "stable", "observed_error_count": 0}, "note": "The last deployment was successful."}` (formatted; exact bytes identical across all 5 reps) |

**FF ON, 5/5 repeats: fully reproducible, but broken.**

| rep | finish_reason | completion_tokens | content |
|---|---|---|---|
| 1-5 (identical) | `length` | 64 (hit `max_tokens`) | `{\n  "run":  \r  \r  \r  \r  \r  \r  \r  \r  \r  \r  \r  \r  \r  \r  \r  \r  \r  \r  \r ` (68 raw chars incl. escapes; exact bytes identical across all 5 reps) |

Every ON repeat: `mistralrs_grammar_ff_attempts_total{outcome="staged"}` delta = 1,
`splices_staged_total` delta = 1, `tokens_fed_total` delta = 1, `empty_splice` delta = 62, zero
`grammar_stopped`/`unsupported`/`matcher_error`. Every OFF repeat: all `grammar_ff_attempts_total`
outcomes 0 except `unsupported` (flag disabled), as expected.

FF ON does not merely produce a *different valid* completion -- it produces a degenerate one: after
emitting `{\n  "run": ` (the object open and the `"run"` key, matching the schema's first required
property), generation gets stuck emitting a `\r` (carriage return) token repeatedly, never
completing the `release_channel_name`/`observed_error_count` fields the schema requires, and never
reaching the JSON's closing braces, until `max_tokens` cuts it off. FF OFF produces a fully valid,
schema-conforming JSON object in fewer tokens than the budget allows. This happens after exactly
one splice is staged and one token fed -- consistent with the single forced-structure point
(the `"run":` key literal and/or its immediately following whitespace/brace) being where the two
paths diverge.

## Distinguishing the five candidate explanations

1. **Genuine FF-induced behavior/correctness divergence: SUPPORTED.** 5/5 reproducibility on both
   sides at N=1 (no batch-shape confound), byte-identical output within each flag setting, with the
   only varying input being the flag itself -- this is about as clean a signal as an HTTP-level
   experiment can produce.
2. **Ordinary sampling/nondeterminism: NOT SUPPORTED for this cell.** `temperature=0.0`, and both
   legs were perfectly self-consistent across 5 repeats each (no variance within a flag setting).
   Whatever caused the earlier N=8 same-schema instability noted in `06-batch-shape.md` (OFF's own
   index-0-vs-index-4 split within one concurrent run) is not present here -- that was a
   concurrency/batch-position effect, and this reproduction deliberately removes concurrency
   entirely (N=1, sequential repeats, no peer requests in flight).
3. **Harness/request-construction differences: RULED OUT.** Confirmed by direct inspection of both
   the original committed JSON and every new repeat's JSON: prompt, schema file, seed, and
   `max_tokens` are identical on both sides in every trial.
4. **Server-state/configuration differences: RULED OUT for the reported trials.** Both legs used
   freshly restarted, `/metrics`-verified processes, identical CLI configuration apart from the
   flag. (One false-fresh restart was caught before use -- see above -- and does not contaminate
   the reported results, which are all against confirmed-fresh processes.)
5. **An actual grammar fast-forward bug: the leading candidate given 1-4 above.** A single staged
   splice, immediately followed by degenerate repeated-token generation that never recovers within
   the token budget, on a schema whose first forced span is a nested-object key (`"run":`) rather
   than a flat string/enum, is consistent with the splice corrupting whatever context the model
   resumes free generation from at that point. This report does not open `mistralrs-core` source or
   propose a fix -- root-causing the specific mechanism (e.g., which token(s) the one splice fed,
   and why generation resumes on `\r`) needs a source-level look this session was told to avoid
   unless proven necessary. It is now evidenced as necessary for a fix, not for further
   measurement.

## Scope note

This reproduction targeted exactly the flagged cell (`nested.schema.json`, N=1). It does not
establish how common this failure mode is across the other three fixture schemas
(`narrow`/`partial`/`wide`) or other N, nor whether it is specific to `nested.schema.json`'s
structure (an object-valued property as the first required field) versus something schema-shape-
independent. That is future scoping work, not run here, per the instruction to keep this
investigation focused and not re-run the full sweep.

## Files added / changed

- `ff_harness.py`: additive `content`/`usage` capture in `concurrency` mode (diff above).
- `plans/ff-round-two/reports/06-b-n1-repro.md` (this report).
- `plans/ff-round-two/reports/06-b-n1-repro/06-b-n1-repro-concurrency-{on,off}-*-rep{1..5}-*.json`
  (10 raw harness reports).

## Git/worktree state

- `/workspace`: branch `grammar-fast-forward`, HEAD `cab8cd3ae`, clean, untouched this session.
- This report, the harness diff, and raw data were committed from the separate worktree at
  `/tmp/ff-artifacts-wt`, branch `ff-demo-artifacts`.
- Two host server restarts this session (OFF then ON), both user-controlled; this container never
  started, stopped, or had process control over either.

## Conclusion

**Reproducible and FF-specific.** 5/5 FF-OFF trials produced identical, schema-valid output;
5/5 FF-ON trials produced identical, degenerate output that never satisfies the schema and burns
the full token budget. The divergence cannot be attributed to batch-shape (N=1), ordinary sampling
variance (perfect within-flag reproducibility at temperature 0), harness/request differences
(byte-identical requests confirmed), or server state (both legs freshly restarted and verified).
The evidence points at an actual fast-forward defect specific to this schema's structure, most
plausibly triggered by the single splice staged immediately after the `"run":` key. This is a
correctness bug candidate, not a batch-shape/scheduling one, and is a distinct finding from the
rest of `06-batch-shape.md`'s (scheduling/discard-rate) results.

Not proceeding to Plan 07.
