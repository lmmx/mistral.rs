# 06 -- Batch-shape measurement: BLOCKED by execution environment (2026-09-10, fourth session)

Status: **blocked, not run.** This session did not execute Plan 06's batch-shape sweep and
produced no ON/OFF, per-dimension, or per-workload numbers. This report documents why, what
evidence already exists, and exactly what is required before a retry can proceed.

No Rust source touched. No rebuild. No server restarted. `/workspace` remained on
`grammar-fast-forward` at `cab8cd3ae` throughout; this report was written from a separate
`git worktree` checkout of `ff-demo-artifacts` (verified not to affect `/workspace`'s branch or
status -- see "Git/worktree state" below), so as not to risk the source branch.

## What Plan 06 actually requires

`06-batch-shape-measurement.md` specifies a sweep of N in `{1, 2, 4, 8}` across three workloads
(A identical schemas, B differing schemas, C ~50% mixed constrained/unconstrained), **each run
under two separate server processes** -- one launched with `MISTRALRS_GRAMMAR_FAST_FORWARD=1`, one
with it unset -- because the flag is read once through a `OnceLock` at pipeline load
(`perf_flags.rs:33-35`, restated in `05-harness.md`). It reports six dimensions per combination:
splice discard rate by reason, forced-tokens-lost, a batch-composition histogram (the one
instrumentation addition the plan permits), splice-width/min-width distribution, end-to-end
tokens/s and forward-pass count flag-on-vs-off, and qualitative complexity/risk notes. The
deliverable is `plans/ff-round-two/reports/06-batch-shape.md` with numbers only, no ranking, no
recommendation, per the plan's "hand back" instruction and the README's "prove/disprove" class
rule (README.md: "Gather evidence; change nothing on the strength of a suspicion" / "Report,
don't decide").

That "hand back" bar is not met by a partial run. Dimension 5 (flag on vs off) is not an optional
extra -- it is one of the six required dimensions, and it is the one dimension no HTTP-only access
to a single already-running process can ever produce, by the plan's own stated constraint. This
session's blocking conclusion therefore matches Plan 06's stated criteria: the plan needs a true
ON/OFF contrast and a histogram addition, and this container can supply neither.

## Why the currently-running ON server is not sufficient

The host server (`unsloth/Qwen3.5-4B-GGUF`, CUDA, quant 4, paged attention, max batch 8, max seqs
8, port 1234) is still running from the prior sessions' launch, with `MISTRALRS_GRAMMAR_FAST_FORWARD=1`
read once at load. Confirmed live from this container:

```
curl -s http://host.docker.internal:1234/health
200

curl -s http://host.docker.internal:1234/metrics | grep grammar_ff
mistralrs_grammar_ff_tokens_fed_total 12
mistralrs_grammar_ff_attempts_total{outcome="matcher_error"} 0
mistralrs_grammar_ff_attempts_total{outcome="empty_splice"} 95
mistralrs_grammar_ff_attempts_total{outcome="unsupported"} 0
mistralrs_grammar_ff_attempts_total{outcome="grammar_stopped"} 6
mistralrs_grammar_ff_attempts_total{outcome="staged"} 9
mistralrs_grammar_ff_splices_staged_total 9
mistralrs_grammar_ff_support_total{supported="true",reason="enabled"} 1
```

This is one process, permanently fixed at flag-on for its lifetime. No request, header, or
harness flag sent to it over HTTP can change what value its `OnceLock` resolved to at load time.
Comparing "this ON process, busy" against "this ON process, idle" is not an ON/OFF comparison --
it would only ever measure request-to-request noise on a single flag setting, which is exactly the
outcome `05-harness.md` and the plan-05 checkpoint (`05-harness-live-validation-checkpoint.md`)
already warned against manufacturing. This session did not attempt it.

A genuine dimension-5 measurement needs a second, independently-launched process with the
environment variable unset (or `=0`), matched on every other configuration: same model, same
quant, same paged-attention settings, same `--max-batch-size`/`--max-seqs`, same port-equivalent
setup, same request/schema/generation parameters. Producing that second process requires control
over the host's process lifecycle.

## What this container cannot do

- **No GPU passthrough.** `nvidia-smi` is unavailable; CUDA inference cannot run inside this
  container regardless of the flag question.
- **No host process-control mechanism.** The CUDA server lives on the host, not this container.
  This container has HTTP reachability to it (`host.docker.internal:1234`) and nothing else -- no
  `docker`, no `ssh`, no shared process manager, no way to start, stop, or restart it, and no way
  to launch a second, independently-configured instance alongside it.
- **No permitted source change.** Dimension 3 (batch-composition histogram) is the one
  instrumentation addition `06-batch-shape-measurement.md` explicitly authorizes, but this
  session was instructed not to modify Rust source under any circumstance, so that addition
  cannot be made even though the plan would otherwise allow it. This blocks dimension 3
  independent of the ON/OFF question, and would block it even if a second process were available.

Given these three, the plan's required experiment -- two matched processes, one instrumentation
change -- is not executable from inside this container in its current form, regardless of how the
HTTP-reachable single process is queried.

## What evidence already exists (do not re-derive, reuse)

- `05-harness-live-validation-checkpoint.md`: `ff_harness.py`'s `concurrency` mode, pointed at
  this same live ON process via `--server-host`/`--server-port`, reproducibly distinguishes real
  FF execution (constrained request: 3 splices staged, 4 tokens fed, twice, matching) from a
  no-FF-activity control (unconstrained request: zero `mistralrs_grammar_ff_*` movement). That
  checkpoint already establishes the harness can drive this server correctly; nothing about the
  HTTP path itself is in question.
- `06-ff-activation-confirmed.md`: all three of Plan 06's FF-activation precondition gates
  (capability, reachability/classification, measurability) are CONFIRMED against this same live
  process. Dimensions 1, 2, and 4 have a denominator whenever a properly sized sweep can actually
  be run.
- `06-ff-observability-audit.md`: traces why the pre-existing counters could not distinguish the
  five staging states, and is the source of the current `mistralrs_grammar_ff_attempts_total`
  outcome labels and the (source-level, not yet exercised) `splice_drops_total{reason=...}` /
  `tokens_dropped_total{reason=...}` counters added for plan 02.
- Counters `mistralrs_grammar_ff_splice_drops_total` and `mistralrs_grammar_ff_tokens_dropped_total`
  exist in `mistralrs-core/src/sequence.rs` (grep-confirmed on `grammar-fast-forward` at
  `cab8cd3ae`) but do not yet appear in this live server's `/metrics` output -- consistent with the
  `metrics` crate's lazy registration and with zero batch-shape discards having occurred yet at
  the request volumes run so far (`splices_staged_total=9`, all apparently homogeneous batches).
  This is not evidence the counters are broken; it is evidence no batch-shape-mismatched batch has
  been observed under the traffic sent to date. A proper N-swept, differing-schema workload (B, N
  >= 2) is what would actually exercise this path, and that is part of what remains unrun.
- None of the above constitutes a dimension-5 measurement or a dimension-3 histogram; both are
  explicitly out of scope of every report listed here, by their own text.

## What is required to unblock Plan 06

In order, the minimum needed for a retry:

1. **Host-side (or equivalently privileged) process control**, sufficient to launch a second
   `mistralrs serve` instance identically configured to the current one except
   `MISTRALRS_GRAMMAR_FAST_FORWARD` unset -- either from a session with a shell on the host itself,
   or from a container granted a process-control channel to it (not just HTTP reachability). Two
   concrete forms this could take: (a) an interactive/agent session run directly on the host
   machine, given the same build (`grammar-fast-forward` at `cab8cd3ae`, `target/debug/mistralrs`,
   confirmed already built there per `06-ff-activation-confirmed.md`), or (b) this container
   upgraded with a control surface (docker/ssh/process manager) reaching the host.
2. **Explicit authorization to add the dimension-3 batch-composition histogram**, the one source
   change `06-batch-shape-measurement.md` permits. Without it, dimensions 1/2/4/6 can be measured
   but dimension 3 cannot, and the deliverable would still be incomplete against the plan's own
   six-dimension list.
3. Once both are available: run the sweep exactly as specified -- N in `{1, 2, 4, 8}`, workloads
   A/B/C, `ff_harness.py --mode concurrency`, flag on and flag off as two separately-launched
   processes, `/metrics` scraped before and after each run, deltas (not absolute values) recorded,
   staggered start times in at least one variant per the plan's requirement that synchronized
   starts are unrealistically favorable.

No partial or ON-only sweep was run in place of the above (per this session's explicit
instruction not to), since an ON-only sweep, however large, cannot answer dimension 5 and would
risk being read as more conclusive than it is.

## Git/worktree state

- `/workspace`: branch `grammar-fast-forward`, HEAD `cab8cd3ae`, working tree clean, unchanged
  throughout this session. Never checked out `ff-demo-artifacts` there.
- This report was written and (if committed) will be committed from a separate `git worktree`
  checkout at `/tmp/ff-artifacts-wt`, branch `ff-demo-artifacts`, HEAD `1abbfc96f` prior to this
  commit. Verified independent of `/workspace` before use: adding the worktree left
  `/workspace`'s branch, HEAD, and `git status --short` unchanged.
- No Rust source modified. No rebuild performed. No server process started, stopped, or
  restarted by this session.

## Files added

- `plans/ff-round-two/reports/06-batch-shape-blocked.md` (this report)
