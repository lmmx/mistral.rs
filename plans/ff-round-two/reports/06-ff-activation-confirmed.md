# 06 — FF-activation precondition: gates A/B/C confirmed on live CUDA run (2026-09-10, third session)

Status: **precondition discharged.** All three gates from `06-batch-shape-measurement.md`'s
"FF-activation precondition" section passed against a live Qwen3.5-4B GGUF CUDA server built from
`grammar-fast-forward` at `c4fcb2d74` (which also carries a source fix landed this session: the
multimodal pipeline's `GeneralMetadata` now routes `supports_grammar_fast_forward` through
`ff_metrics::resolve_support(false, false)` instead of a hardcoded `false`, since Qwen3.5-4B loads
through `MultimodalLoader`).

This session ran the CUDA server (a capability the prior two sessions in this branch's history did
not have — see `06-ff-activation-precondition.md` for the container that had neither a GPU nor a
Rust toolchain), sent one real constrained HTTP request, and scraped `/metrics`.

## Command and startup evidence

```
MISTRALRS_GRAMMAR_FAST_FORWARD=1 target/debug/mistralrs serve \
  --model-id unsloth/Qwen3.5-4B-GGUF --quant 4 --paged-attn on \
  --max-batch-size 8 --max-seqs 8 --no-ui --port 1234 -v
```

Startup log:

```
grammar fast-forward support resolved for this pipeline supported=true reason="enabled"
Pipeline input modalities are [📝 Text, 🖼️ Vision, 🎬 Video]
```

## Request

One constrained request with a JSON-schema grammar was sent to
`http://127.0.0.1:1234/v1/chat/completions`. It completed normally:

```
finish_reason="stop"  completion_tokens=32  avg_compl_tok_per_sec=222.22223
```

## `/metrics` after the request

```
mistralrs_grammar_ff_support_total{supported="true",reason="enabled"} 1

mistralrs_grammar_ff_attempts_total{outcome="empty_splice"} 23
mistralrs_grammar_ff_attempts_total{outcome="staged"} 3
mistralrs_grammar_ff_attempts_total{outcome="grammar_stopped"} 2
mistralrs_grammar_ff_attempts_total{outcome="unsupported"} 0
mistralrs_grammar_ff_attempts_total{outcome="matcher_error"} 0

mistralrs_grammar_ff_splices_staged_total 3
mistralrs_grammar_ff_tokens_fed_total 4
```

## Gate-by-gate

- **Gate A — capability: CONFIRMED.** `ff_support_total{supported="true",reason="enabled"} = 1`.
  The multimodal-pipeline fix (routing through `resolve_support` instead of a literal `false`) is
  what let Qwen3.5-4B report `enabled` at all; before that fix this pipeline would have recorded
  `pipeline_unsupported` (or, more precisely, never resolved to `true`, since the literal `false`
  bypassed the resolver entirely).
- **Gate B — reachability and outcome distribution: CONFIRMED.** 28 total constrained-decode-step
  attempts (23 + 3 + 2 + 0 + 0), all classified, none `unsupported`, none `matcher_error`. The
  staging path was reached repeatedly, not just once.
- **Gate C — non-zero staged splices: CONFIRMED.** `splices_staged_total = 3`,
  `tokens_fed_total = 4`.

## What this does and does not establish

**Established:** the live Qwen3.5-4B GGUF CUDA pipeline resolves FF support to `enabled`, a real
constrained HTTP request reaches the FF attempt path repeatedly, 3 of 28 attempts staged a splice,
those splices fed 4 tokens total, 23 attempts found no forceable bytes at that grammar position
(`empty_splice`, not a defect per `06-ff-observability-audit.md` §2.2), 2 attempts hit an
already-finished grammar (`grammar_stopped`), and zero attempts were unsupported or hit a matcher
error. This is a single-request smoke test, not a sweep: 28 decode steps is not a statistically
powered sample, and no flag-off comparison was run in this session.

**Not established, and explicitly out of scope for this checkpoint:** any quantitative speedup
attributable to fast-forward. Nothing here measures tokens/s or forward-pass count with the flag on
vs off, nothing here characterizes splice-width distribution or batch composition (Plan 06
dimensions 1-4), and nothing here attributes any prior benchmark result (e.g. the pytest speedup
noted elsewhere in this branch's history) to fast-forward, PagedAttention, batching, or any other
mechanism. That attribution and quantification is exactly what Plan 06's batch-shape sweep and
dimension 5 (tokens/s, flag on vs off) are for, and they have not been run.

## Precondition status for Plan 06

**Discharged.** All three gates in `06-batch-shape-measurement.md`'s FF-activation precondition
section passed. Plan 06's dimensions 1, 2 and 4 (splice discard rate, forced-tokens-lost, and
splice-width/min-width distribution) now have a denominator and may proceed to a properly sized
sweep. Dimension 5 (tokens/s and forward passes, flag on vs off) was already measurable regardless
of these gates and remains unmeasured by this session.

## Files added

- `plans/ff-round-two/reports/06-ff-activation-confirmed.md` (this report)

## Source changes this session

- `mistralrs-core/src/pipeline/multimodal.rs`: `GeneralMetadata.supports_grammar_fast_forward` for
  `MultimodalLoader` now reads `crate::pipeline::ff_metrics::resolve_support(false, false)` instead
  of the literal `false`. This is what let Qwen3.5-4B (loaded via the multimodal pipeline) resolve
  FF support at all; no other pipeline's gating changed.
