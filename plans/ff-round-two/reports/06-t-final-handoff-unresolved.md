# 06-T -- final handoff: grammar fast-forward / Qwen3.5 GDN nested-schema bug, UNRESOLVED
(2026-09-11, twenty-fourth and final session of this investigation arc)

Status: **investigation closed by the user without a fix. The bug is real, reproducible, and
root-caused only down to an architectural boundary, not to a specific line of code.** This report is
the terminal handoff for `plans/ff-round-two/` Plan 06. No further fix should be attempted from this
document alone without first re-reading Sec 3-6 below; do not re-run the investigations this arc already
completed (Sec 3) without new evidence that specifically contradicts them.

## 1. The unresolved bug -- executive summary

- Grammar fast-forward (FF), when ON, produces **incorrect constrained-decoding output** for a specific
  nested JSON schema on `unsloth/Qwen3.5-4B-GGUF`.
- The failure is **deterministic and reproducible**: identical, byte-for-byte, across fresh-restart
  repeats at `temperature=0`.
- **FF OFF produces valid, schema-conformant JSON** for the identical request.
- **FF ON produces a `\r` (carriage-return) loop** inside the nested `"run"` object, continuing until
  `max_tokens` is exhausted, terminating with `finish_reason="length"` rather than `"stop"`.
- **The bug remains unresolved.** No fix was implemented. The root cause was narrowed to an
  architectural boundary (Sec 4, Sec 6) but never pinned to a specific defective line, kernel, or
  mechanism. A CUDA-graph-invalidation mitigation was designed (last session) but its premise was
  disproven by a live test before it was implemented (Sec 3) -- it is explicitly **not** a candidate fix
  going forward.

## 2. Exact reproduction

```bash
curl -s http://localhost:1234/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "Report the status of the last deployment. Respond with only the requested JSON object."}],
    "max_tokens": 64,
    "temperature": 0.0,
    "enable_thinking": false,
    "seed": 42,
    "grammar": {
      "type": "json_schema",
      "value": {
        "type": "object",
        "properties": {
          "run": {
            "type": "object",
            "properties": {
              "release_channel_name": {"type": "string", "enum": ["stable", "beta", "nightly"]},
              "observed_error_count": {"type": "integer", "minimum": 0, "maximum": 99}
            },
            "required": ["release_channel_name", "observed_error_count"],
            "additionalProperties": false
          },
          "note": {"type": "string"}
        },
        "required": ["run", "note"],
        "additionalProperties": false
      }
    }
  }'
```

**FF ON (observed this session, live host):**

```text
finish_reason=length
completion_tokens=64
content begins:
{
  "run":  \r  \r  \r  \r  \r  \r  \r  \r  \r  \r  \r  \r  \r  \r  \r  \r  \r  \r  \r ...
```

(Full raw response this session:
`{"id":"1","choices":[{"finish_reason":"length","index":0,"message":{"content":"{\n  \"run\":  \r  \r  \r  \r  \r  \r  \r  \r  \r  \r  \r  \r  \r  \r  \r  \r  \r  \r  \r ","role":"assistant","tool_calls":null},"logprobs":null}],...,"usage":{"completion_tokens":64,"prompt_tokens":28,"total_tokens":92,...}}`.)

**FF OFF (established in `06-b-n1-repro.md`, N=1, five fresh-restart repeats, identical every time):**

```text
finish_reason=stop
completion_tokens=47
content:
{"run": {"release_channel_name": "stable", "observed_error_count": 0}, "note": "The last deployment was successful."}
```

FF ON was likewise 5/5 identical across fresh restarts in `06-b` (`length`, 64 tokens, the same `\r`-loop
byte sequence every time) -- this session's live re-run above reproduces that exact `06-b` result again,
now also with CUDA decode graphs disabled (Sec 3).

## 3. Important controls / things ruled out

Stated explicitly so this is not re-investigated:

- **N=1.** Batch shape/composition is not the cause -- confirmed directly, not inferred (`06-b`).
- **`temperature=0`, five fresh-restart repeats each side: 5/5 identical FF-ON failures, 5/5 identical
  FF-OFF successes** (`06-b`). Ordinary sampling nondeterminism is not the explanation.
- **Request bodies were byte-identical** across the ON/OFF comparison, confirmed from the raw JSON, not
  the harness's summary (`06-b`).
- **llguidance's forced-byte mechanism was audited at the source level** (`06-d`). Both
  `Matcher::consume_ff_tokens` and `TokenParser::consume_ff_tokens` bottom out in the same
  `ff_tokens()`/single-token-consume primitives that ordinary (non-FF) grammar masking already uses via
  `compute_mask_or_eos()`. There is no separate "bulk consume" code path in llguidance itself. The basic
  content-level forced-token mechanism is not the likely source.
- **PagedAttention slot allocation and the widened-window slot mapping were traced** (`06-h`, static;
  `06-i`, runtime confirmation). The FF widened step's KV write slot matched the later ordinary step's
  predecessor-slot read. The widened window and logits selection (which row of the widened output feeds
  sampling) were also traced and match the expected last-row-of-window convention.
- **CUDA decode graphs were investigated at length** (`06-q`, `06-r`) as the leading candidate: ordinary
  `q_len=1` decode immediately after a widened FF step is served by `NormalPipeline::
  try_cuda_decode_graph_forward` -> `CudaDecodeGraphState::replay`, which bypasses `forward_embeds`
  entirely. **This session, the user ran the exact reproduction above with CUDA decode graphs disabled
  (`MISTRALRS_CUDA_GRAPHS=0`) and grammar FF ON, and the identical `\r`-loop failure still occurred.**
  This is a live, empirical result, not static analysis: **CUDA graph replay is not required for the
  bug.** It rules out the entire `06-q`/`06-r` CUDA-graph-boundary line of investigation as the (sole)
  explanation -- the bug reproduces in the fully-eager `forward_embeds` path too.
- **The flush-before-widened-step fix was implemented and exercised** (`06-n` design, `06-o`
  implementation: `engine/mod.rs`, flush `flush_recurrent_speculative_transitions` for the affected
  sequences immediately before a widened FF step runs, gated on `pending_ff_width.is_some()`). Per
  `06-p`, this flush executes successfully and reaches the correct sequence/slot. **It did not fix the
  exact reproduction above.** Do not propose this fix again.
- **A CUDA-graph-invalidation idea was designed in the prior session** (bump `HybridCache`'s existing
  `recurrent_storage_generation` counter after any `Decode`-batch-kind, `query_len>1` GDN forward, to
  force the next decode dispatch through eager capture-on-miss instead of graph replay). **This was never
  implemented in `/workspace`.** Its entire premise (that graph replay specifically was implicated) was
  disproven by the CUDA-graphs-disabled test in this same session, before implementation. It should
  **not** be described as an implemented or validated fix, and should not be resurrected without new
  evidence that specifically re-implicates CUDA graphs.

## 4. What the investigation discovered about Qwen3.5/GDN

- Qwen3.5's text backbone is a **hybrid** model: some layers are ordinary full attention, others are
  Gated DeltaNet (GDN) linear-attention layers with pooled, slot-indexed recurrent state
  (`conv_state`, `recurrent_state`, and an optional deferred-state accumulator).
- **Ordinary `q_len=1` decode can use a GDN "deferred-decode" fast path**: instead of materializing
  `recurrent_state` on every token, it appends a small pending update to a separate `deferred_state`
  buffer and only periodically (or on explicit flush) folds those pending rows into `recurrent_state`.
- **Grammar FF creates a widened multi-token decode window** (`query_len > 1`) when it splices multiple
  already-known grammar-forced tokens into one forward call.
- That widened path is **structurally incompatible with the deferred-decode fast path** (which requires
  `query_len==1`) and instead goes through the **general multi-token recurrent path**
  (`forward_recurrent_core`, the `gated_delta_rule_recurrence_kernel_vmajor_grouped` CUDA kernel),
  which materializes state directly.
- **The single functional code change grammar-FF makes to this entire subsystem, relative to the
  `master` merge-base (`d5ae0f18f2170f10d30880cb7d21fb0880410e7b`), is one dispatch relaxation** in
  `mistralrs-core/src/gdn/backend.rs`'s `causal_conv1d` function: previously, `RecurrentBatchKind::Decode`
  required `seq_len == 1` or the function `bail!`ed with "GDN decode expects a single-token query."
  Grammar FF's widened window needed `Decode`-tagged calls with `seq_len > 1` to work at all, so this
  guard was relaxed to fall through to `causal_conv1d_full` (previously used only for genuine sequence-start
  prefill) instead of erroring. **This is the exact reachability change that makes the widened-decode
  scenario possible in the first place** -- confirmed by diffing every GDN/CUDA-graph-adjacent file
  against the merge-base (`06-r`): `pipeline/cuda_graph.rs`, `kv_cache/hybrid_cache.rs`, `gdn/layer.rs`,
  `gdn/cache.rs`, `cuda/gdn.rs`, and `cuda/gdn.cu` are all byte-identical to `master`. This one dispatch
  relaxation, and its downstream consequences, is **the most important architectural boundary left for
  future investigation** (restated in Sec 6).

## 5. Conv-state findings

- Static analysis (`06-r`, `06-s`) proved, by tracing exact index arithmetic with concrete values (not
  just structural pattern-matching), that the `conv_state` storage conventions between the multi-token
  (`causal_conv1d_full`, used by the widened FF step) and single-token (`causal_conv1d_update`, used by
  ordinary decode) paths **are equivalent**: both address the pool row via the identical
  `gdn_state_row(slot_indices, b, 0, 1)` formula, both maintain the same `[conv_dim, kernel_size]`
  circular most-recent-token window in the same oldest-to-newest order, and the state-write kernel
  (`save_conv_state_kernel`) is a pure data copy with no arithmetic -- worked through with concrete prior/
  new-token values and confirmed to produce byte-identical windows and final state to what an equivalent
  sequence of single-token updates would produce (`06-s`, Sec 2).
- A **real floating-point difference was found and is unresolved**: when `kernel_size == 4` (the actual
  Qwen3.5/Qwen3-Next GDN config value) and the input stride qualifies, the widened path's width-4-tiled
  output kernel (`causal_conv1d_full_width4_tiled_kernel`) accumulates the convolution sum via
  right-to-left `__fmaf_rn` (fused multiply-add) chains, while the single-token update path's equivalent
  kernel (`gdn_conv_width4_update`) accumulates via plain left-to-right multiply-then-add. These are not
  guaranteed bit-identical under IEEE 754. **This is a real numerical asymmetry in the conv1d *output*
  (not in the `conv_state` bytes it writes, which are proven identical), but it was never proven to
  explain the catastrophic, deterministic `\r` loop** -- a consistent, always-wrong-at-the-same-position
  failure at `temperature=0` across fresh restarts is not the usual signature of epsilon-scale rounding
  noise, and this gap was explicitly flagged as a likely-insufficient, secondary finding, not a
  root-cause candidate (`06-s`, Sec 3-4).

## 6. Most likely remaining investigation boundary

**Do not claim a root cause has been found.** The strongest unresolved boundary, per the full arc of
evidence (Sec 3-5), is the **GDN recurrent-state and output transition from the widened multi-token
path to the subsequent ordinary single-token path** -- specifically the eager `causal_conv1d_full` /
`forward_recurrent_core` write, and everything downstream of it (recurrent_state, the delta-rule
kernels, and whatever mixed_qkv values feed them), handed off to the next `q_len=1` step's consumption
of that state, whether via eager `forward_embeds` or CUDA-graph replay (now proven not to matter, Sec 3).
This is a narrower and better-justified target than CUDA graphs (ruled out this session), PagedAttention
slot mapping (traced and matched, `06-h`/`06-i`), or the basic llguidance forcing mechanism (audited and
structurally sound, `06-d`).

Concretely, future work should **compare actual GDN `recurrent_state` values and GDN layer outputs**:

1. Immediately before and immediately after the widened multi-token step, and
2. At the first subsequent `q_len=1` step,

**against an equivalent sequence of ordinary single-token steps** producing the same logical token
stream (i.e., an FF-OFF run at the same logical position). This requires live GPU value dumps, not
further source reading -- static analysis of this exact boundary was pursued as far as it can go without
runtime data (`06-r`, `06-s`). Given the CUDA-graphs-disabled result this session, this comparison should
now be done with graphs off (simpler: pure eager `forward_embeds` on both legs), which also removes any
residual worry about the comparison itself being contaminated by graph-replay behavior.

A separate, unexplored angle worth flagging for whoever resumes this: the specific symptom -- the model
repeatedly emitting the exact same non-terminal whitespace-class byte (`\r`) dozens of times in a row
immediately after `"run":` -- is also consistent with a **grammar-recognizer-state** issue (the FF
splice leaving the grammar matcher's internal state permissive of whitespace when it should require `{`)
rather than a purely numerical/GDN issue. This was not investigated this arc (the investigation converged
early on the GDN/CUDA-graph boundary per `06-c` and stayed there); it is offered here only as an
unexplored possibility, not as a finding.

## 7. Source changes / fixes that were actually made

Correctness fixes and mechanism work, landed and part of the branch (see `git log master..
grammar-fast-forward`), distinct from this investigation's diagnostics:

- The grammar FF splice lifecycle/accounting fixes from Plan 02 (`e7bba90be` widen the decode window,
  `e9b0eac07`/`d217eb32b` compute and replay splices, `49848e501` correct splice handling under batching/
  PagedAttention/sampling order, `3b37de65e` populate replayed-token bytes and fail on unrecoverable
  splice rollback plus FF tests, `a3851faba` resolve splices in the engine and account for width,
  `2c061c838` drop staged splices on realloc and reject them in the prefill-chunk window,
  `19da3d451` account for stranded splices).
- **AnyMoe was disabled for grammar FF** (`4ba40ca9e`) because its expert routing is window-level and
  could diverge under widened FF.
- Observability metrics were added for grammar FF (`83deb86ad` staged/fed counters and splice-drop
  reasons, `6c1edc2bb`/`9c413eed4` resolved-support classification, `2dd39c871` enabled FF for GGUF/GGML
  text pipelines).
- **Runtime tracing instrumentation was added during this investigation** (`1adb03ea0`, `0b2773ea2`,
  `30c68b149` -- `ff_trace` debug-level logging of FF splicing and KV/recurrent-state management) and
  remains in the branch, DEBUG-gated and harmless to leave in place.
- **The flush-before-widened-step change was implemented and tested** (`06-n`/`06-o`, in
  `engine/mod.rs`) **but did not resolve this reproduction.** It is a real, working piece of defensive
  bookkeeping (it does correctly flush GDN deferred state before a widened step) but is not, by itself,
  the fix for the bug in this report.
- **Do not imply the bug was fixed.** It was not. No change in the branch as of this session resolves the
  reproduction in Sec 2.
- **The proposed CUDA-graph invalidation idea from the prior session was never implemented** and must not
  be listed as a landed or validated change (Sec 3).
- `/workspace` (branch `grammar-fast-forward`) remains dirty with exactly one uncommitted file,
  `mistralrs-core/src/vision_models/qwen3_5/text.rs`, containing the `06-q` GDN-state checksum diagnostic
  (`ff_trace_gdn_state_checksum`) plus its two `forward_embeds` call sites. This diagnostic is DEBUG-gated,
  additive-only, and safe to keep or drop; it was left in place across every session in this arc and is
  still uncommitted as of this report.

## 8. Useful artifacts / reports

All in `plans/ff-round-two/reports/` on branch `ff-demo-artifacts` (this branch; unrelated history to
`grammar-fast-forward`, checked out separately as a worktree at `/tmp/ff-artifacts-wt` throughout this
arc). Most relevant, in investigation order:

- `06-b-n1-repro.md` -- the canonical N=1, five-repeat-each-side reproduction this report's numbers come
  from.
- `06-d-llguidance-audit.md` -- source-level audit ruling out a separate/relaxed forced-token consume path
  in llguidance.
- `06-e-kv-row-root-cause.md` -- deep static trace of KV-row accounting around the widened-to-ordinary
  transition (despite the filename, its own status line is "deep static trace only" -- treat as a traced
  hypothesis, not a confirmed root cause; superseded by `06-h`/`06-i`'s direct slot-match confirmation).
- `06-h-kv-slot-allocation-trace.md` / `06-i-slot-match-confirmed.md` -- static trace and runtime
  confirmation that PagedAttention slot allocation across the transition is correct.
- `06-n-fix-design-flush-before-widened-step.md` / `06-o-fix-implemented.md` / `06-p-handoff-kernel-
  aliasing-closed.md` -- the flush-before-widened-step fix: design, implementation, and confirmation that
  it executes correctly but does not fix the bug.
- `06-q-cuda-decode-graph-bypass-found.md` -- discovery that ordinary post-widened decode can bypass
  `forward_embeds` via CUDA decode-graph replay (the finding this session's live test superseded as *the*
  explanation, per Sec 3, though the underlying mechanism trace remains accurate).
- `06-r-conv1d-dual-kernel-path-narrowed.md` -- branch-vs-`master` diff proving the single functional GDN/
  CUDA-graph-adjacent change is the `causal_conv1d` dispatch relaxation (Sec 4 above).
- `06-s-conv1d-state-equivalence-proven-output-fmaf-gap-found.md` -- worked-example proof of `conv_state`
  equivalence and the fmaf/accumulation-order gap (Sec 5 above).
- `06-t-final-handoff-unresolved.md` -- this report.

## Files added

- `plans/ff-round-two/reports/06-t-final-handoff-unresolved.md` (this report). No other files changed;
  no source changed on this branch this session.
