# 06-Q -- breakthrough: ordinary post-widened decode bypasses `forward_embeds` entirely via
CUDA decode-graph replay (2026-09-11, twenty-first session)

Status: **investigation finding, not a fix.** This session added a GDN-state checksum diagnostic
inside `Qwen3_5TextModel::forward_embeds` per `06-p`'s recommendation, ran it against the live host
server, and found the checksum logs were entirely absent for ordinary decode steps following the
widened FF step. Tracing that absence to its cause is this session's actual result: ordinary
`query_len=1` decode after the widened step is served by a raw CUDA graph replay that never calls
`forward_embeds`, so nothing at the Rust model level -- including this session's own instrumentation
-- can observe what happens to GDN state on that path. No source was modified beyond the diagnostic
already described; no fix implemented; no further hypothesis investigated yet, per explicit
instruction.

## 1. What was tried first: the runtime checksum diagnostic (per `06-p` Sec 4)

Added `Qwen3_5TextModel::ff_trace_gdn_state_checksum` (private helper) plus two call sites in
`mistralrs-core/src/vision_models/qwen3_5/text.rs`:

- `forward_entry` -- immediately after `recurrent_metadata` is obtained, before any of
  `forward_embeds`'s own conditional state mutations (transition-log apply, deferred-state flush).
- `forward_exit` -- immediately after the per-layer computation closure returns `Ok`, before return.

Each call gathers the active sequence's physical slot row from `RecurrentStatePool::{conv_state,
recurrent_state}` (via `HybridCache::state_indices_host()`, the same handle the real kernel dispatch
uses) and logs four `f64` reductions per `LinearAttention` layer: `conv_sum`, `conv_sumsq`, `rec_sum`,
`rec_sumsq`, tagged with `boundary`, `query_len`, `layer_idx`, `slots`. Gated behind
`tracing::enabled!(Level::DEBUG)` since `to_scalar` forces a CUDA sync.

This diagnostic remains uncommitted in `/workspace` (branch `grammar-fast-forward`) as of this report.

## 2. The runtime evidence that redirected the investigation

The host was rebuilt and rerun with the standard N=1 nested-JSON-schema FF-ON repro. The pasted log
tail showed the widened step (`logical_position=53`, `sampled_token=763`) completing, followed by
fourteen further ordinary decode positions (54 through 67):

```text
logical_position=54 sampled_token=328   pending_ff_width=None use_pending_ff=false predecessor slot check_pos=53 check_slot=117
logical_position=55 sampled_token=760   pending_ff_width=None use_pending_ff=false predecessor slot check_pos=54 check_slot=118
logical_position=56 sampled_token=22449 pending_ff_width=None use_pending_ff=false predecessor slot check_pos=55 check_slot=119
...
logical_position=67 sampled_token=248046
flush_recurrent_transitions_for_sequences: entry sequence_ids=[1] slots=[] uses_gdn_deferred_state=true
flush_deferred_recurrent_state: resolved host_slots host_slots=Some([])
flush_deferred_recurrent_state: host_slots empty, no-op
```

**Zero `ff_trace: gdn_state_checksum` lines appear anywhere in this range**, despite:

- `mistralrs_core=debug` being active for the whole crate (other `ff_trace` lines from the very same
  module, `mistralrs_core::vision_models::qwen3_5::text`, do fire, e.g. the
  `flush_recurrent_transitions_for_sequences` line above -- ruling out a logging-target
  misconfiguration).
- `pending_ff_width=None` / `use_pending_ff=false` for every one of these positions, confirming they
  are ordinary (non-widened) decode steps, exactly the steps the checksum was meant to bracket.

This directly falsified the assumption the diagnostic plan was built on: that ordinary decode
necessarily runs through `forward_embeds`.

## 3. Root cause of the missing logs: a second, independent CUDA graph mechanism

Static trace of the call path for ordinary `query_len=1` decode, confirmed by reading full function
bodies (not inferred):

- `NormalPipeline::forward_step` (`mistralrs-core/src/pipeline/normal.rs:2365`) calls
  `self.try_cuda_decode_graph_forward(...)` at line 2413 **before** ever calling
  `self.model.forward(...)`. If it returns `Ok(Some(replay))` (lines 2422-2429), that replay's logits
  are returned immediately and `self.model.forward()` -- and therefore `forward_embeds` -- is **never
  invoked** for that step.
- `try_cuda_decode_graph_forward` (`normal.rs:1766`) gates on: graphs enabled
  (`cuda_decode_graphs_enabled()`), `RecurrentBatchKind` supported, `self.model.
  supports_cuda_decode_graphs()` (Qwen3.5 hard-codes this `true` via `SUPPORTS_CUDA_DECODE_GRAPHS`,
  `text.rs:2840,2925`), no speculative proposer, PagedAttention decode metadata present (not a prefill
  chunk), `q_len == 1`, CUDA device. **No check of `seq.recognizer` / grammar state appears anywhere
  in this function** -- confirmed by reading the full body, not just the early-return conditions.
- `CudaDecodeGraphState::replay` (`mistralrs-core/src/pipeline/cuda_graph.rs:1976`) only updates
  `input_ids` and metadata host-staging buffers (`entry.host_staging.update(...)`), then launches the
  previously captured graph (`entry.launch(...)`). **It contains no call into `self.model.forward` /
  `forward_embeds`** -- confirmed by reading the full function body.
- The Rust model path (`forward_embeds`, including its `deferred_gdn` branch) is only executed **once
  per captured graph key/bucket**, inside `capture_cuda_decode_graph_step`
  (`normal.rs:2043`): once eagerly for real (`self.model.forward(...)`, line 2103, producing
  `warmup_logits`), and once more inside a CUDA stream-capture closure (lines 2123-2134) purely to
  record the kernel graph ("CUDA stream capture records recurrent writes without executing them,"
  `normal.rs:2107`). After that one-time capture, the recorded kernel sequence -- including whichever
  GDN branch (`deferred_gdn` true/false) was live at capture time -- is replayed verbatim on every
  subsequent call for that key. The Rust-level conditional is never re-evaluated again.

This is the concrete mechanism behind items 1-4 the user asked to record:

1. Ordinary `query_len=1` Qwen3.5 decode can be served entirely by
   `NormalPipeline::try_cuda_decode_graph_forward` / `CudaDecodeGraphState::replay`, without entering
   `Qwen3_5TextModel::forward_embeds`.
2. Therefore the GDN checksum instrumentation added this session, which lives inside
   `forward_embeds`, structurally cannot fire for ordinary post-widened decode -- its absence in the
   log is the expected behavior of the code as written, not a bug in the diagnostic.
3. The widened grammar-FF step (`query_len > 1`) cannot use the decode graph at all: dispatch is
   unconditionally skipped for `q_len != 1` (`normal.rs:1838`). It always takes the eager
   `self.model.forward` path, `deferred_gdn` is forced false (that branch requires `query_len==1`),
   the pre-forward `flush_deferred_recurrent_state(&hybrid_cache, None)` runs, and the general
   multi-token recurrence kernel (`gated_delta_rule_recurrence_kernel_vmajor_grouped`, previously
   audited safe for in-place aliasing in `06-p`) executes directly, fully materializing
   `recurrent_state`/`conv_state` in the pool.
4. The very next ordinary step can then replay a CUDA graph captured at an earlier, unrelated point in
   time (most likely server warmup/precapture via `precapture_cuda_decode_graphs_impl`,
   `normal.rs:1954`) rather than re-executing the Rust model forward path.

## 4. Why this explains the observed correctness boundary

Restating the fact this investigation has centered on since `06-c`: the widened step's own sampled
token (position 33 in the original notation, position 53 in this session's fresh log) is correct;
the very next ordinary decode step is the first wrong token. Given Sec 3, the boundary now has a
structural explanation, not just a symptom description: it is precisely the seam between two
independently-realized execution paths for touching the same pooled GDN state --

- the widened step's **eager, always-fresh** `forward_embeds` call (full Rust dispatch, live
  multi-token kernel, full materialization), and
- the following step's **pre-recorded, replayed** CUDA graph (fixed kernel sequence, captured at some
  unrelated earlier moment, using whatever `deferred_gdn` branch was true then).

No flush occurs between them either: the pasted log confirms `pending_ff_width=None` for position 54,
so the engine-level flush (`flush_recurrent_speculative_transitions`) that gates on a pending FF splice
does not run for this transition -- it only runs immediately before a widened step, not immediately
after one.

## 5. The earlier CUDA-graph ruling was too broad

`06-g`'s conclusion -- "grammar-constrained sequences cannot re-enter CUDA graph paths, since those
paths require `SequenceRecognizer::None`" -- is not wrong on its own terms, but it evidently described
a *different* graph mechanism (the resident-sampling / token-sampling graph path referenced in that
report), not the per-token *model forward* decode graph in `normal.rs` traced this session. There are
at least two independent CUDA graph subsystems in this pipeline (this `CudaGraphComponent::Target`
decode-forward graph, and separately the DFlash speculative-decode graph machinery in
`speculative/dflash.rs`), and `try_cuda_decode_graph_forward` specifically has no grammar-recognizer
gate at all. This session's finding narrows -- not reopens -- `06-g`: CUDA graph involvement is back
in play, but only for this one specific mechanism, and the mechanism is now identified precisely
rather than being an open-ended re-suspicion of "CUDA graphs" generally.

## 6. Negative result restated (do not re-litigate)

The engine-level flush-before-widened-step fix (`06-n`/`06-o`) **does execute successfully** for the
correct sequence and GDN slot -- this was already confirmed in `06-p` via the runtime trace showing
`pending_ff_width=Some(1)` -> `flush_recurrent_speculative_transitions` called -> `flushed=true` --
and it **did not** fix the reproduction. Nothing this session changes that conclusion; it is restated
here only for continuity, not reopened as a live hypothesis.

## 7. What is proven vs. what remains a hypothesis

**Proven this session, by direct code reading (not inference):**

- Ordinary `query_len=1` decode can bypass `forward_embeds` entirely via CUDA decode-graph replay.
- The widened `query_len>1` FF step cannot use that graph and always runs eager.
- The graph replay path has no grammar/recognizer gate.
- `replay()` performs no Rust-level model computation.

**Not yet proven -- this is the next hypothesis, not yet investigated:**

We have **not** shown that the GDN state a captured decode graph reads/writes (the `deferred_state`
staging buffer and/or however the deferred kernel addresses `recurrent_state`/`conv_state`) is
actually stale, wrongly shaped, or invalid after the eager widened step runs. The mechanism in Sec 3-4
is a structurally plausible seam, not a confirmed defect -- no runtime or static evidence yet shows the
captured graph's kernel reads incorrect data at that seam. This remains open.

## 8. Next investigation boundary (do not proceed further this session)

The user's explicit next step, recorded verbatim for continuity:

1. Inspect what GDN state/buffers the captured CUDA decode graph addresses (in particular
   `deferred_state` vs. the pooled `recurrent_state`/`conv_state`), and how the eager widened path
   changes them.
2. Compare the current `grammar-fast-forward` branch against its base/PR parent to determine whether
   this is a regression caused specifically by the grammar-FF widened-step mechanism, or a pre-existing
   Qwen3.5 CUDA-decode-graph issue that FF merely exposes (e.g. any eager multi-token forward --
   unrelated to grammar FF -- interleaved with graph-captured decode might trigger the same class of
   bug).

Do not implement a fix and do not add further instrumentation until that comparison is done.

## 9. Repository state at end of session

- `/workspace`, branch `grammar-fast-forward`: **dirty**, one file changed
  (`mistralrs-core/src/vision_models/qwen3_5/text.rs`, the checksum diagnostic from Sec 1), left
  uncommitted. No other source touched. The diagnostic is harmless to leave in place (DEBUG-gated) and
  remains useful for a future boundary that *does* go through `forward_embeds`, but it cannot observe
  the seam identified in this report.
- This report was written and committed from `/tmp/ff-artifacts-wt`, branch `ff-demo-artifacts`
  (recreated this session after the prior worktree directory was found pruned/missing at session
  start; `git worktree prune` then `git worktree add /tmp/ff-artifacts-wt ff-demo-artifacts` restored
  it cleanly against the existing branch tip, verified independent of `/workspace`).

## Files added

- `plans/ff-round-two/reports/06-q-cuda-decode-graph-bypass-found.md` (this report). No other files
  changed on this branch this session.
