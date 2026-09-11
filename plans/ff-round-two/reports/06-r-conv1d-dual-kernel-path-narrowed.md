# 06-R -- static trace of the CUDA decode-graph/GDN-state boundary; narrows to the
causal_conv1d_full/causal_conv1d_update dual-kernel handoff (2026-09-11, twenty-second session)

Status: **investigation finding, not a fix.** Per explicit instruction this session did not add
instrumentation, did not modify runtime behavior, and did not run a live reproduction. This is a
source-only trace of exactly the boundary `06-q` identified (CUDA decode-graph replay vs the eager
widened forward's effect on pooled GDN state), plus a full diff of this branch against its merge-base
for every file touching that boundary. No source was changed this session; `text.rs` remains dirty
with only the diagnostic from `06-q` (uncommitted, unchanged).

## 1. What was traced, and where

Per the user's five-point scope: (1) what the captured CUDA decode graph reads/writes for GDN, (2) how
`recurrent_state`/`deferred_state`/`conv_state`/`state_indices` are allocated and addressed, (3) what
the eager widened forward writes vs what the captured graph expects, (4) whether capture bakes in
anything that an eager widened step can invalidate, (5) whether this is a regression introduced by
`grammar-fast-forward` or pre-existing. All five are answered below from direct reads of
`mistralrs-core/src/pipeline/cuda_graph.rs`, `.../pipeline/normal.rs`, `.../kv_cache/hybrid_cache.rs`,
`.../gdn/{layer,cache,backend}.rs`, `.../cuda/gdn.rs`, and `.../cuda/gdn.cu`.

## 2. Established facts

**Buffer allocation and addressing (items 1-2):**

- `RecurrentStatePool` (`hybrid_cache.rs:445`) owns `conv_state` `[capacity, conv_dim, kernel_size]` and
  `recurrent_state` `[capacity, heads, K, V]`. `GdnDeferredStatePool` (`hybrid_cache.rs:276`) owns
  `key`/`delta`/`decay` `[capacity, GDN_DEFERRED_STATE_DEPTH, ...]` and a `pending_rows` u32 cursor
  `[capacity]`. `GDN_DEFERRED_STATE_DEPTH = 4` (`gdn.cu:3607`). All four are addressed by **physical
  pool row = slot index**, not by batch position; `state_indices` is the per-call `[B]` table mapping
  batch row to physical slot.
- On CUDA, every GDN layer forward (widened or ordinary, decode or prefill) is built via
  `GdnLayerCache::checkout(pool, &indices)` (`gdn/cache.rs:72`), which is **unconditionally the pooled,
  in-place variant** (`Self::pooled`, `slots: Some(indices)`) except when `packed_layout.is_some()`.
  `packed_gdn_layout` (`packed_gdn.rs:70`) requires `ctx.flash_params().packed` -- a distinct
  prompt-batch-packing feature, unrelated to grammar FF, gated additionally on
  `RecurrentBatchKind::Prefill` and first-prompt-chunk. It is `None` for any decode-continuation
  request, including a grammar-FF widened step. **Confirmed by reading `packed_gdn_layout`'s guards
  directly**, not assumed: the widened step in this repro takes the identical in-place/pooled addressing
  as ordinary decode.
- `try_cuda_decode_graph_forward` (`normal.rs:1766`) and `capture_cuda_decode_graph_step` (`normal.rs
  :2043`) never reference `recurrent_state`, `conv_state`, or the deferred-state tensors by name --
  confirmed by grep across `cuda_graph.rs` (0 hits for `recurrent_state`/`conv_state`/`deferred_state`
  outside doc comments). The only GDN-related value the graph machinery treats as a **dynamic**,
  per-replay input is `state_indices`, copied host-to-device via `CudaGraphHostStaging::copy_from`
  (`copy_state_indices`, `cuda_graph.rs:831`) before every `entry.launch()`. The pool tensors
  (`recurrent_state`/`conv_state`/`deferred_key`/`delta`/`decay`/`pending_rows`) are **not** re-bound
  per replay; their device pointers, whatever they were at capture time, are baked into the recorded
  CUDA graph nodes as fixed kernel arguments (standard `cudaStreamBeginCapture` semantics -- confirmed
  by the absence of any host-staging entry for them).

**Item 4 (can capture bake in something a widened step invalidates):**

- `HybridCache` tracks `recurrent_storage_generation: u64`, bumped by `advance_recurrent_storage_generation()`
  on every pool-reallocating event: capacity growth (`hybrid_cache.rs:1578`), transition-log reservation
  (`:1666`), deferred-state reservation/disablement (`:1724`, `:1777`), and the checkpoint-lane resize
  path (`:1862`, `:2088`). Deferred-state resize is included in the same `install_resized_storage` call
  as `conv_state`/`recurrent_state` (`:1560-1569`), so a pool grow does bump the generation and would
  invalidate stale pointers.
- `CudaDecodeGraphState::observe_recurrent_storage_generation` (`cuda_graph.rs:1955`) clears **all**
  captured graph entries (`self.clear()`) the moment the observed generation differs from the previous
  one, and this is called unconditionally at the top of every `try_cuda_decode_graph_forward` invocation
  (`normal.rs:1885`), before any replay lookup. So a pool resize between the widened step and the
  following ordinary step, if it happened, would force a fresh capture rather than replay stale
  pointers. **No pool-resizing event is structurally expected in this repro** (N=1, one sequence, stable
  batch composition across the widened/ordinary boundary) -- this mechanism is a plausible generic
  safeguard, not a live suspect for this specific reproduction, absent evidence of a resize actually
  firing.

**Deferred-state cursor lifecycle** (the mechanism `06-q` flagged as "not yet proven stale"):

- `forward_deferred_decode` (`gdn/layer.rs:723`, used only for `query_len==1` decode) writes the new
  token's key/decay/delta into the deferred pool at row = current cursor value, then
  `gdn_deferred_cursor_advance_kernel` (`gdn.cu:4821`) advances the cursor with wraparound (`cursor ==
  DEPTH-1 ? 0 : cursor+1`). When `deferred_rows == DEPTH-1` at entry, the SAME kernel call also folds all
  `DEPTH` pending rows into `state_pool` in-line (`gdn.cu:4757-4765`) before the cursor wraps to 0 -- an
  automatic self-flush every 4 decode tokens.
- The explicit flush (`flush_deferred_state_cuda` / `gdn_flush_deferred_state_value_major_128_kernel`,
  `gdn.cu:4842`) folds `[0, deferred_rows)` into `state_pool`, then unconditionally zeros
  `deferred_cursor[active_slot]` via `gdn_deferred_cursor_clear_kernel` (`gdn.cu:4967`).
- The widened step (query_len>1) forces `deferred_gdn=false` (`text.rs:2427`), which triggers
  `flush_deferred_recurrent_state(&hybrid_cache, None)` (`text.rs:2438`) **before** the eager multi-token
  kernel runs. With `slots=None`, this resolves `cache.state_indices_for_device(...)` -- the same active
  state indices the widened forward itself uses. The eager multi-token kernel that follows
  (`forward_recurrent_core` / the general `gated_delta_rule_recurrence_kernel_vmajor_grouped` path)
  never touches `deferred_key`/`delta`/`decay`/`pending_rows` at all -- it writes `recurrent_state`
  directly. So after the widened step: `deferred_cursor[slot] == 0` (cleared by the pre-step flush, never
  re-set by the eager kernel), and `recurrent_state[slot]` is fully materialized.
- The next ordinary step's captured graph (deferred-decode kernel) reads `deferred_rows =
  deferred_cursor[active_slot]` live from GPU memory at replay time (not baked into the graph as a
  constant -- it is a kernel-body memory read against a fixed-address buffer). With
  `deferred_rows==0`, the pending-fold loop is a no-op and `state` is loaded directly from the
  freshly-materialized `recurrent_state`. **This entire chain is internally self-consistent; no stale-read
  defect was found.**

**conv_state format/addressing consistency between the two kernel families (item 3):**

- `causal_conv1d` (`gdn/backend.rs:824`) dispatches to `causal_conv1d_update` (decode, seq_len==1) or
  `causal_conv1d_full` (everything else, including the widened FF window) based on `batch_kind` and
  `seq_len`.
- Both CUDA kernel families address the pool row via the identical formula `gdn_state_row(slot_indices,
  b, 0, 1) = slot_indices[b]` (`gdn.cu:270-274`) -- verified identical arguments `(0, 1)` at every call
  site in both `causal_conv1d_update_kernel`/`causal_conv1d_update_width4_kernel` and
  `causal_conv1d_full_kernel`/`causal_conv1d_full_width4_tiled_kernel`/`save_conv_state_kernel`.
- Both use the identical oldest-to-newest circular-window convention (index `kernel_size-1` = most
  recent). `save_conv_state_kernel` (`gdn.cu:1501`, used by `causal_conv1d_full` for **both** the
  generic and width4-tiled output paths) computes `pad = kernel_size - seq_len`: for `i < pad` it
  copies `prior[i + seq_len]` (the newest `pad` entries of the old state), else it copies from the new
  input tokens directly -- verified this produces the same layout `causal_conv1d_update_kernel`'s
  shift-left convention expects to read next, for both `seq_len < kernel_size` and `seq_len >=
  kernel_size`.
- **No format, addressing, or convention mismatch was found by direct kernel inspection.**

## 3. Branch diff vs merge-base (item 5) -- the actual regression surface

Merge-base of `grammar-fast-forward` and `master`: `d5ae0f18f2170f10d30880cb7d21fb0880410e7b`.

```
git diff d5ae0f18f...HEAD -- pipeline/cuda_graph.rs pipeline/normal.rs kv_cache/hybrid_cache.rs \
    gdn/ cuda/gdn.rs cuda/gdn.cu vision_models/qwen3_5/text.rs
```

Result: **`cuda_graph.rs`, `hybrid_cache.rs`, `gdn/layer.rs`, `gdn/cache.rs`, `cuda/gdn.rs`, and
`cuda/gdn.cu` are byte-identical to the merge-base -- zero diff.** All deferred-state pool machinery,
cursor kernels, CUDA graph capture/replay logic, and generation-based invalidation described in Sec 2
pre-exist unmodified on `master`. Only three files differ, and only one of the changes is functionally
relevant to GDN/CUDA-graph state:

- **`gdn/backend.rs`** (the only functional change): relaxes `causal_conv1d`'s dispatch guard. Before:
  `RecurrentBatchKind::Decode` implied `seq_len == 1` or the function `bail!`ed ("GDN decode expects a
  single-token query"). After: `Decode` batch-kind with `seq_len > 1` now falls through to
  `causal_conv1d_full` instead of erroring. **This combination -- a `Decode`-tagged, multi-token,
  pool-addressed forward, immediately followed by a `Decode`-tagged, single-token, pool-addressed
  forward on the same slot -- was structurally unreachable before this branch.** `Prefill`-tagged calls
  (the pre-existing use of `causal_conv1d_full`) only ever occur once at sequence start, never followed
  by a same-slot decode reading state the "full" kernel just wrote in a *mid-generation* sense with a
  large existing history.
- `pipeline/normal.rs`: one unrelated field addition (`supports_grammar_fast_forward`) on the pipeline
  metadata struct -- no GDN or CUDA-graph code path touched.
- `vision_models/qwen3_5/text.rs`: tracing/logging only (the `ff_trace` debug! calls from commit
  `1adb03ea0`, plus `06-q`'s still-uncommitted checksum diagnostic). The `deferred_gdn` gate, the
  flush-before-widened-step call, and all dispatch logic in `forward_embeds` were already present on
  `master`, byte-for-byte, before this branch's tracing was layered on top -- confirmed by diffing the
  non-tracing lines.

**Conclusion for item 5:** this is not a broad regression in the CUDA-graph/GDN subsystem; the entire
subsystem this session traced (cursor lifecycle, generation invalidation, pool addressing, conv-state
save/restore) is pre-existing and unmodified. The branch's sole causal contribution is enabling, for the
first time, a `Decode`+multi-token forward to occur at all. Whatever divergence exists must live either
in a subtle behavioral difference between `causal_conv1d_full` and `causal_conv1d_update` that
`06-r`'s static trace did not surface, or in a pre-existing subsystem bug that was simply unreachable
until this branch made the `Decode`+multi-token combination possible.

## 4. Ruled out this session (by direct source reading)

- Stale/unflushed deferred cursor after the widened step -- traced the full flush -> eager-kernel ->
  replay chain; self-consistent, no stale read found (Sec 2).
- CUDA-graph pointer staleness from an intervening pool resize -- guarded by
  `recurrent_storage_generation`, checked on every `try_cuda_decode_graph_forward` call, and no
  resize-triggering event is structurally expected in this N=1 single-sequence repro.
- conv_state layout/addressing mismatch between `causal_conv1d_full` and `causal_conv1d_update` --
  verified identical row-addressing formula and identical oldest-to-newest window convention across all
  four kernel variants (generic and width4-specialized, both directions).
- The widened step silently using the non-pooled/"gathered" GDN cache path (which would leave the
  persistent pool stale while its own output still looked correct) -- ruled out: `packed_layout` is
  `None` for this repro by construction (`flash_params().packed` is a distinct, inactive prompt-packing
  feature), so the widened step takes the identical pooled/in-place path as ordinary decode.
- (Restated from `06-o`, not re-investigated this session) the engine-level flush-before-widened-step fix
  executes successfully but does not fix the reproduction.

## 5. Remaining hypothesis -- not disproven, not confirmed

No static defect was found in any of the mechanisms items 1-4 asked about, despite tracing every named
buffer and every kernel that touches it. What *is* certain: the single branch-introduced change
(Sec 3) is the only way this exact scenario -- a `Decode`-tagged multi-token pooled forward immediately
followed by a `Decode`-tagged single-token pooled forward on the same slot -- can occur anywhere in this
codebase, and no test anywhere validates that `causal_conv1d_full` and `causal_conv1d_update` produce
identical state when the same token stream is split across the two kernel families. The one relevant
test, `causal_conv1d_prefill_continues_from_existing_state` (`gdn/backend.rs:1868`), checks
`causal_conv1d_full`'s **CPU** implementation against itself split at a different chunk boundary -- it
does not compare against `causal_conv1d_update`, and it does not touch CUDA at all.

`causal_conv1d_full_kernel` / `causal_conv1d_full_width4_tiled_kernel` / `save_conv_state_kernel` and
`causal_conv1d_update_kernel` / `causal_conv1d_update_width4_kernel` are two independently-written,
hand-optimized CUDA implementations of what should be the same causal-convolution update. Structural
inspection found no discrepancy, but a subtle numerical or indexing difference specific to actual
runtime values (dtype rounding order, a stride combination not exercised by the CPU-only test, or
something in the width4-tiled fast path specifically) cannot be excluded by reading code structure
alone. This is the live hypothesis.

Separately, and out of scope for "first wrong token" specifically: `GDN_DEFERRED_STATE_DEPTH = 4`'s
wraparound self-flush means unrelated drift could in principle compound over many widened splices in one
generation; flagged for completeness, not investigated further this session.

## 6. Exact next measurement required

Static analysis is exhausted for this boundary; the remaining hypothesis needs runtime GPU values, not
more source reading.

1. On the live CUDA host, dump the **raw conv_state values** (not just sum/sumsq -- those cannot
   distinguish "correct value, wrong slot" from "actually correct") for the specific active slot: once
   immediately after the widened step's `causal_conv1d_full` call, and once immediately before the next
   ordinary step's `causal_conv1d_update` call.
2. Because the ordinary step is served by CUDA-graph replay and bypasses `forward_embeds` entirely
   (`06-q`, Sec 3), the existing `ff_trace_gdn_state_checksum` diagnostic cannot observe it as-is. Either
   (a) temporarily disable CUDA decode graphs for one repro run (env/flag toggle, no source change) so
   the ordinary step falls back to eager `forward_embeds` and can be checksummed on both sides of the
   boundary, or (b) add a one-off host-side dump inside `capture_cuda_decode_graph_step`'s warmup call
   and `CudaDecodeGraphState::replay`'s launch, gated identically to the existing DEBUG diagnostic.
3. Compare against the equivalent position's conv_state in an FF-OFF sequential run (same prompt,
   same logical position, ordinary single-token decode only).
4. If conv_state already diverges at that boundary, the bug is confirmed to be in the
   `causal_conv1d_full` vs `causal_conv1d_update` kernel pair specifically (localizes to one of the four
   kernels named in Sec 5). If conv_state matches but the decoded token still diverges, the remaining
   hypothesis should shift to `recurrent_state`/the delta-rule kernels (`gated_delta_rule_recurrence_kernel_vmajor_grouped`,
   previously audited for in-place-aliasing safety in `06-p` but not re-verified this session for this
   specific cross-path handoff).

## 7. Repository state at end of session

- `/workspace`, branch `grammar-fast-forward`: dirty, one file changed
  (`mistralrs-core/src/vision_models/qwen3_5/text.rs`, the `06-q` checksum diagnostic), left uncommitted
  and unmodified this session. No other source touched; no instrumentation added.
- This report was written and committed from `/tmp/ff-artifacts-wt`, branch `ff-demo-artifacts`
  (worktree recreated this session after the registered worktree entry was found stale/pruned at session
  start -- `git worktree prune` then `git worktree add /tmp/ff-artifacts-wt ff-demo-artifacts` restored
  it cleanly against the existing branch tip, verified independent of `/workspace` via `readlink -f`
  before any write).

## Files added

- `plans/ff-round-two/reports/06-r-conv1d-dual-kernel-path-narrowed.md` (this report). No other files
  changed on this branch this session.
