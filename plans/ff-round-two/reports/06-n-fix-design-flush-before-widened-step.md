# 06-N -- fix design: flush GDN deferred state before a grammar-FF widened decode step (2026-09-10, eighteenth session)

Status: **fix design and comparison only. No source modified, no build.** Per explicit instruction,
this report identifies a recommended fix and its exact call site but does not implement it.

This follows `06-k` (candidate), `06-l` (cursor lifecycle trace), and `06-m` (confirmed the deferred-
decode path is active for this exact model: GGUF `ssm.state_size=128`,
`ssm.inner_size/ssm.time_step_rank=4096/32=128`, matching `GDN_DECODE_K_DIM`/`GDN_DECODE_V_DIM`
exactly). This report traces the existing GDN lifecycle APIs and compares three candidate fixes.

## 1. Existing GDN lifecycle APIs

| API | Location | Scope | Behavior |
|---|---|---|---|
| `reserve_recurrent_decode_deferred_storage()` | `text.rs:1499`, called once from `pipeline/multimodal.rs:1448-1449` | whole pipeline, load time | Allocates `deferred_state` if `deferred_decode_supported` (dims/dtype). One-shot, not per-step. |
| `disable_recurrent_decode_deferred_storage()` | `text.rs:1528` -> `hybrid_cache.rs:1740` | whole pipeline | Sets `deferred_state = None` for every GDN layer. Per `hybrid_cache.rs`'s own test
  (`deferred_state_can_only_be_disabled_before_slot_allocation`), only valid **before slots are
  allocated** -- a load-time toggle, not a per-step one. |
| `flush_current_recurrent_state()` | `text.rs:1718-1735` | whole current batch (`cache.state_indices()`) | Applies pending transition-log entries, then flushes `deferred_state` into `recurrent_state` for every active sequence. |
| `flush_recurrent_transitions_for_sequences(seq_ids)` | `text.rs:1737-1756` | specific sequence IDs | Same, scoped to the given sequences (`cache.recurrent_slots_for_sequences`). |
| `Pipeline::flush_recurrent_speculative_transitions(seq_ids)` | trait default (no-op `Ok(())`) at `pipeline/mod.rs:1836-1841`; overridden at `pipeline/multimodal.rs:2839-2843` and `pipeline/normal.rs:2567-2571` to forward to `self.model.flush_recurrent_speculative_transitions(seq_ids)` (`SpeculativeTargetMixin`, `text.rs:1739` for Qwen3.5) | specific sequence IDs, generic across any `Pipeline` | The one entry point already reachable from generic (non-model-specific) engine code. **Already used for a non-speculative-decoding purpose**: `pipeline/mod.rs:1939`, inside `snapshot_paged_recurrent_prefix` -- prefix-cache snapshotting calls it to make `recurrent_state` authoritative before reading it, exactly the class of need FF has. |

Both flush functions are idempotent and cheap when nothing is pending: `flush_deferred_recurrent_state`
early-returns on an empty slot list (`text.rs:1687-1689`), and the CUDA test
(`deferred_decode_matches_eager_across_wrap_and_flush_cuda`, `cuda/gdn.rs:10613-10938`) exercises
partial and full flushes without side effects on untouched slots. Post-flush, the cursor is confirmed
reset to 0 (`assert_eq!(deferred_cursor..., vec![0; CAPACITY])`).

## 2. Comparing the three candidate fixes

**Disable deferred decode around the widened step -- ruled out structurally.** The toggle
(`disable_recurrent_decode_deferred_storage`) can only run before slot allocation, per its own test's
name. There is no code path to flip it off for one step and on again for the next; doing this
"properly" would mean disabling the fast path for the pipeline's entire lifetime, for every sequence,
killing the performance benefit `deferred_state` exists for. Fails "preserves ordinary deferred-decode
behavior" outright, not just on cost grounds.

**Make the multi-token recurrence path consume/reconcile `deferred_state` itself -- architecturally
symmetric but the largest, riskiest change.** `causal_conv1d` (`gdn/backend.rs:823-833`) already has
exactly this shape of fix for `conv_state`, with an explicit comment: *"A fast-forward decode window
can carry more than one token; causal_conv1d_full handles arbitrary widths..."* (`06-l` Sec 2). The
symmetric fix for `recurrent_state` would extend `recurrence_cuda_from_convolved`'s `seq_len != 1`
branch (or `prepare_state_for_backend`) to fold `deferred_key`/`delta`/`decay` into the base state
before calling the prefill-style/chunked kernels (`try_fused_vmajor_prefill_recurrence_cuda`,
`chunked_gated_delta_rule_recurrence_cuda`, etc.) -- new numerical kernel code, likely needing parallel
work on the CPU/Metal backends too. Good long-term direction; too large a surface for "smallest,
behavior-preserving."

**Flush before the widened step -- recommended.** Reuses tested, already-existing machinery; touches
only per-step orchestration in `engine/mod.rs`; adds no new trait methods, no new kernels. Flushing
folds any pending deltas into `recurrent_state` and resets the cursor to 0, so the widened step's
direct advance of `recurrent_state` starts from a consistent base; since nothing writes into
`deferred_state` during the widened step itself, the state is left consistent (cursor 0,
`recurrent_state` current through the widened step) for the next ordinary step's fast path to resume
correctly.

## 3. Recommended fix: exact call site

`mistralrs-core/src/engine/mod.rs`, inside the `res = { let mut pipeline = get_mut_arcmutex!(self.pipeline); ... }`
block (currently starting at line 1860), right after acquiring `pipeline` and before whatever builds
`PagedAttentionMeta`/invokes the forward pass:

```rust
if let Some(_width) = pending_ff_width {
    let ff_seq_ids: Vec<usize> = guards_mut.iter().map(|seq| *seq.id()).collect();
    pipeline.flush_recurrent_speculative_transitions(&ff_seq_ids)?;
}
```

`pending_ff_width` is already computed at `engine/mod.rs:1821-1822`, before the pipeline lock, and is
`Some(w)` if and only if this step is about to build a widened window. Per
`resolve_pending_ff_batch`'s own discard-on-mismatch logic (`speculative/staging.rs:15-38`,
`staged_batch_state_from_widths`), a batch can only reach `Some` if *every* sequence in `guards_mut`
carries the same nonzero splice width -- a mix of splicing and non-splicing sequences forces `Mixed`,
which discards all splices in that batch first. So when this branch fires, every sequence in
`guards_mut` is one about to widen; no per-sequence filtering is needed.

Why this doesn't touch anything else:
- **Ordinary decode**: `pending_ff_width` is `None` on every non-FF step; the branch never runs.
- **FF-OFF**: `pending_ff_tokens` is never populated when the flag is off, so `pending_ff_width` is
  structurally always `None`.
- **Non-hybrid models**: `flush_recurrent_speculative_transitions`'s default is a no-op `Ok(())`
  (`pipeline/mod.rs:1836-1841`, `speculative/target.rs:144`).
- **Speculative decoding**: mutually exclusive with FF per sequence already (`inputs_processor.rs`
  bails if both `active_pending_ff_tokens` and `active_staged_speculative_len` are nonzero for the same
  sequence), so this call never fires for a sequence mid-speculative-commit.

## 4. Invariants to test once implemented (not run this session)

1. **Core empirical check**: rerun the exact N=1 `nested.schema.json` repro. FF-ON should produce
   `finish_reason=stop`, 47 tokens, byte-identical content to FF-OFF -- not just valid JSON.
2. **Position 33 unchanged**: re-run `06-f`-style instrumentation; the flush should not change the
   widened step's own already-correct sampled token. If it does, the model here is incomplete and needs
   revisiting before trusting the fix.
3. **Zero effect on ordinary decode**: the new flush call count should exactly equal
   `mistralrs_grammar_ff_splices_staged_total`'s delta, never more, never firing when
   `pending_ff_width == None`.
4. **Zero effect on FF-OFF and non-hybrid models**: no new code path is reachable; a smoke test on a
   plain (non-GDN) model with FF on is a good regression guard given the default-no-op contract.
5. **Speculative decoding untouched**: run a speculative-decoding session (no grammar) and confirm
   `flush_recurrent_speculative_transitions` call counts/timing are unchanged from today.
6. **Mixed-batch case**: confirm a batch with some FF-splicing and some ordinary sequences either never
   reaches this code with `pending_ff_width == Some` (per the discard-on-mismatch logic), or, if it
   somehow does, that the flush correctly scopes to only the intended sequence IDs.
7. **Error path sanity**: a flush failure (`?` propagating `candle_core::Error`) should surface as a
   normal step error, not panic or silently swallow.

## 5. Open point this fix does not resolve on its own

`06-k`/`06-l` still could not explain why the widened step's own output (`logical_position=33`,
token `763`) was already correct in the *current, unflushed* code, despite `forward_recurrent_core`
provably ignoring `deferred_state` entirely. This fix is derived from the traced mechanism regardless
of that open question, since flushing before a general-path read is unconditionally the correct
operation per the subsystem's own design contract (recurrent_state alone is only valid immediately
after a flush). But if applying it does **not** change position 34's output once tested (invariant 1),
that is a strong signal the real defect lies elsewhere -- most likely inside the multi-token CUDA
recurrence kernels themselves, or in an interaction with Qwen3.5's MTP/speculative head -- rather than
in the missing flush this report identifies.

## Files added

- `plans/ff-round-two/reports/06-n-fix-design-flush-before-widened-step.md` (this report). No other
  files changed; no source code touched or built.

## Git/worktree state

- `/workspace`: branch `grammar-fast-forward`, HEAD `cab8cd3ae`. Carries the accumulated local,
  uncommitted `tracing::debug!` instrumentation from `06-f`/`06-i`'s sessions; this report neither adds
  to nor removes it, and it is not part of this commit.
- This report was written and committed from `/tmp/ff-artifacts-wt`, branch `ff-demo-artifacts`,
  verified independent of `/workspace`.

Not implementing the fix or running any test this session, per instruction.
