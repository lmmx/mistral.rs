# 06-O -- implemented: flush GDN deferred state before a grammar-FF widened decode step (2026-09-10, nineteenth session)

Status: **fix implemented in `/workspace` (`grammar-fast-forward`), not built, not tested, not
committed to that branch.** Per `06-n`'s recommendation. No refactor, no GDN kernel changes, no other
behavioral changes. This container has no `cargo`/`rustc`, so the change is verified by hand against
the surrounding code, not by compilation -- building and running the `06-b`/`06-f` reproduction on the
host is the remaining step.

## 1. The change

`mistralrs-core/src/engine/mod.rs`, inserted after `scheduled_token_counts` is built (previously ending
at line 1858) and before the `res = { ... }` block that acquires the pipeline lock for the actual step:

```diff
+                        // A widened FF window bypasses GDN's deferred-decode fast path (seq_len != 1),
+                        // so flush pending deferred state first or the next ordinary step reads it
+                        // against an already-advanced recurrent_state.
+                        if pending_ff_width.is_some() {
+                            let ff_seq_ids: Vec<usize> =
+                                guards_mut.iter().map(|seq| *seq.id()).collect();
+                            let ff_recurrent_flush = get_mut_arcmutex!(self.pipeline)
+                                .flush_recurrent_speculative_transitions(&ff_seq_ids);
+                            handle_pipeline_forward_error!(
+                                "grammar fast-forward recurrent-state flush",
+                                ff_recurrent_flush,
+                                &mut guards_mut,
+                                self.pipeline,
+                                'lp,
+                                self.prefix_cacher
+                            );
+                        }
+
                         let res = {
```

18 lines added, nothing else touched -- exactly the call site `06-n` identified, using the existing
`Pipeline::flush_recurrent_speculative_transitions` entry point rather than any new trait method or
kernel change.

## 2. Implementation notes

- **Error handling reuses the existing convention exactly.** `handle_pipeline_forward_error!` is
  invoked here with the identical argument shape (`&mut guards_mut, self.pipeline, 'lp,
  self.prefix_cacher`) as its use on `res` two lines below (`engine/mod.rs`, pre-existing code) -- a
  flush failure gets the same per-sequence error response and cache reset as any other step-level
  failure, via `continue 'lp` inside the macro, rather than an unhandled `?`.
- **The gate is exactly `pending_ff_width.is_some()`.** Per `06-n`'s analysis of
  `resolve_pending_ff_batch`/`staged_batch_state_from_widths` (`speculative/staging.rs:15-38`), this can
  only be `Some` when every sequence in `guards_mut` carries the same nonzero pending-FF splice width,
  so no per-sequence filtering of `ff_seq_ids` is needed.
- **`guards_mut.iter().map(|seq| *seq.id())`** mirrors the pre-existing call shape three lines above it
  (`guards_mut.iter().map(|seq| seq.num_computed_tokens())`, used to build `num_computed_before_step`) --
  same iterator type, same auto-deref through `&&mut Sequence`, so this pattern is already proven to
  compile in this exact scope.
- No new `use` statements needed; `handle_pipeline_forward_error!` is already used later in the same
  function.

## 3. Build/test status: not possible in this container

No `cargo`/`rustc` is available here (`which cargo rustc` fails). The change was verified by hand
against the surrounding code (matching macro argument shapes and iterator patterns already in use a
few lines away, per Sec 2), not by compilation. **Building
(`cargo build --features "cuda flash-attn cudnn"`) on the host is required before this can be trusted
to compile, let alone fix the bug.**

## 4. No regression test added, and why

Per instruction, no test was invented since a meaningful CPU-only one could not be identified:
- `engine/mod.rs` has no existing `#[cfg(test)]` module at all; adding a harness from scratch for one
  18-line conditional inside a large async engine loop would be a disproportionate, novel addition.
- Making the gating logic independently unit-testable would require extracting it into a standalone
  function -- a refactor, explicitly out of scope for this task.
- The precondition this fix leans on (`pending_ff_width`'s batch-homogeneity semantics) already has
  coverage in `speculative/staging.rs` (`mixed_staged_widths_disable_batched_verification_input` and
  neighboring tests) -- no new coverage needed there.
- The behavior actually being fixed (GDN deferred-state reconciliation) exists only behind
  `#[cfg(feature = "cuda")]` and is only meaningfully exercised by a live Qwen3.5 model; the test that
  actually matters is the N=1 reproduction from `06-b`/`06-f`, which is inherently a live CUDA check,
  not a unit test.

## 5. What's left (not run this session)

Same as `06-n`'s invariant list:
1. Build, then rerun the exact N=1 `nested.schema.json` repro; FF-ON should now match FF-OFF
   byte-for-byte (`finish_reason=stop`, 47 tokens, identical content).
2. Confirm position 33's already-correct output is unchanged by the flush.
3. Confirm the new flush call count matches `mistralrs_grammar_ff_splices_staged_total`'s delta exactly.
4. Confirm zero effect on FF-OFF and on non-hybrid models.
5. Confirm speculative decoding (no grammar) is unaffected.
6. Confirm the mixed-batch case behaves as expected (or is structurally unreachable, per the
   discard-on-mismatch logic).
7. Confirm a flush failure surfaces as an ordinary step error, not a panic.

`06-n`'s open point still stands: if this fix does not change position 34's output once tested, that
would indicate the real defect is elsewhere (most likely the multi-token CUDA recurrence kernels
themselves, or an MTP/speculative-head interaction), not the missing flush this report implements.

## Files added

- `plans/ff-round-two/reports/06-o-fix-implemented.md` (this report, in the `ff-demo-artifacts`
  worktree). No other files changed in this worktree.

## Git/worktree state

- `/workspace` (`grammar-fast-forward`): the fix above is applied and **uncommitted** in the working
  tree, on top of `1adb03ea0` (a `chore: tracing implementation to debug FF splicing and KV
  management` commit made outside this session's actions -- likely the user's own host-side commit of
  the `06-f`/`06-i` debug instrumentation). Untracked `ff-off.log`/`ff-on.log`/`ff-on-slots.log` remain
  from earlier manual runs, unrelated to this change. Not committed to `grammar-fast-forward`; not
  built.
- This report was written and committed from `/tmp/ff-artifacts-wt`, branch `ff-demo-artifacts`,
  verified independent of `/workspace`.

Not building, testing, or committing the fix to `grammar-fast-forward` this session, per scope.
