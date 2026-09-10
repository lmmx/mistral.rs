# 06-K -- strong root-cause candidate: GDN deferred-state lifecycle is not integrated with grammar fast-forward widened decoding (2026-09-10, fifteenth session)

Status: **strong root-cause candidate, not yet confirmed.** Static source trace only, as instructed.
No source modified, no build, no benchmarks, no experiments. This report deliberately does not claim
the bug is found -- see Sec 5 for the one load-bearing open question that must be resolved before this
can be upgraded from candidate to confirmed cause.

This follows a change in investigative direction after `06-i`/`06-j` found the KV-cache/PagedAttention
layer (address, block allocation, kernel indexing, stream ordering) fully consistent at every level
this investigation could reach. Per explicit instruction, this session set the KV-cache theory aside
and traced Qwen3.5's *other* mutable per-sequence state: the recurrent state carried by its Gated
DeltaNet (GDN) linear-attention layers, entirely separate from PagedAttention's KV cache.

## 1. Qwen3.5 is a hybrid attention/linear-attention architecture

`Qwen3_5TextModel` (`mistralrs-core/src/vision_models/qwen3_5/text.rs`, backing
`NormalLoaderType::Qwen3_5`, confirmed the loader this repro's GGUF pipeline actually instantiates)
alternates full-attention layers with `LayerImpl::LinearAttention(gdn)` layers implemented by
`GatedDeltaNet` (`mistralrs-core/src/gdn/layer.rs`). Each GDN layer carries its own per-sequence
recurrent state (`GdnLayerCache`: `conv_state`, `recurrent_state`, and -- the subject of this report --
`deferred_state`, `gdn/cache.rs:13-19`) that is independent of, and invisible to, everything `06-e`
through `06-j` examined (KV cache, PagedAttention block tables, slot mappings). This state exists
purely because of Qwen3.5's architecture; a plain transformer would have none of it.

## 2. The deferred-state fast path for ordinary decode

`deferred_state` is a small, fixed-capacity pending-update pool (`GDN_DEFERRED_STATE_DEPTH = 4`,
`cuda/gdn.rs:52`). Every ordinary width-1 decode step's forward call
(`GatedDeltaNet::forward_projected_with_context`, `gdn/layer.rs:511-557`) takes the
`forward_deferred_decode` branch (`gdn/layer.rs:543-554`) whenever `batch_kind == Decode && seq_len ==
1` (plus dtype/dims/pool-state conditions checked by `deferred_decode_supported`,
`gdn/layer.rs:271-282`). That path (`gdn/layer.rs:723-741`) computes its output from
`cache.recurrent_state` **combined with** the pending deltas in `cache.deferred_state`
(`deferred_recurrence_context`, `gdn/layer.rs:672-710`, passing both `state_pool: &cache.recurrent_state`
and `deferred_key`/`deferred_delta`/`deferred_decay`/`deferred_cursor` into the same kernel call). It
does not write the pending deltas back into `cache.recurrent_state` as part of this call.

`deferred_state` is reserved **unconditionally at model/pipeline load time**
(`pipeline/multimodal.rs:1448-1449`, `model.reserve_recurrent_decode_deferred_storage()?`, inside the
one-time pipeline-setup code, not per-request and not gated on speculative decoding or grammar FF being
in use for any given request) -- confirmed present whenever this model's dtype/dims satisfy
`deferred_decode_supported`'s checks, independent of what feature the current request happens to use.

## 3. The only reconciliation path belongs to a different feature entirely

The one function that folds `deferred_state`'s pending deltas back into `cache.recurrent_state`,
`flush_current_recurrent_state` (`text.rs:1718-1735`, and its per-sequence variant
`flush_recurrent_transitions_for_sequences`, `text.rs:1737-1756`), is exposed **only** through
`crate::speculative::SpeculativeTargetMixin` (`text.rs:2710-2742`,
`fn flush_recurrent_state_for_current_batch(&self) { self.flush_current_recurrent_state() }`). Grepped
every reference to `flush_current_recurrent_state` in the codebase: its only real caller is
`speculative.rs:1033`, itself part of the speculative-decoding target-model plumbing. Also grepped
`sampling.rs`, `inputs_processor.rs`, `engine/mod.rs`, and `speculative/staging.rs` (the entire grammar
fast-forward call chain traced in `06-c` through `06-j`) for any reference to `recurrent`/`deferred`:
the only hits are `flush_recurrent_speculative_transitions` (a *different* mechanism, the pending
"transition log" used for speculative-decoding checkpoint commits, called only from a hybrid
prefix-cache snapshot path, `sampling.rs:164`) and prefix-cache recurrent-state restore/reset code
(`engine/mod.rs`'s `stage_recurrent_reset`/`stage_recurrent_restore`, an unrelated paged-recurrent-
prefix-cache feature). **Nothing in the grammar fast-forward code path calls
`flush_current_recurrent_state`, `flush_recurrent_state_for_current_batch`, or
`disable_recurrent_decode_deferred_storage` at any point.**

## 4. What the widened FF step actually does to this state

The widened (`query_len=2`) FF step fails `forward_deferred_decode`'s `seq_len == 1` gate purely as a
structural side effect of having two tokens in its window -- there is no code that deliberately routes
FF steps away from the deferred path with awareness of what that implies for GDN state. It falls
through to `forward_recurrent_core` (`gdn/layer.rs:560-577`), which calls `causal_conv1d` and
`apply_recurrence_from_convolved` (`gdn/backend.rs:824`, `gdn/backend.rs:226`) -- both read directly
verified, in full, to contain **no reference to `cache.deferred_state` anywhere in either function
body.** They read and write only `cache.conv_state`/`cache.recurrent_state`.

This means the widened step advances `cache.recurrent_state` directly, with no participation from, or
acknowledgment of, whatever is sitting in `cache.deferred_state`'s pending buffer at that moment. It
does not clear, flush, or otherwise invalidate `deferred_state`. Immediately afterward,
`cache.deferred_state.is_some()` is still true (nothing disabled it), so the very next ordinary decode
step resumes the fast deferred path and combines the **just-advanced** `recurrent_state` with whatever
pending deltas were left over from **before** the widened step -- a combination that was never a
valid state for that kernel to receive, since the deltas and the base state no longer share a common
reference point.

## 5. The one open question this report deliberately does not resolve

**Why is the widened step's own output (`logical_position=33`, token `763`, confirmed matching FF-OFF
in `06-f`) correct, given that `forward_recurrent_core` ignores `deferred_state` entirely?** If
`deferred_state` held any genuinely pending (unflushed) deltas at the moment of the splice, the widened
step's read of `cache.recurrent_state` alone should have been stale and produced a wrong token too --
yet it did not.

The most likely explanation, not yet verified: the depth-4 pending buffer may reach a state where it is
fully reconciled (cursor back to empty) periodically as an ordinary side effect of the fast decode
path's own internal bookkeeping (e.g. an implicit flush-on-wraparound inside the CUDA kernel every
`GDN_DEFERRED_STATE_DEPTH` steps), and the splice may have landed at such a point by virtue of step
count, not because anything FF-aware ensured it. This is speculation stated as speculation -- this
session did not trace the cursor/wraparound mechanics closely enough to confirm or refute it. **Until
this is resolved, this report is a strong candidate, not a confirmed cause**, per explicit instruction
not to overclaim.

## 6. Why this is more compelling than the KV-cache candidates

`06-i` confirmed the KV block/slot addressing is correct at runtime; `06-j` found no per-row or
cross-call state in the vendored CUDA attention/cache kernels that could explain a transition-specific
defect. Both rule out mechanisms that would need to explain "the widened step's forward pass itself
must have read or written something wrong." The GDN candidate does not have that requirement: it
predicts the widened step's *own* output can be entirely correct (computed from a valid, if
soon-to-be-orphaned, `recurrent_state`) while the *state it leaves behind* is what corrupts the next,
structurally unrelated step -- matching the observed signature (position 33 right, position 34 wrong,
permanently) more precisely than any KV-address or kernel-indexing theory could, since those would need
to explain a defect localized to one specific forward pass rather than a state-lifecycle mismatch that
manifests one step later.

## 7. Next step (separate, not run this session)

Per explicit instruction, a narrowly scoped follow-up trace of the deferred-state cursor/buffer
lifecycle at exactly the five points around this transition (before/after position 31, before/after
the widened position 33 step, before position 34) is planned as a distinct next report. It is not
included here so this report's claim stays exactly as tight as the evidence gathered in this session
supports.

## Files added

- `plans/ff-round-two/reports/06-k-gdn-deferred-state-candidate.md` (this report). No other files
  changed; no source code touched, built, or benchmarked.

## Git/worktree state

- `/workspace`: branch `grammar-fast-forward`, HEAD `cab8cd3ae`. Carries the accumulated local,
  uncommitted `tracing::debug!` instrumentation from `06-f`/`06-i`'s sessions; this report neither adds
  to nor removes it, and it is not part of this commit.
- This report was written and committed from `/tmp/ff-artifacts-wt`, branch `ff-demo-artifacts`,
  verified independent of `/workspace`.

Not proceeding to a fix. Not proceeding to the cursor-lifecycle trace within this same report, per
instruction to keep this report's claim tightly scoped to what was actually established this session.
