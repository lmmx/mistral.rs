# 06-L -- GDN deferred-state cursor/buffer lifecycle trace around the transition (2026-09-10, sixteenth session)

Status: **static source trace only, as instructed. No source modified, no build, no benchmarks, no
experiments.** Traces the deferred-state cursor/buffer mechanics `06-k` left unresolved. This
narrows `06-k`'s open question considerably and finds one additional, precise piece of evidence
(an explicit code comment acknowledging grammar fast-forward's widened window) but ultimately
reduces to a single fact this session cannot determine from source alone: whether the deferred-decode
fast path is even active for this specific model/pipeline configuration. `06-k`'s claim is not yet
upgraded to confirmed; it is narrowed to one concrete, checkable precondition.

## 1. How the deferred-state ring buffer actually behaves (read from `cuda/gdn.rs`'s own test)

`deferred_decode_matches_eager_across_wrap_and_flush_cuda` (`cuda/gdn.rs:10613-10938`) runs 14
sequential single-token decode steps against a depth-4 (`GDN_DEFERRED_STATE_DEPTH = 4`) deferred-state
pool, comparing the deferred path's output against an independent "eager" ground-truth recurrence
(`fused_decode_recurrence_cuda`, which writes directly and fully into its own state tensor every
step) at every single step, **without ever calling an explicit flush** for the main loop's own
`deferred_state`/`deferred_cursor` until steps 8, 10, and 13 (deliberately, to test partial-depth
flushes). The cursor is asserted to cycle by simple modulo (`expected_cursor = (expected_cursor + 1) %
GDN_DEFERRED_STATE_DEPTH`, line 10810) -- it does not reset itself on reaching depth. Despite this,
`assert_close` on the deferred path's output passes at **every** step, including steps 4 through 7 and
9, 11, 12 -- i.e. steps where the cursor has already wrapped past depth at least once with no
intervening flush.

**Conclusion:** `cache.recurrent_state` alone can be arbitrarily far behind the sequence's true current
state, and the deferred-decode kernel path (`forward_deferred_decode`) still produces the exact correct
output every time, because it always combines `cache.recurrent_state` with whatever is currently in the
ring buffer (`deferred_key`/`delta`/`decay`, up to `deferred_cursor` entries) to reconstruct the true
current state on the fly. **`flush_current_recurrent_state` is not needed for the deferred path's own
correctness under continued single-token decoding.** Its only purpose is to materialize
`cache.recurrent_state` into a self-sufficient, currently-accurate tensor for some *other* consumer
that does not know how to combine it with the ring buffer -- i.e. exactly a caller like
`forward_recurrent_core`.

## 2. `causal_conv1d` already has explicit, deliberate FF-awareness -- but only for `conv_state`

`gdn/backend.rs:823-833`:
```rust
pub fn causal_conv1d(...) -> Result<Tensor> {
    let (_, seq_len, _) = x.dims3()?;
    // A fast-forward decode window can carry more than one token; causal_conv1d_full handles
    // arbitrary widths, so only fall to the single-token path when seq_len is actually 1.
    if matches!(batch_kind, RecurrentBatchKind::Decode) && seq_len == 1 {
        causal_conv1d_update(x, conv1d_weight, dims, cache)
    } else {
        causal_conv1d_full(x, conv1d_weight, dims, cache)
    }
}
```
This is new evidence this session, not present in `06-k`: whoever wrote this function was explicitly
aware that a grammar fast-forward decode window can carry more than one token, and routed the
**convolution** state (`cache.conv_state`, the short sliding-window buffer, a separate piece of state
from `recurrent_state`/`deferred_state`) through the width-generic `causal_conv1d_full` accordingly.
This confirms FF-awareness exists in this subsystem for at least one piece of GDN state.

**No equivalent comment, condition, or code exists for `recurrent_state`/`deferred_state`.** The
recurrence dispatch (`recurrence_cuda_from_convolved`, `gdn/backend.rs:428-554`) branches purely on
`seq_len == 1` vs not (line 451), with no reference to `batch_kind` at all in that branch and no
reference to `cache.deferred_state` anywhere in the function. For `seq_len != 1` (which includes both
genuine prompt prefill *and* an FF-widened decode window -- both hit the same code, since this
function cannot distinguish them), it calls one of several "prefill-style"/chunked recurrence kernels
(`try_fused_vmajor_prefill_recurrence_cuda`, `vmajor_prefill_gated_delta_rule_recurrence_cuda`,
`chunked_gated_delta_rule_recurrence_cuda`, `gated_delta_rule_recurrence_cuda`), every one of which
operates on `state_flat` sourced from `prepare_state_for_backend(cache, ...)`
(`gdn/backend.rs:754-781`), which reads `cache.recurrent_state` directly and has no
`deferred_state` parameter or reference at all -- confirmed by reading the full function body.

## 3. The cursor/state table for positions 31-34, as far as static reading can fill it in

| point | GDN path executed | `cache.recurrent_state` | `cache.deferred_state` cursor/contents |
|---|---|---|---|
| before pos 31 (ordinary step) | `forward_deferred_decode`, *if* deferred decode is active for this model (Sec 4) -- otherwise `forward_recurrent_core`'s internal `seq_len==1` eager branch | Stale by however many ordinary steps have elapsed since the last flush (never, in this repro) -- *if* deferred decode is active; otherwise always current | Holds up to 4 pending entries, cycling by modulo -- *if* active; otherwise the field is unused (stays whatever it was initialized to, likely still `Some` but never read) |
| after pos 31 | (same as above) | Unchanged (deferred path never writes it) *if active*; advanced to include token 31 *if not active* (eager path keeps it current) | Cursor advances by one (mod 4) *if active*; irrelevant otherwise |
| before widened pos 33 (the FF step) | N/A (not yet executed) | As above | As above |
| after widened pos 33 | `forward_recurrent_core` (confirmed, structurally forced by `seq_len == 2 != 1`) -> `recurrence_cuda_from_convolved`'s multi-token branch -> `prepare_state_for_backend` reads `cache.recurrent_state` directly, ignoring `deferred_state` entirely | **Directly advanced by this step to include both the backlog token (328) and the replayed splice token (5917)**, becoming the new authoritative value as of position 33 | **Untouched** -- not read, not cleared, not reset. Whatever was pending before the widened step (if deferred decode was active) remains sitting in the ring buffer, now referring to a `recurrent_state` baseline that no longer matches what the buffer's deltas were computed relative to |
| before ordinary pos 34 | `forward_deferred_decode`, if `cache.deferred_state.is_some()` (unchanged by the widened step) and the other gate conditions still hold | Correctly advanced through position 33 (per the row above) | Same stale contents as left behind after position 33's step -- now paired with a `recurrent_state` baseline that has moved out from under it |

The bottom-right cell is the crux: **if** the deferred-decode fast path is active for this model, then
position 34's step combines a *correct, freshly-advanced* `recurrent_state` with a *pending buffer*
that describes deltas relative to the *pre-widened-step* baseline -- a combination the deferred
kernel's math was never designed to receive, since in every other circumstance (ordinary sequential
single-token decode) the ring buffer's pending entries are always relative to whatever
`cache.recurrent_state` currently holds *at the moment they were written*, and nothing else ever moves
that baseline out from under them except an explicit, coordinated flush.

## 4. The one fact this session could not establish: is the deferred-decode fast path even active for this model?

`forward_deferred_decode` requires `deferred_decode_supported`, which requires (among other things)
`self.dims.head_k_dim == GDN_DECODE_K_DIM` and `self.dims.head_v_dim == GDN_DECODE_V_DIM`
(`gdn/layer.rs:271-282`), both hardcoded to `128` (`cuda/gdn.rs:18-19`). Qwen3.5's actual
`linear_key_head_dim`/`linear_value_head_dim` (`vision_models/qwen3_5/config.rs:242-243`) are read
from the model's GGUF metadata at load time -- this session found only test-fixture values (`16`, in
unit-test configs, `config.rs:470-486`), not the real `unsloth/Qwen3.5-4B-GGUF` model's actual values,
which live in the GGUF file itself on the host, not in this repository's source and not accessible to
this container. **This is a model-configuration fact, not a code-logic question, and static source
reading cannot resolve it.**

This determines which of two scenarios applies:
- **If `head_k_dim == head_v_dim == 128` for this model** (plausible -- these constants look
  purpose-built for exactly this model family, given the whole deferred-decode subsystem's naming and
  the reservation call's unconditional invocation at pipeline load for any hybrid model): the
  deferred-decode fast path is active throughout ordinary decode, `06-k`'s mechanism applies exactly as
  described, and this report's Sec 3 table is the accurate account of what breaks.
- **If not**: `deferred_decode_supported` returns false, `cache.deferred_state` is never populated
  (`reserve_recurrent_decode_deferred_storage` returns `Ok(false)` and never installs the pool,
  `text.rs:1511-1513`), every decode step -- ordinary and widened alike -- goes through
  `forward_recurrent_core`, and ordinary single-token steps use `recurrence_cuda_from_convolved`'s own
  internal `seq_len == 1` branch (`fused_decode_recurrence_cuda`, the same *eager*, always-correct
  recurrence the CUDA test in Sec 1 used as its ground truth), which writes `cache.recurrent_state`
  directly and completely on every call. In that scenario **there is no deferred-state staleness
  problem at all**, `06-k`'s mechanism does not apply to this repro, and the investigation would need
  to look elsewhere.

## 5. Single smallest runtime/config observation to resolve this

Either of two independent, minimal checks would settle Sec 4 (neither requires a raw KV/memory dump):

1. **Read the GGUF metadata directly**, on the host, for `unsloth/Qwen3.5-4B-GGUF`'s
   `linear_key_head_dim`/`linear_value_head_dim` values (or equivalently, whatever GGUF key
   `gguf/normal_config.rs` maps into `Qwen3_5Config`'s `linear_key_head_dim`/`linear_value_head_dim`
   fields) and compare against `128`. This requires no server run at all, just inspecting the already-
   downloaded GGUF file's metadata (e.g. via `gguf-dump` or equivalent) -- the cheapest possible check.
2. **One `tracing::debug!` line**, gated on `RecurrentBatchKind::Decode && seq_len == 1`, logging
   `cache.deferred_state.is_some()` and `cache.slots.is_some()` from inside
   `forward_projected_with_context` (`gdn/layer.rs:511` area) for a live ordinary decode step. If both
   are `Some`/`true`, the fast path is active and `06-k`'s mechanism is live; if either is `false`/`None`,
   it is not, and `06-k`'s candidate is ruled out for this specific repro.

Neither was run this session, per instruction to keep this a static trace only.

## Files added

- `plans/ff-round-two/reports/06-l-gdn-cursor-lifecycle-trace.md` (this report). No other files
  changed; no source code touched, built, or benchmarked.

## Git/worktree state

- `/workspace`: branch `grammar-fast-forward`, HEAD `cab8cd3ae`. Carries the accumulated local,
  uncommitted `tracing::debug!` instrumentation from `06-f`/`06-i`'s sessions; this report neither adds
  to nor removes it, and it is not part of this commit.
- This report was written and committed from `/tmp/ff-artifacts-wt`, branch `ff-demo-artifacts`,
  verified independent of `/workspace`.

Not proceeding to a fix or to further instrumentation this session, per instruction.
