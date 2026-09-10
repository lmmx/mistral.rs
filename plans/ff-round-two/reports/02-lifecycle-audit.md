# 02 — Lifecycle audit: where a staged fast-forward splice can be stranded

**Status: complete. Written before any of the code edits in this plan's Step 2/3.** Branch
`grammar-fast-forward`, tip `8850efa93` (matches `plans/ff-round-two/reports/01-baseline.md`'s
recorded tip; this session's manual baseline run separately confirmed `cargo build -p
mistralrs-core`, `cargo test -p mistralrs-core` (1368 passed, 0 failed, 5 ignored), doc-tests (2
passed, 0 failed, 4 ignored), and `cargo clippy -p mistralrs-core -- -D warnings` all pass at this
tip).

## The mechanism, restated precisely

Within one call to `sample_and_add_toks_inner`:

1. Any splice staged on a *prior* step is replayed first, via `apply_pending_ff_tokens` →
   `finish_or_add_toks_to_seq` per token (`pipeline/sampling.rs:849-859`). By the time a sequence
   reaches step 2 below, `seq.pending_ff_tokens` is guaranteed empty for it.
2. `sample_sequence` samples one token and, if the grammar can continue (`!llg.is_stopped() &&
   !ends_turn`), calls `llg.consume_token` and may stage a **new** splice for the *next* decode
   window (`sampling.rs:1614-1638`, staging itself at `:1630-1631`).
3. The very same sampled token is then passed to `finish_or_add_toks_to_seq`
   (`sample_and_add_toks_inner`, `sampling.rs:934`), which decides whether the sequence is done.

So exactly one splice can ever be present on a sequence about to terminate: the one staged in step
2 of the *same* call, for a token that step 3 may finish the sequence on. The three existing
discard call sites (`preemption`, `realloc`, `batch_shape`) all run on a sequence that is about to
continue running (return to `Waiting`, or run another decode step) — none of them fire when the
sequence instead becomes terminal in step 3. That is the whole defect: **no discard call exists on
the state transition into `Done(_)` or `Error`**, for any reason.

This also means the fix does not need to special-case *which* `StopReason` fires. Every reason
that can be produced at step 3 shares the same call chain, so one discard call per terminal-state
transition point closes all of them at once. That is reflected below: rather than a per-`StopReason`
table, the audit is a per-*call-site* table, and each `StopReason` is cross-referenced to the call
site(s) that can produce it.

## Two nuances the plan asked to resolve explicitly

**EOS sampled while `llg.is_accepting()` is false.** `ends_turn` (`sampling.rs:1616-1618`) requires
*both* `eos_tok.contains(token)` *and* `llg.is_accepting()`. But `Sequence::is_done`
(`sequence.rs:1737-1743`), called from `finish_or_add_toks_to_seq`, computes `StopReason::Eos` from
`eos_tok.contains(tok)` alone — it does not consult the grammar's acceptance state at all. The two
checks are independent, so `ends_turn == false` (grammar not accepting) does not imply
`StopReason::Eos` cannot fire: it can, for a token that carries whatever the model's tokenizer
config calls an EOS token id, if `llg.validate_tokens(&[token])` reports it as a valid grammar
continuation via the `None`-bias early-return at `sampling.rs:1544-1551` (i.e. the grammar accepts
the token as literal content without itself moving into an accepting state). This requires the
grammar's valid-continuation set to overlap the model's EOS token id — a real but narrow condition,
not exercised by `ff_bench.py`'s fully-forcing regex. **Verdict: reachable in principle by
construction of the code, not reproduced end-to-end in this session** (no toolchain to run a live
model). It funnels through `finish_or_add_toks_to_seq`'s ordinary `is_done.is_some()` path exactly
like every other `StopReason::Eos`, so it needs no special handling beyond the general fix below.

**`StopReason::ToolCalls`.** `Sequence::is_done` never produces `ToolCalls` directly. It is
produced two ways, both inside `finish_or_add_toks_to_seq`: (a) the early tool-completion check at
`sampling.rs:218-231`, which can set `is_done = Some(StopReason::Eos)` and call `seq.set_state`
directly *before* the function's general `is_done` handling runs, and (b) the streaming tool-call
parser at `sampling.rs:335-347`, which can set `is_done = Some(StopReason::ToolCalls)` from a
`None` starting point — i.e. a token that the grammar was still willing to continue past (hence
staged a splice) can *independently* be judged tool-call-complete by `ToolCallState`, whose
notion of "done" is separate from the grammar matcher's `is_stopped()`. This is exactly the
plan's suspicion: `ToolCalls` is the terminal reason most likely to coincide with a staged splice,
because the two "done" signals (grammar vs. tool-call parser) are not the same signal. Both paths
converge on the same `seq.set_state(Done(reason))` call sites the general fix targets (see below),
so `ToolCalls` needed no separate reason label — but it is the case that most motivated writing the
fix at the *call site*, not by `StopReason` variant.

## Site-by-site table

Legend: **STRAND** = confirmed reachable stranding path, now fixed. **HANDLED** = already
discarded before this plan (unchanged). **UNREACHABLE** = argued unreachable by construction;
no code change.

| Site | What happens | Verdict |
|---|---|---|
| `pipeline/sampling.rs:494` (`finish_or_add_toks_to_seq`, streaming `is_done.is_some()` branch — covers `Eos`, `Length`, `ModelLength`, `StopTok`, `StopString`, `ToolCalls`) | Sets `Done(reason)` with no discard. Reachable whenever a splice was staged for this same token at `sampling.rs:1630-1631` and the token also finishes the sequence. | **STRAND → fixed**: discard added immediately after `set_state`. |
| `pipeline/sampling.rs:481-483` (`finish_or_add_toks_to_seq`, streaming response-channel-closed branch) | Sets `Done(Canceled)` directly when `maybe_send_streaming_response` fails (client disconnected mid-stream), independent of `is_done`. | **STRAND → fixed**: discard added after `set_state`. |
| `pipeline/sampling.rs:506` (`finish_or_add_toks_to_seq`, non-streaming `is_done.is_some()` branch) | Same shape as the streaming branch, for non-streaming requests. `reason` is later shadowed to `ToolCalls` at `:599` for the response's `finish_reason` field only — the persisted `SequenceState` is set once, here, with the original `is_done` reason. | **STRAND → fixed**: discard added after `set_state`. |
| `pipeline/sampling.rs:225-227` (`finish_or_add_toks_to_seq`, early tool-completion check) | Sets `Done(Eos)` directly, ahead of the two branches above, when a required tool call completes. Control always continues into one of the two branches above afterward, which now discard. | **Covered transitively** — no separate edit needed; the discard downstream still runs on the same call. |
| `utils/mod.rs:100` (`handle_seq_error_stateaware_ok!`, used at `sampling.rs:311,522` for a delta/tokenizer decode failure inside `finish_or_add_toks_to_seq`) | Sets `Error` directly and returns early, bypassing the `is_done` branches above. Reachable on the same token that just staged a splice, if decoding that token's bytes then fails. | **STRAND → fixed**: discard added after `set_state`, inside the macro (both call sites are `&mut Sequence`). |
| `utils/mod.rs:253` (`handle_pipeline_forward_error!`, used at `engine/mod.rs:1343,1449,1953,1991,2035,2066,2085` for decode-step forward-pass failures; `:1310,1519` for prompt-step) | Sets `Error` on every sequence in the failed batch. A **decode**-step failure can hit sequences holding a splice staged on a *prior* successful step (not yet consumed, since consumption happens only after a successful forward pass, at the top of `sample_and_add_toks_inner`). | **STRAND → fixed**: discard added after `set_state`, for every `seq` in the loop. (Prompt-step call sites are additionally covered by the "no splice in prompt phase" argument below, so the fix is a no-op there, not a special case.) |
| `paged_attention/scheduler.rs:1237-1244` (`TERMINATE_ALL_NEXT_STEP` global-shutdown cancel, inside `schedule()`) | Iterates `self.running` (already-decoding sequences) and `prompt_running`, setting `Done(Canceled)` directly. No discard. | **STRAND → fixed**: discard added per sequence, after `set_state`. Not independently unit-tested — `TERMINATE_ALL_NEXT_STEP` is a process-global `AtomicBool` and flipping it in a test risks interference with the crate's parallel test run; covered by code review and by the identical, tested fix in `cancel_closed_response_groups` below, which shares the same bug shape. |
| `paged_attention/scheduler.rs:1442-1451` (`cancel_closed_response_groups`) | Sets `Done(Canceled)` on any running/waiting sequence whose response channel closed. No discard. | **STRAND → fixed**, and covered by a new test (`closed_response_cancellation_discards_a_staged_ff_splice`). |
| `scheduler/default_scheduler.rs:237-242` (`TERMINATE_ALL_NEXT_STEP`) | Same shape as the paged case, over `self.running`. `DefaultScheduler` has no token budget or preemption, but it **does** run sequences with `supports_grammar_fast_forward = true` — the CPU GGUF demo (`ff_bench.py`) runs under this exact scheduler (confirmed in `docs/journal/2026-09-09-fast-forward-second-round-research.md`'s "Current State" list, which states only that `DefaultScheduler` "needs no splice accounting" for *batch composition*, not that it never carries a splice). | **STRAND → fixed**, same reasoning/caveat as the paged `TERMINATE_ALL_NEXT_STEP` site above (global-flag test risk). |
| `scheduler/default_scheduler.rs:333-338` (`cancel_closed_response_groups`) | Same shape as the paged case. | **STRAND → fixed**, and covered by a new test (`closed_response_cancellation_discards_a_staged_ff_splice`). |
| `sequence.rs:1373` (`set_toks_and_reallocate`, reason `"realloc"`) | Already calls `discard_pending_ff_tokens("realloc")`. Plan-listed as already handled. | **HANDLED** — unchanged. |
| `paged_attention/scheduler.rs:1386` (`_preempt`, reason `"preemption"`) | Already calls `discard_pending_ff_tokens("preemption")`, after `set_state(Waiting)`. Plan-listed as already handled. | **HANDLED** — unchanged. Also the reference for get-order-right: this call already discards *after* the state transition, which is why the new call sites above follow the same order (see "Ordering" below). |
| `speculative/staging.rs:65` (`resolve_pending_ff_batch`, reason `"batch_shape"`) | Already calls `discard_pending_ff_tokens("batch_shape")` for every sequence when the batch's staged widths are not homogeneous. Not listed in the plan's "already handled" pair, but is the same shape and was already correct. | **HANDLED** — unchanged. |
| `engine/mod.rs:948` (`reject_prompt_for_cuda_memory`, sets `Error`) | Operates only on `scheduled.prompt` rows (prompt-phase). A sequence in prompt phase has never called `sample_sequence`, so it can never hold a staged splice — established for the general case in the second-round research entry's "Withdrawn after checking" section and re-verified here for this specific site by reading the function (it only ever receives rows from the prompt-step CUDA submission path). | **UNREACHABLE** — no change. |
| `engine/mod.rs:1540` (prompt-step `OneShot` sequences → `Done(GeneratedImage)`) | Also prompt-phase only (`scheduled.prompt`), and `OneShot` sequence types belong to diffusion/speech pipelines, which set `GeneralMetadata::supports_grammar_fast_forward = false` unconditionally (`diffusion.rs:252`, `speech.rs:330`). Doubly unreachable. | **UNREACHABLE** — no change. |
| `pipeline/response.rs:70,102` (`send_image_responses`/`send_speech_responses` → `Done(GeneratedImage/GeneratedSpeech)`) | Only called from the diffusion/speech/multimodal response-assembly paths (`pipeline/mod.rs:2193,2253,2839,2899`), all of which belong to pipelines with `supports_grammar_fast_forward = false` (`multimodal.rs:1531`, `diffusion.rs:252`, `speech.rs:330`). | **UNREACHABLE** — no change. |
| `pipeline/response.rs:129,151` (`send_raw_responses`/`send_embedding_responses` → `Done(Length(0))`) | `return_raw_logits` sequences never reach `sample_sequence` (return early from `send_raw_responses` before the `should_sample_step` gate, `pipeline/mod.rs:2683-2697,2719`; independently confirmed by the second-round research entry). Embedding pipelines set `supports_grammar_fast_forward = false` (`embedding.rs:706`) and never sample either. | **UNREACHABLE** — no change. |
| `paged_attention/scheduler.rs:1985,1990,2143,2202` and `default_scheduler.rs` test-module `set_state` calls the plan cites | On this tip, these line numbers land inside `#[cfg(test)] mod tests` (verified: `mod tests` starts at `scheduler.rs:1559`, all four call sites are past that). They are test fixtures, not production termination sites. | **Not a production site** — plan's line numbers reference test code at this tip; the corresponding real sites are the `cancel_closed_response_groups`/`TERMINATE_ALL_NEXT_STEP` rows above, which is what the two production `Canceled` sites in this file (`:1242`, `:1449`) actually are. |
| `SequenceState::FinishedIgnored` (`paged_attention/scheduler.rs:643`) | Set only when rejecting an oversized prompt at admission, before any decode step. | **UNREACHABLE** — no change. |
| `SequenceState::FinishedAborted` | Defined (`sequence.rs:165`) but not constructed anywhere in this codebase at this tip. | **UNREACHABLE** (dead variant) — no change. |

### `StopReason` variant cross-reference

| Variant | Producible via | Stranding call site(s) |
|---|---|---|
| `Eos` | `Sequence::is_done` (including the EOS-while-not-accepting case above); early tool-completion check | `sampling.rs:494` / `:506` |
| `Length`, `ModelLength`, `StopTok`, `StopString` | `Sequence::is_done` / `Sequence::add_token` | `sampling.rs:494` / `:506` |
| `ToolCalls` | Streaming tool-call parser (`sampling.rs:335-347`) | `sampling.rs:494` |
| `Canceled` | Scheduler force-cancel (`TERMINATE_ALL_NEXT_STEP`, `cancel_closed_response_groups`); streaming send failure | `sampling.rs:481-483`; both schedulers' two sites |
| `GeneratedImage`, `GeneratedSpeech` | Diffusion/speech response assembly only | Unreachable (pipelines that produce these never support fast-forward) |
| (n/a) `SequenceState::Error` | Sampling/decode error paths | `utils/mod.rs:100`, `:253` |

## Fix implemented (Step 2)

One reason constant, `"sequence_end"`, added at every **STRAND** site above — no material
distinction between the `StopReason` variants was found that would justify separate reasons (the
existing `batch_shape`/`preemption`/`realloc` reasons already distinguish the categories that
matter operationally; "the sequence finished" is one category). No staging condition at
`sampling.rs:1616-1631` was touched, and the three existing counters/reasons are unchanged.

**Ordering, and why it matters:** `discard_pending_ff_tokens` can itself set `SequenceState::Error`
if the llguidance rollback fails (`sequence.rs:1273`). Every new call site therefore sets the
terminal state *first* and discards *after* — matching the existing `_preempt` pattern
(`paged_attention/scheduler.rs:1382,1386`) — so that a rollback failure's `Error` is the last write
and is not clobbered by a `Done(reason)` written afterward. Getting this backwards (discard before
`set_state`) would silently turn a matcher-corruption failure back into a normal `Done` completion.
This is covered by `sequence.rs`'s new
`terminal_state_set_before_discard_is_not_clobbered` test (for the ordinary, non-failing rollback
case — `SequenceRecognizer::None` never fails a rollback, so the failure path itself is not
exercised by a unit test; no existing test in the crate exercises an actual failing
`Matcher::rollback` either, so this is consistent with prior coverage).

## Step 3: making the invariant checkable

Chose the token-terms invariant (plan's option), since plan 06 needs forced-tokens-lost, not
spans-lost: added `mistralrs_grammar_ff_tokens_dropped_total{reason}`, incremented by
`splice.len()` alongside the existing `mistralrs_grammar_ff_splice_drops_total` in
`discard_pending_ff_tokens` (`sequence.rs`).

State assertions (`active_pending_ff_tokens().is_empty()` after driving the sequence through a
termination path) were added at every site where the crate already has, or cheaply supports, a test
harness that does not require mocking `dyn Pipeline`:

- `sequence.rs`: `discard_pending_ff_tokens_clears_a_staged_splice`,
  `discard_pending_ff_tokens_is_a_noop_without_a_staged_splice`,
  `terminal_state_set_before_discard_is_not_clobbered` — the core invariant, independent of caller.
- `paged_attention/scheduler.rs`: `closed_response_cancellation_discards_a_staged_ff_splice`.
- `scheduler/default_scheduler.rs`: `closed_response_cancellation_discards_a_staged_ff_splice`.

**Not independently tested, with reasons:**
- `pipeline/sampling.rs`'s `finish_or_add_toks_to_seq` call sites and `utils/mod.rs`'s two macros
  require a `dyn Pipeline` to invoke; no mock or fake `Pipeline` implementation exists anywhere in
  this crate's test suite (checked: no `impl Pipeline for` under any `#[cfg(test)]` module), and
  building one is out of scope for a minimal accounting fix. These are covered by the code-path
  argument in the table above (they funnel into the same `set_state`-then-discard shape verified
  directly on `Sequence`) and by the unit tests on `discard_pending_ff_tokens` itself.
- Both `TERMINATE_ALL_NEXT_STEP` sites: flipping a process-global `AtomicBool` from a `#[test]`
  risks interfering with other tests in the same (parallel) test binary run; the identical bug
  shape is proven fixed by the `cancel_closed_response_groups` tests instead.

## Environment note

No Rust toolchain (`cargo`/`rustc`) is on this session's `PATH` (same finding as
`plans/ff-round-two/reports/01-baseline.md`). All analysis above is from reading the source; none
of it has been compiled or run in this session. The three commands below must be run in an
environment with the toolchain before this plan's exit criteria can be called satisfied:

```bash
cargo build -p mistralrs-core
cargo test  -p mistralrs-core
cargo clippy -p mistralrs-core -- -D warnings
```
