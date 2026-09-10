# 06-G -- static source trace of the two widened-to-ordinary transition candidates (2026-09-10, eleventh session)

Status: **static source trace only, as instructed. No source code modified, no build, no benchmarks,
no live requests.** Traces both candidates `06-f` named (`num_computed_tokens` bookkeeping and CUDA
decode-graph re-entry) against `/workspace` (`grammar-fast-forward` @ `cab8cd3ae`, untouched -- the
local instrumentation patch from the `06-f` session remains in the working tree but is unrelated to
and unread by this report). One candidate is ruled out definitively; the other is traced fully and
found internally consistent by construction, which does not confirm it innocent -- see Sec 4.

Reproduction and runtime facts (unchanged, not re-run): `06-b-n1-repro.md` through `06-f-runtime-
divergence-point.md`. The one fact this report depends on throughout: FF-ON's widened (`query_len=2`)
step at logical position 33 samples `763`, matching FF-OFF exactly; the very next step (logical
position 34, ordinary `query_len=1`) is the first divergence (`06-f` Sec 3-4).

## Candidate 1: `num_computed_tokens` bookkeeping at the transition

**What exact state changes after the width-2 FF step?**

`engine/mod.rs:1827-1830` captures `before = seq.num_computed_tokens()` for every scheduled sequence
*before* that step's forward pass and sampling run. `engine/mod.rs:1831-1858` computes `scheduled =
seq.num_uncomputed_tokens() + staged + pending_ff` -- for our repro's widened step, `pending_ff = 1`
(the staged splice), `staged = 0`, so `scheduled = 1 (backlog token "328") + 1 = 2`, matching the
observed `query_len=2`. After the step's pipeline call returns (`res` at `engine/mod.rs:1860`, which
runs the forward pass *and* sampling synchronously, including both `apply_pending_ff_tokens`'s replay
`add_token` call and the real sample's `add_token` call -- confirmed via `sequence.rs:1630-1689`,
`add_token` only pushes to `self.tokens`/`self.logprobs` and never touches
`self.num_computed_tokens`), `engine/mod.rs:2109-2117` runs:
```rust
if seq.num_computed_tokens() == before {
    seq.advance_num_computed_tokens(scheduled);
}
```
Since nothing in the intervening call touches the raw `num_computed_tokens` field (only `self.tokens`
grew, and `num_computed_tokens()`'s getter is `self.num_computed_tokens.min(self.len())`,
`sequence.rs:1443-1445` -- the clamp cannot change the read value here because `self.len()` only grew),
the guard passes and `num_computed_tokens` advances by exactly `2`, from `before` to `before + 2`.

**What exact code handles the next width-1 step?**

The very next engine loop iteration repeats the same sequence: `num_computed_before_step` is
re-captured (now `before + 2`), and `scheduled_token_counts` is recomputed as `seq.num_uncomputed_tokens
()` (`pending_ff`/`staged` are both `0` now, since the splice was consumed last step and none is
newly staged before this step's forward pass runs). `num_uncomputed_tokens() = self.len() -
self.num_computed_tokens()`. Both `self.len()` and `self.num_computed_tokens()` grew by exactly `2`
since the pre-widened-step snapshot, so this difference equals whatever `num_uncomputed_tokens()` was
*before* the widened step -- the ordinary steady-state value, `1`. This matches the observed
`query_len=1` at position 34 and is the normal "one new backlog token" pattern every ordinary decode
step has.

Separately, `paged_attention/scheduler.rs:540-545`'s `completion_token_cost` -- which the
`PagedAttentionScheduler` uses ahead of each step to size its block-allocation/token-budget request,
per `06-c`/`06-e`'s citation -- computes the identical formula (`num_uncomputed_tokens() + staged +
pending_ff`) from the same `Sequence` state, at scheduling time (before `inputs_processor.rs` builds
the window). Since it reads the same fields this report already traced as correctly advanced, it
would also compute `1` for the transition step, consistent with the engine-side value.

**Can this mechanism explain why position 33 is correct but position 34 is wrong?**

Not from what this report traced. The advance-and-consume arithmetic composes correctly across the
transition: `before -> before+2` after the widened step, then `(before+2) -> before+2` read as the
next step's own `before`, with `scheduled=1` correctly reflecting one backlog token. No off-by-one,
double-advance, or stale-read was found in this chain by static reading. This is a stronger version of
`06-c`/`06-e`'s citation of the same lines (which called it "consistent on paper" without deriving the
before/after values across two consecutive steps); this report derives the actual arithmetic and it
still holds.

One thing this report did **not** trace: whether the `PagedAttentionScheduler`'s actual **block
allocation** (not just its token-budget arithmetic) grows the sequence's allocated KV blocks/slots by
the correct amount in lockstep with this counter, specifically for a widened step. `completion_token_
cost`'s *inputs* were confirmed correct; whether the scheduler's block-allocation code consumes that
cost correctly (allocates 2 new slots' worth of space when told `2`, not something else) was not
followed further this session -- that is a distinct, deeper claim from "the counter arithmetic is
correct," and remains unverified.

**Exact lines/functions:** `engine/mod.rs:1827-1830` (`num_computed_before_step`), `engine/mod.rs:1831-
1858` (`scheduled_token_counts`), `engine/mod.rs:2109-2117` (guarded advance),
`sequence.rs:1443-1457` (`num_computed_tokens`/`num_uncomputed_tokens`/`advance_num_computed_tokens`
getters/setters), `sequence.rs:1630-1689` (`add_token`, confirmed not to touch the counter),
`paged_attention/scheduler.rs:540-545` (`completion_token_cost`).

## Candidate 2: CUDA decode-graph re-entry at the transition

**What exact state changes after the width-2 FF step?** Nothing relevant to this candidate --- see
below.

**What exact code handles the next width-1 step, and can it explain the divergence?**

Both mechanisms that could let a step bypass the ordinary eager forward-pass-plus-host-sampling path
gate on the identical function, checked fresh (not cached) on every call:

- The CUDA resident-sampling / decode-tail loop (`06-c`/`06-e`'s `account_cuda_decode_rows`/
  `continue_cuda_decode_batch` in `engine/mod.rs`) is only continued via
  `sampling::can_submit_cuda_token_batch_seqs` (`pipeline/execution.rs:352`).
- The forward-pass CUDA-graph lookahead launch (`pipeline/mod.rs:2380-2389`,
  `cuda_decode_lookahead`) requires, in the same boolean expression,
  `sampling::can_submit_cuda_token_batch_seqs(input_seqs)` (`pipeline/mod.rs:2385`) -- this report's
  one new fact beyond `06-e`: the *forward-pass* graph-replay path (which actually computes logits,
  as opposed to the GPU-resident *sampling* path `06-e` examined) is gated by the exact same function,
  not a separate, independently-behaving check.

`can_submit_cuda_token_batch_seqs` (`sampling.rs:1173-1193`) calls `cuda_token_sampling_plan`
(`sampling.rs:1032-1042`) per sequence, which returns `None` -- excluding the whole batch -- the
instant `!matches!(&seq.recognizer, SequenceRecognizer::None)` is true. Both call sites read
`seq.recognizer` live, off the current `Sequence`, on every invocation; there is no cached or
step-delayed copy of this check anywhere in either path.

Our sequence's recognizer is `SequenceRecognizer::Llguidance(_)` for its entire generation (from the
grammar's activation until it fully stops, which for this fixture is near the very end of a correct
completion -- well past position 34). This means **both** CUDA fast paths are excluded, identically,
on every single step of the generation -- before the splice, during it, and after it -- not
specifically or differentially at the widened-to-ordinary transition.

**Can this mechanism explain why position 33 is correct but position 34 is wrong?** No. FF-OFF's
sequence carries the identical `SequenceRecognizer::Llguidance(_)` state for its entire generation and
is excluded from both CUDA fast paths in exactly the same way at every step, yet produces a fully
correct 47-token completion. If CUDA-graph engagement were responsible for corrupting position 34
specifically, it would require the gate to behave *differently* for FF-ON's sequence than for FF-OFF's
at that one step -- but the gate is a pure function of `seq.recognizer`, identical on both sides at
every step of this repro. There is no code path by which this mechanism could fire only for FF-ON and
only at that one transition.

**Verdict: ruled out.** Not "untested" (as `06-e` had to leave it, before the CUDA-graph-exclusion
gate for the *forward-pass* path specifically had been checked) -- this report traces the forward-pass
gate directly and finds it identical in mechanism and behavior to the already-excluded resident-
sampling gate, uniformly excluding this whole class of sequence regardless of FF state or step
position.

**Exact lines/functions:** `pipeline/mod.rs:2380-2389` (`cuda_decode_lookahead`),
`pipeline/execution.rs:352` (`can_launch_cuda_decode_tail` gate, cited by `06-e`),
`sampling.rs:1032-1042` (`cuda_token_sampling_plan`), `sampling.rs:1173-1193`
(`can_submit_cuda_token_batch_seqs`).

## Ranking

1. **CUDA decode-graph re-entry: ruled out.** Traced to a single, shared, per-call gate that
   uniformly excludes this class of sequence at every step regardless of FF activity, on both legs of
   the comparison. Not a viable explanation for a transition-specific divergence.
2. **`num_computed_tokens` bookkeeping: not ruled out, but no defect found by static reading.** The
   counter arithmetic this report could trace (the engine-side advance-and-consume cycle, and the
   scheduler's token-budget formula that reads the same state) composes correctly across the exact
   transition in question. This is a *stronger* clearance than `06-c`/`06-e` could give (this report
   derives the actual before/after values across two consecutive steps; they only cited the lines as
   "consistent on paper"), but it does not cover the scheduler's actual block-allocation bookkeeping
   downstream of that arithmetic (Sec "Candidate 1", last paragraph), which remains untraced.

**Neither candidate can be statically confirmed as the cause.** Candidate 2 is eliminated; candidate 1
survives static reading but is not proven innocent, since one specific sub-mechanism (PagedAttention's
actual block/slot allocation, as opposed to its token-budget arithmetic) was not followed to its
conclusion this session.

## Minimal runtime observation that would move this forward

Since neither hypothesis was confirmed as the defect, and the PagedAttention scheduler's block-
allocation behavior (not just its cost arithmetic) is the one sub-mechanism candidate 1 leaves
unverified, the smallest useful next runtime check is at the KV-cache slot/block level, not the
token-count level `06-e`/`06-f` already instrumented:

- Log the actual physical slot index (or block id + block offset) the KV cache write for
  `logical_position=33`'s token (`763`) lands on, and separately, the slot index the very next step's
  forward pass (`logical_position=34`) reads back for that same logical position when building its
  attention context. If these disagree, the defect is in block/slot allocation or addressing at the
  transition (supporting a refined version of candidate 1); if they agree, the defect is likely below
  the Rust boundary entirely (the CUDA kernel body itself, `06-e`'s Sec 8 item 1, still unread by any
  session so far).

This was not run this session, per instruction to stop after the analysis.

## Files added

- `plans/ff-round-two/reports/06-g-transition-candidates.md` (this report). No other files changed;
  no source code touched, built, or benchmarked.

## Git/worktree state

- `/workspace`: branch `grammar-fast-forward`, HEAD `cab8cd3ae`. Still carries the `06-f` session's
  local, uncommitted `tracing::debug!` instrumentation in the working tree; this report neither adds
  to nor removes it, and it is not part of this commit.
- This report was written and committed from `/tmp/ff-artifacts-wt`, branch `ff-demo-artifacts`,
  verified independent of `/workspace` (`git worktree list`, distinct device/inode, consistent with
  every prior session's check).

Not proceeding further this session, per instruction.
