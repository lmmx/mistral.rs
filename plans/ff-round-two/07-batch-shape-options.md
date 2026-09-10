# 07 — Choose and specify one batch-shape improvement (New C, New B, D1)

**Class:** decide. **Blocked by:** 06 **and an explicit human decision after reading 06's report.**

## Do not start this plan on your own initiative

Nothing here is authorised by the research. All three options are unmeasured, they are alternatives
to each other, and at least one of them may turn out not to be worth building at all. If you have
arrived here without a human having read `reports/06-batch-shape.md` and named a direction, stop.

Once a direction is named, implement **exactly one** option, as its own commit series, with its own
tests. Do not bundle two.

## The three options

### C — truncate splices to the batch minimum

Not considered by any round-one entry. `Matcher::rollback` takes a token count, so a splice of
length K can be shortened to m by rolling back `K - m` tokens; the first m stay committed and stay
valid, because they were forced by the grammar independently of what follows.
`Sequence::discard_pending_ff_tokens` (`sequence.rs:1261-1279`) already performs the full-length case
of exactly this operation, and `Matcher::rollback` accepts any count, so a partial rollback simply
has no caller today.

A batch in which **every** sequence carries a splice — two concurrent JSON-schema requests at
different positions in their own grammars — can then be made homogeneous by truncating every splice
to `min(K_1..K_n)` instead of discarding all of them. Today that batch feeds nothing.

Shape of the change:

- `Sequence::truncate_pending_ff_tokens(&mut self, m: usize, reason: &'static str)` beside
  `discard_pending_ff_tokens`, calling `Matcher::rollback(K - m)` and reproducing the existing
  failure handling — `discard_pending_ff_tokens` sets `SequenceState::Error` when the rollback fails
  (`sequence.rs:1261-1279`), and truncation must do the same, not silently continue.
- `resolve_pending_ff_batch` (`speculative/staging.rs:59-68`) computes the batch minimum and calls
  truncation instead of discarding, **only** when the minimum is non-zero.
- Metrics: count truncations, and tokens retained vs tokens rolled back, so the effect is visible in
  the same place the drop rate is.
- Tests: unit-level, no GPU, mirroring the two existing tests at `speculative/staging.rs:150`
  and `:163` — a mismatched-width batch is truncated to the minimum rather than discarded; a batch
  containing a zero-width row is still discarded wholesale; a rollback failure sets the error state.

Costs one extra partial rollback per sequence per step on batches that currently pay a **full**
rollback per sequence per step, so arithmetic is unlikely to be the deciding factor. What it does
not help: a batch mixing constrained and unconstrained requests, where the minimum is zero — which
is precisely what workload C in plan 06 measures.

### B — make the scheduler splice-aware

`completion_batch_indices` (`paged_attention/scheduler.rs:547-579`) partitions on carries-a-splice
rather than on splice **width**.

The splice review's constraint 7 rules out grouping completion batches on splice width, and that
reasoning is sound — splice lengths are data-dependent, so grouping on width serialises the batch to
one sequence per step. It does **not** rule out the weaker predicate: preferring to co-schedule rows
that carry a splice at all, a partition into two groups rather than one group per observed width.

Risks that must be addressed in the specification, not discovered later: the round-robin cursor
exists for fairness, and a preference that keeps splice-carrying rows together can starve
splice-less rows. Any proposal must carry a fairness argument and a test that exercises it.

Note the corollary the round-one entries understate: even with D1's ragged windows implemented, the
round-robin cursor will keep mixing splice-carrying rows with splice-less ones, and a splice-less
row pins the useful width to zero unless the padding scheme handles it. B is therefore not strictly
an alternative to D1.

### D1 — ragged-width decode windows

**Owned by the development plan**, deferred and unstarted: padding kept out of the KV cache and out
of `slot_mapping`, with the window builder at `pipeline/inputs_processor.rs:1664-1670` relaxed.
Largest of the three. If the measurement points here, the deliverable of this plan is a **written
specification handed back**, not an implementation — D1 is a change to KV accounting and deserves
its own review cycle.

## Comparison the specification must contain

Whichever option is chosen, the commit series is preceded by a short design note answering, with
06's numbers cited:

| | C truncation | B scheduler preference | D1 ragged windows |
|---|---|---|---|
| Fraction of measured loss it recovers | | | |
| Cases it cannot help | | | |
| Touches window builder? | no | no | yes |
| Touches KV accounting / `slot_mapping`? | no | no | yes |
| Touches scheduling fairness? | no | yes | no |
| New failure modes | | | |

## Exit criteria

One option implemented (or, for D1, specified) with tests; the design note published to
`plans/ff-round-two/reports/07-decision.md`; the other two options recorded as not-chosen with the
number that ruled them out.
