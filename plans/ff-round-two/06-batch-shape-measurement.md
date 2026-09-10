# 06 — Measure batch-shape behaviour (New B, and the sizing input for New C and D1)

**Class:** prove/disprove. **Blocked by:** 01, **02** (the counters must balance first), 05
(`concurrency` mode), and a CUDA or Metal build.

**This plan makes no design decision and changes no scheduling code.** It produces numbers and hands
them back. Plan 07 is where a choice is made, by a human, after reading this report.

## The finding being sized

`PagedAttentionScheduler::select_completion_batch` (`paged_attention/scheduler.rs:621-632`) picks the
decode batch through `completion_batch_indices` (`:547-579`). That function reads the splice-carrying
width of the row at the cursor as `active_staged_speculative_len()` and skips every row whose value
differs. It applies **no** equivalent test on `active_pending_ff_tokens().len()`.

With fast-forward on and no speculative proposer configured, `active_staged_speculative_len()` is 0
for every row, so the filter is a no-op: the scheduler admits rows on token budget alone, including
rows with differing splice widths and rows carrying no splice. One step later
`resolve_pending_ff_batch` (`speculative/staging.rs:59-68`) sees a non-homogeneous batch and discards
**every** splice in it, rolling each matcher back.

This is not a correctness bug — the discard is what makes it safe. It is an architectural gap:
batch composition and splice viability are decided in two places, the first with no knowledge of the
second and the second able only to say no.

Note also that `DefaultScheduler` admits sequences by count under `DefaultSchedulerMethod::Fixed(n)`
(`scheduler/default_scheduler.rs:301-304`), holds no token budget and has no preemption path — and
it is the scheduler the CPU GGUF demo ran under, not the paged one whose accounting the branch
changed. The 6.1–6.2x figure in `RESULTS.md` therefore says nothing about any of this.

## Why 02 is a hard prerequisite

Splices staged on a step whose sampled token ends the sequence are counted in the denominator and can
never appear in the numerator, so the drop ratio reads low until New E is fixed. Measuring first and
fixing after produces a number nobody can use.

## Dimensions to measure

No threshold is pre-committed, and none should be invented. Measure these six and report them:

1. **Splice discard rate by reason** — `mistralrs_grammar_ff_splice_drops_total{reason="batch_shape"}`
   against `mistralrs_grammar_ff_splices_staged_total`, with the other reasons broken out separately
   so `batch_shape` is not inflated by preemption or realloc.
2. **Forced tokens lost, not just splices lost** — via the `ff_tokens_dropped_total` counter added in
   plan 02, against `mistralrs_grammar_ff_tokens_fed_total`. A high splice-drop rate on short splices
   matters much less than a low one on long splices.
3. **Batch composition histogram** — per decode step, the count of rows carrying a splice vs rows
   carrying none. Report the fraction of steps that are all-splice, mixed, and none. This requires a
   small instrumentation addition (a histogram or a set of counters at the point
   `resolve_pending_ff_batch` inspects the batch); it is the one code change this plan permits,
   alongside 2's counter.
4. **Splice-width distribution, and per-batch minimum width.** Record `min(K_1..K_n)` per all-splice
   batch. This is the direct upper bound on what New C's truncation could recover, so measuring it
   here removes the need to build New C to find out whether New C is worth building.
5. **End-to-end effect** — tokens/s and total forward passes, flag on vs flag off, under each
   workload, so the drop rate can be read against what it costs.
6. **Complexity/risk inputs** for plan 07 — not measured, but recorded from the runs: whether any
   workload produced starvation or fairness anomalies in the round-robin cursor, and whether KV/slot
   accounting showed any pressure.

## Workloads

Sweep `N` in `{1, 2, 4, 8}` for each of:

- **A. Identical schemas.** N concurrent requests, same grammar, started together. The most
  favourable possible case: even here, span length tracks each request's own position in its own
  grammar, so agreement is not guaranteed.
- **B. Differing schemas.** N concurrent requests, each a different JSON schema from the fixture set.
  This is the shape that produces differing splice widths and the one the research names.
- **C. Mixed.** ~50% grammar-constrained, ~50% unconstrained. This is the case where the per-batch
  minimum is zero and New C cannot help by construction — measure how much of real traffic it is.

Run each workload with the flag on and with the flag off (two processes, per plan 05), and stagger
request start times in at least one variant so requests are at different grammar positions — a
synchronised start is an unrealistically favourable alignment.

## Deliverable

`plans/ff-round-two/reports/06-batch-shape.md`: the six dimensions, per workload, per N, as tables;
raw counter deltas attached as JSON; the exact build (CUDA/Metal, feature flags), model, and
harness invocations.

**Numbers only.** No recommendation, no ranking of B/C/D1, no "this suggests we should". Hand back.
