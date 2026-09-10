# 01 — Establish a working baseline

**Class:** prerequisite. **Blocks:** every other plan.

## Why this exists

`docs/journal/2026-09-09-fast-forward-development-plan.md` states, in its own words, that **none of
the ten commits `3f651cc86`..`8e4e21606` has ever been compiled** — no `cargo build`, no
`cargo test -p mistralrs-core`, no `cargo clippy`. The second-round research entry likewise says no
`cargo` invocation completes in the container it was written in, and `which cargo` in the container
these plans were written in returns nothing.

Every finding downstream is a claim about code that has not been shown to build. Instrumenting,
measuring or "fixing" that code before it compiles produces evidence about nothing.

## Step 1 — toolchain gate

```
cargo --version && rustc --version
```

If either is absent: **stop**. Do not proceed to plans 02–08. Write
`plans/ff-round-two/reports/01-baseline.md` recording the absence, and hand back. The only work
that may proceed without a toolchain is the *documentation* half of plan 08 and the audit-table
half of plan 04 (both are reading exercises), and they must be labelled as unverified-by-build.

## Step 2 — baseline build and test

On `grammar-fast-forward` at `8850efa93`, CPU defaults, no feature flags:

```
cargo build -p mistralrs-core
cargo test  -p mistralrs-core
cargo clippy -p mistralrs-core -- -D warnings
```

Record the exact command, the toolchain version, and the full output of any failure.

## Step 3 — expected fragile spots

The development plan names the two changes most likely to fail mechanically. Check these first if
the build breaks:

1. **`ff_test_sequence`** moved into `mistralrs-core/src/speculative/staging.rs`'s test module in
   `a3851faba` with its imports re-derived by reading, never compile-checked.
2. **`Sequence::discard_pending_ff_tokens(reason: &'static str)`** (`sequence.rs:1261`) gained its
   parameter in `83deb86ad`. Call sites: `paged_attention/scheduler.rs:1386` (`"preemption"`),
   `sequence.rs:1373` (`"realloc"`), and the `"batch_shape"` site in `speculative/staging.rs`.

Also confirm the three added tests compile and pass:
`completion_batches_reserve_slots_for_pending_fast_forward_splices`
(`paged_attention/scheduler.rs:4118`), `resolve_pending_ff_batch_discards_mismatched_splice_widths`
and `resolve_pending_ff_batch_keeps_equal_width_splices` (`speculative/staging.rs:150`, `:163`).

## Step 4 — repair rule

Only **mechanical** repairs are in scope here: missing imports, a changed signature at a call site,
a moved item's visibility, a clippy lint on new code.

**Do not change behaviour to make a build or a test pass.** If a test fails for a substantive
reason, record it in the baseline report as a finding and stop — a failing assertion on this branch
is a result, not an obstacle. If a compile error can only be resolved by choosing between two
behaviours, stop and hand back.

Commit any mechanical repairs as a single commit on `grammar-fast-forward`, message body listing
each repair and why it is mechanical.

## Deliverable

`plans/ff-round-two/reports/01-baseline.md` containing:

- toolchain versions and host (CPU, no CUDA assumed);
- each command run verbatim and its result;
- every repair made, with the file, the error it fixed, and the argument that it is mechanical;
- test summary: counts passed/failed/ignored, and the full text of any failure;
- an explicit statement of whether a CUDA or Metal build is available on this host, since plans 04
  and 06 need one (`paged_attn_supported()` is a compile-time `const fn` returning `false` on a CPU
  build, `utils/mod.rs:297-305`).

## Exit criteria

`cargo build`, `cargo test -p mistralrs-core` and `cargo clippy -- -D warnings` all pass on
`grammar-fast-forward`, and the baseline report exists. Anything less is a hand-back, not a
proceed-with-caveats.
