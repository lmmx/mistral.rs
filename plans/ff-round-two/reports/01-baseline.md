# 01 — Baseline report

**Status: NOT RUN (toolchain unavailable). Not a build failure — the build was never attempted
because `cargo`/`rustc` do not exist on this session's `PATH`.**

## Step 1 — toolchain gate

```
$ cargo --version
bash: cargo: command not found

$ rustc --version
bash: rustc: command not found

$ which cargo rustc
(no output, exit 127)
```

Per plan step 1, this is a stop condition: do not proceed to plans 02–08.

## Host evidence

- Container: Debian GNU/Linux 12 (bookworm), `x86_64-unknown-linux-gnu`, 20 vCPUs, kernel
  `6.8.0-57-generic`.
- `/root/.cargo`: permission denied (not accessible to this session's user, `node`).
- `/home/node/.cargo`, `/usr/local/cargo`: do not exist.
- A stale `target/` build directory exists at the repo root (`/workspace/target/`, untracked —
  present in `git status` at session start). Its `target/.rustc_info.json` records a **prior**
  successful `rustc --version` invocation:
  ```
  rustc 1.98.0 (88d9e12ae 2026-08-18)
  host: x86_64-unknown-linux-gnu
  toolchain path: /home/louis/.rustup/toolchains/stable-x86_64-unknown-linux-gnu
  ```
  This shows a `rustc 1.98.0` toolchain was used to build this tree at some point under a
  different user (`louis`), not under the current session's `node` user. That toolchain is not on
  this session's `PATH` and its files were not located (rustup home not present for `node`, and
  `/root/.cargo` is inaccessible). **This is evidence a toolchain exists somewhere in this
  environment's history, not that this session can currently build.** I did not attempt to chase
  it down, per the instruction not to spend time provisioning or working around a missing
  toolchain.
- No CUDA/Metal build could be assessed for the same reason. Given the CPU-only host (no GPU
  devices apparent, no CUDA toolkit checked because it's moot without `cargo`), plans 04 and 06
  should assume a **CPU-only build** is the best case here unless the user's own dev environment
  has CUDA/Metal — that must be confirmed on the machine that actually runs the build.

## Steps 2–4 — not attempted

`cargo build -p mistralrs-core`, `cargo test -p mistralrs-core`, and
`cargo clippy -p mistralrs-core -- -D warnings` were **not run**. No mechanical repairs were made
or attempted, and none should be inferred from this report — nothing in `mistralrs-core` was
touched. This session made no commits on `grammar-fast-forward`.

## What to run in your development environment

On `grammar-fast-forward` (confirmed tip in this checkout: `8850efa93`, matches the plan's
expected tip), from the repo root, with a working Rust toolchain on `PATH`:

```bash
git checkout grammar-fast-forward
cargo --version && rustc --version

cargo build -p mistralrs-core
cargo test  -p mistralrs-core
cargo clippy -p mistralrs-core -- -D warnings
```

If any of these fail, capture the full output. Per the plan's repair rule: only mechanical
repairs are in scope (missing imports, a changed signature at a call site, a moved item's
visibility, a clippy lint on new code) — do not change behavior to make a build or test pass. If a
test fails for a substantive reason, that is a finding to report, not something to fix here. If a
compile error can only be resolved by choosing between two behaviors, stop and hand back rather
than choosing.

Specifically check the two fragile spots the plan names:

1. `ff_test_sequence` in `mistralrs-core/src/speculative/staging.rs`'s test module
   (moved there in `a3851faba`).
2. `Sequence::discard_pending_ff_tokens(reason: &'static str)` (`sequence.rs:1261`, parameter added
   in `83deb86ad`) and its three call sites: `paged_attention/scheduler.rs:1386`
   (`"preemption"`), `sequence.rs:1373` (`"realloc"`), and the `"batch_shape"` site in
   `speculative/staging.rs`.

And confirm these three tests compile and pass:
- `completion_batches_reserve_slots_for_pending_fast_forward_splices`
  (`paged_attention/scheduler.rs:4118`)
- `resolve_pending_ff_batch_discards_mismatched_splice_widths` (`speculative/staging.rs:150`)
- `resolve_pending_ff_batch_keeps_equal_width_splices` (`speculative/staging.rs:163`)

If mechanical repairs are needed, commit them as a single commit on `grammar-fast-forward` with a
message body listing each repair and why it is mechanical, then re-run this report's three
commands and update this file (or hand it back to this session) with the real results before any
of plans 02–08 proceed.

## Exit criteria — not satisfied

Plan 01's exit criteria (`cargo build`, `cargo test -p mistralrs-core`, and
`cargo clippy -- -D warnings` all passing on `grammar-fast-forward`) are **not satisfied**: none of
the three commands has been run. This is a hand-back per the plan's own rule ("Anything less is a
hand-back, not a proceed-with-caveats"). Plans 02–09 remain blocked.
