# 04 -- Audit of the nine `RecurrentBatchKind::SpeculativeDecode` sites

Status: Deliverable 1 (audit table, reading only) complete. Deliverable 2 (CPU control-flow
confirmation via `tracing::debug!` and a live run) **not executed** -- see "Why Deliverable 2 was
not run" below. Deliverable 3 (CUDA experiment spec) written and marked blocked, as instructed.
**No branch was modified.** All source was read via `git show grammar-fast-forward:<path>` from the
`ff-demo-artifacts` worktree, without checking out or switching either worktree.

Branch read: `grammar-fast-forward`. Tip checked: `4ba40ca9e` (current tip per this session's git
log; the plan's own reference point is `8850efa93`, and the two commits between them --
`19da3d451` splice-accounting, `4ba40ca9e` AnyMoE gating -- do not touch any of the nine sites
below, confirmed by reading both diffs' file lists indirectly via the surrounding code shown here
matching the plan's cited line numbers).

## Why Deliverable 2 was not run

Deliverable 2 requires adding `tracing::debug!` instrumentation to nine call sites in
`mistralrs-core` and running a live CPU inference per recurrent family. Two independent blockers
apply in this session:

1. No Rust toolchain is available (`cargo`/`rustc` not on `PATH`), consistent with every prior
   report in this plan set (`01-baseline.md`, `02-lifecycle-audit.md`, `03-anymoe-divergence.md`).
2. This plan's operating rules for this session explicitly forbid modifying source code and
   forbid switching out of the `ff-demo-artifacts` worktree, both of which Deliverable 2 requires
   (it edits `mistralrs-core/src/` files on `grammar-fast-forward` and runs a built binary).

Both blockers are independently sufficient. Deliverable 2 is therefore **not run**, not failed --
no attempt was made. It remains exactly as specified in the plan, ready for a session with a Rust
toolchain and permission to commit to `grammar-fast-forward`. Because Deliverable 2 is the only
piece of this plan that produces empirical confirmation of "which arm ran," every "safe to share"
answer below is a **source-level** conclusion (reading the taken branch and arguing about it), not
a runtime-confirmed one, except where marked otherwise.

## Deliverable 1 -- the audit table

Legend for "Safe to share with an FF window?": **yes (already shared)** = the site does not
actually distinguish `Decode` from `SpeculativeDecode`, so an FF window already takes the same
branch a speculative window would, with no code change; **no (by design)** = the speculative arm's
assumptions do not hold for FF and the taken fallback is correct only because it is the general
path; **unknown -- needs CUDA** = the taken arm is proven correct only on CUDA and no CUDA build is
available here.

| # | Site | `RecurrentBatchKind` used | Can an FF window reach it? | What the `SpeculativeDecode` arm does | What the taken arm does instead | State contract the spec arm assumes | Safe to share? | Evidence |
|---|---|---|---|---|---|---|---|---|
| 1 | `gdn/layer.rs:532` (`forward_projected_with_context`, gating `forward_speculative_checkpoints`) | `== SpeculativeDecode` (exclusive, `checkpoint_lanes > 1` also required) | Yes -- every FF decode step is labelled `Decode`, never this variant | Runs a CUDA kernel that writes per-lane conv/recurrence *checkpoints* so a later verify step can roll any lane back independently | Falls to `forward_recurrent_core` (`causal_conv1d` + `apply_recurrence_from_convolved`), which commits state for the whole window sequentially and unconditionally | The window holds **proposals that may be rejected per-lane**, so each lane's state must be independently restorable | **no (by design), but correctness-safe**: `causal_conv1d` carries an explicit comment ("A fast-forward decode window can carry more than one token; `causal_conv1d_full` handles arbitrary widths") routing multi-token `Decode` windows to the same conv path prefill uses; `apply_recurrence_from_convolved`'s general branch is the same function prefill calls with `seq_len > 1`. Performance-only: no checkpointing means an FF window pays the general kernel instead of the fused speculative one. | Source-level (CPU path read in full; CUDA kernel body not inspected, but the dispatch decision and its CPU-path sibling are) |
| 2 | `vision_models/qwen3_5/text.rs:840` (`should_stash_gdn_replay`) | `== Some(SpeculativeDecode)` (exclusive, plus `query_len > 1`, `!native_speculative_commit`, `store_spec_hidden`, `continuation_without_cache`) | Yes | Returns `true`, causing a per-layer `GdnReplayStash` snapshot (conv + recurrent state) to be captured before `gdn_cache.commit(..)`, for later replay/rollback of speculative lanes | Returns `false`; no stash is captured; `gdn_cache.commit(..)` still runs unconditionally right after | The window's trailing tokens are **unverified drafts** that may need a byte-for-byte state replay later | **yes -- not a mismatch at all**: FF's own rollback need is handled one layer up, at the sequence/`llguidance` level (`discard_pending_ff_tokens`), and only ever applies to a splice **not yet fed to the model** (report 02: replay happens after the forward pass, and only a newly-staged splice for the *next* window can be discarded). No FF window is ever re-run through this layer after being committed, so no stash is needed here for FF's own correctness. This is not "conservative fallback," it is "correctly inapplicable." | Source-level |
| 3 | `vision_models/qwen3_5/text.rs:2321-2347` (`speculative_gdn` gating `transition_gdn`/`checkpoint_gdn`; adjacent `deferred_gdn` gated on `query_len == 1`) | `== SpeculativeDecode` for `speculative_gdn`; `== Decode` (+ `query_len==1`) for the separate `deferred_gdn` gate | Yes, for both gates | `transition_gdn`/`checkpoint_gdn` true: uses the transition-log or per-lane checkpoint path, deferring/logging state changes instead of committing them immediately | `!transition_gdn` with an active transition log: `apply_pending_recurrent_transitions_for_current_batch` eagerly applies pending transitions. `!deferred_gdn` with active deferred state: `flush_deferred_recurrent_state` eagerly materializes it. Both are the same "apply now" fallback the plan describes. | Deferred/logged state changes can be applied lazily because the window is provisional | **no (by design), correctness-safe**: eager application is exactly what non-speculative decode already does at `query_len==1`; FF only changes `query_len`, and the underlying recurrence math (site 1) is confirmed general. Performance-only. Round one's F3 (`docs/journal/.../fast-forward-development-plan.md`) is the `deferred_gdn` half of this row; it is recorded here, not redesigned -- owned by B1. | Source-level |
| 4 | `pipeline/normal.rs:2210` (`snapshot_hybrid_recurrent_checkpoints`, `transitions_supported`) | `== SpeculativeDecode` (exclusive) | Yes | If the model supports recurrent speculative transitions *and* the cache uses a transition log, skips snapshotting entirely (`Ok(None)`) -- rollback is handled by the transition log itself, cheaply | Calls `flush_recurrent_state_for_current_batch()` (eager materialize), then takes a **full recurrent-state snapshot per active slot index** so a failed CUDA-graph capture attempt can be rolled back | The transition log can undo a failed capture attempt cheaply; without it, the only way to undo is a full state copy taken beforehand | **no (by design), correctness-safe**: this is the exact same snapshot/restore mechanism already used in production for ordinary width-1 `Decode` CUDA-graph capture (this function is not new to FF; FF just makes it also fire at width > 1). Performance-only: extra full-state copy cost scales with active slot count on every graph-capture attempt during an FF window, not a new risk. | Source-level |
| 5 | `pipeline/multimodal.rs:1874` (`uses_nonmutating_recurrent_transition_log`, gating `snapshot_hybrid_recurrent_checkpoints`) | `== SpeculativeDecode` (exclusive) | Yes | Same as row 4, multimodal pipeline variant | Same as row 4: eager flush + full per-slot snapshot | Same as row 4 | **no (by design), correctness-safe**, same reasoning as row 4 -- identical structure, different pipeline struct. | Source-level |
| 6 | `pipeline/multimodal.rs:2233` (startup CUDA-graph precapture loop over `widths`) | Assigns `SpeculativeDecode` only when `q_len > 1 && speculative` | Windows with `q_len > 1`: **no** -- `widths` is built only from `self.model.speculative_graph_plans()` (real native-speculative/MTP proposal widths), never from grammar FF. An FF window's width is not a member of `widths` at all. | For a precaptured native-speculative width, labels the graph key `SpeculativeDecode` so it matches the runtime key built by `recurrent_batch_kind_for_input` when an actual speculative batch is staged | Nothing -- there is no FF-labelled entry in this loop to take an alternate branch. An FF-width decode step simply finds no precaptured graph (cache miss against `state.contains(&key)` at runtime) and falls to on-demand capture via row 4/5's mechanism, or eager execution if capture is disabled/fails. | Precapture only needs to cover widths that will recur; only real speculative-decode widths are assumed to recur at fixed width | **unknown -- performance-only but unmeasured**: this is not a branch FF "fails to reach," it is a startup list FF widths were never added to. Whether this matters depends on how often the same FF width recurs and whether on-demand capture (rows 4/5) amortizes it -- a question row 4/5 already show is at least correctness-safe, just not measured for cost here. | Source-level (mechanism only; no cost data) |
| 7 | `vision_models/mod.rs:184` (`text_decode_position_ids_from_context`, `matches!` guard) | `matches!(.., Some(Decode \| SpeculativeDecode))` -- **both variants in the same arm** | Yes, and it already lands in the same arm as `SpeculativeDecode` | (no separate arm -- see next column) | Both `Decode` and `SpeculativeDecode` call `crate::pipeline::decode_positions_tensor(..)` identically | None -- there is no distinct speculative-only assumption at this site | **yes -- already shared, not a mismatch**: this site does not gate on `SpeculativeDecode` exclusively; it groups `Decode` and `SpeculativeDecode` together against the third variant, `Prefill`. FF windows (`Decode`) already take the identical code path a speculative window would. This refines the second-round research entry's blanket claim ("every one of them takes its other branch") for this specific site: there is no "other branch" here. | Source-level, plus an existing unit test (`mrope_position_ends_are_decode_only`) exercising the shared arm |
| 8 | `vision_models/mod.rs:243` (`#[cfg(test)] mod tests`, `mrope_position_ends_are_decode_only`) | Constructs a context `.with_recurrent_batch_kind(RecurrentBatchKind::SpeculativeDecode)` | N/A -- this is test code, not a production dispatch site | Exercises row 7's shared function with `SpeculativeDecode` as the "decode-like" representative variant (the test's own name says "decode only", i.e. it is verifying decode-shaped behaviour, and picked `SpeculativeDecode` as one instance of that shared shape) | N/A | N/A | **not applicable**: same shape as report 02's finding about `scheduler.rs`/`default_scheduler.rs` test-module line numbers -- the plan's line citation lands inside `#[cfg(test)] mod tests` (module starts above line 209 in this file), not a live dispatch site. Nothing to classify as correctness- or performance-relevant because nothing here executes at inference time. | Source-level |
| 9 | `pipeline/cuda_graph.rs:2379` (`cuda_decode_graph_batch_kind_supported`) | `matches!(kind, Decode \| SpeculativeDecode)` -- **both variants in the same arm** | Yes, and it already lands in the same arm as `SpeculativeDecode` | (no separate arm) | Both `Decode` and `SpeculativeDecode` are "supported" for CUDA decode graphs; only `Prefill` is excluded | None | **yes -- already shared, not a mismatch**: same shape as row 7. FF windows are already eligible for CUDA decode graphs by this predicate; whatever cost FF pays for decode graphs is governed by rows 4/5/6 (capture/rollback and precapture coverage), not by this eligibility check. | Source-level |

### Summary of the nine-row classification

- **Genuine speculative-only gates where FF takes a conservative fallback (rows 1, 3, 4, 5):**
  performance-only by source-level argument. The fallback in every case is a code path already
  proven in production for a different (narrower or prefill-width) case -- not new, unverified
  logic invented for FF. No correctness issue found.
- **A gate whose absence for FF is correct, not conservative (row 2):** FF has no use for the
  capability the gate withholds, because FF's own rollback happens strictly before the model ever
  sees a token, not after.
- **A precapture list FF windows are simply never added to (row 6):** not a branch comparison at
  all; a startup-coverage gap. Performance-only, cost unmeasured.
- **Sites that do not actually distinguish `Decode` from `SpeculativeDecode` (rows 7, 9):** the
  premise "nine sites gate FF out" does not hold uniformly. These two already treat an FF window
  identically to a speculative one. This narrows, without disputing, the second-round research
  entry's New A framing.
- **Not a production site (row 8):** test-module code picked up by the plan's line-number citation.

No row's answer required marking `unknown` outright at the control-flow level -- every row's *taken
branch* is legible from source. What remains genuinely unknown is **CUDA numerical/cost
confirmation** for rows 1, 3, 4, 5 (marked in Deliverable 3 below), and **whether row 6's missing
precapture coverage costs anything measurable** (no experiment specified for this by the plan;
noted as an open question below).

## Deliverable 2 -- control-flow confirmation

**Not run.** See "Why Deliverable 2 was not run" above. No instrumentation was added to
`grammar-fast-forward` in this session, and no model was downloaded or executed. The plan's model
table (LFM2.5-230M, Qwen3.5-0.8B, Granite 4 hybrid) is unchanged and ready for the next session with
a toolchain; this report does not repeat it since it's already fully specified in
`04-recurrent-site-audit.md` on this branch.

One clarification for whoever runs it: rows 7 and 9 of the table above need no instrumentation to
resolve control flow -- they are already proven to take the same arm for `Decode` and
`SpeculativeDecode` by the `matches!` pattern itself, independent of any live run. Instrumenting
them would confirm nothing new; the CPU run should focus on rows 1-6 (the GDN/hybrid-cache sites),
where the taken branch genuinely differs by `RecurrentBatchKind`.

## Deliverable 3 -- the CUDA half, specified and marked blocked

**BLOCKED -- needs CUDA.** No CUDA-capable environment is available in this session, and none was
attempted. Cross-references: round one's F3, and B1/R1 in
`docs/journal/2026-09-09-fast-forward-development-plan.md`, all carry the same blocker.

Experiment spec (unchanged from what the plan already specifies, restated here for a single
reference point once a CUDA machine is available):

1. **Flag-off vs flag-on token equality**, on a hybrid-recurrent CUDA build (GDN family --
   Qwen3.5-0.8B is the smallest checkpoint that reaches `gdn/layer.rs` and
   `vision_models/qwen3_5/text.rs`), under a partially forcing grammar, greedy sampling, fixed seed,
   two separate OS processes (the `perf_flags` `OnceLock` means one process cannot exercise both
   settings -- same constraint report 03 documents for `MISTRALRS_GRAMMAR_FAST_FORWARD`).
2. **tokens/s and forward-pass count for both processes**, on a completion with a high forced
   fraction, to quantify what rows 1, 3, 4, 5's fallback costs relative to the speculative-checkpoint
   path it would otherwise take.
3. **Per-site arm counts from Deliverable 2's instrumentation** (once added), run on the same CUDA
   host, to confirm the CPU control-flow result (rows 1-6 take the fallback; rows 7-9 are
   non-issues) also holds on CUDA -- i.e. that no CUDA-only code path reintroduces a
   `SpeculativeDecode`-exclusive branch that the CPU build's `#[cfg(not(feature = "cuda"))]`
   sibling hides.

Do not attempt to emulate this. No numerical or performance result is reported for it in this
document because none was produced.

## What this audit could not establish

- **CUDA-kernel-level correctness of the fallback path**, for rows 1, 3, 4, 5. The CPU-path
  evidence (the `causal_conv1d`/`apply_recurrence_from_convolved` general branch is shared with
  prefill, which is proven at arbitrary `seq_len`) is source-level reasoning by analogy to the CPU
  sibling, not a proof that `recurrence_cuda_from_convolved` and the CUDA `causal_conv1d_full`
  kernel preserve the same guarantee. This needs Deliverable 3.
- **Whether row 6's precapture gap has a measurable cost.** No experiment for this specific
  question is in the plan; the plan's Deliverable 3 experiment (item 2 above) would surface it
  indirectly (forward-pass count includes any on-demand capture overhead) but does not isolate it.
  This is a candidate addition for a future plan, not something to build now.
- **Whether the taken fallback path is exercised at all on any currently-obtainable checkpoint.**
  No model was downloaded or run in this session (Deliverable 2 not executed), so "the fallback
  runs" is a source-level control-flow argument (the `if`/`else` structure and the guard
  conditions), not an observed trace.

## Recommended next action

Run Deliverable 2 (CPU control-flow confirmation) in a session with a Rust toolchain, scoped to
rows 1-6 only (rows 7-9 need no instrumentation per the note above). That is the cheapest remaining
step that turns this audit's source-level conclusions into confirmed ones, and it is a prerequisite
Question 2 in the second-round research entry names before any CUDA time is spent. No other action
is recommended from this session: this plan's explicit non-goal (do not modify any of the nine
branches) was honored, and nothing found here changes that instruction -- rows 1, 3, 4, 5 look like
correctness-safe performance costs, not defects, and rows 2, 7, 8, 9 are not mismatches at all.
