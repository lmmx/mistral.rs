# Grammar fast-forward tokens: minimal upstream-ready patch

## Context

sumac's journal (`lmmx/sumac` PR #12) prototyped a mistral.rs patch that lets a grammar-constrained
decode skip forward passes for tokens the grammar has already fully determined
(`llguidance::Matcher::consume_ff_tokens()`), by staging them and feeding a widened window into the
next decode step instead of one token at a time. It measured this as correctness-neutral
(byte-identical output) and, on a fully-forceable regex grammar, 6.4x faster; on sumac's own diluted
tool-call grammar, ~3%. That patch was written against v0.9.2 and never sent upstream. This repo
(`lmmx/mistral.rs`, a fork, currently at v0.9.3 / HEAD `d5ae0f18f`) is where we now build a real patch
for it, aimed at Qwen3.5 (GDN-hybrid, GGUF) since that's the sumac use case.

The goal is not to port the old patch mechanically. Two rounds of Explore-agent research against the
current tree found the old patch's premise partly obsolete and found existing plumbing we should
reuse instead of duplicating:

- **The GDN bugfix in the old patch is unnecessary now.** v0.9.2's GDN decode path hard-bailed on
  `seq_len != 1`. Current trunk already added a third `RecurrentBatchKind::SpeculativeDecode` variant
  (`pipeline/mod.rs:726-744`) specifically so verify-then-accept speculative decoding could push
  multi-token windows through GDN (`gdn/backend.rs`'s bail only fires for the plain `Decode` variant,
  not `SpeculativeDecode`; `causal_conv1d_full` and the general recurrence path are already
  multi-token-safe and unit-tested that way). We get free, already-proven GDN multi-token support by
  routing fast-forward windows through this existing kind instead of touching `gdn/backend.rs` at all.
- **The window-widening plumbing already exists for a different, adjacent mechanism**
  (`Sequence::staged_speculative_tokens` / `SpeculativeTokens`, consumed generically by
  `inputs_processor.rs`'s `make_completion_chunk`) but is verify-then-accept shaped: it always
  requests logits at every widened position (`context_lens.push((0, query_len))`, unconditionally),
  because a real speculative draft might be wrong. Grammar fast-forward tokens are *not* draft
  proposals — the grammar guarantees they're the only legal continuation — so there's nothing to
  verify, and we can (and should) narrow the lm_head/logit computation to just the one new position
  after the splice. That narrowing is where most of the "10x for regex" story actually comes from
  (skipping N-1 vocab projections, not just N-1 forward-pass dispatches), so it's the one thing worth
  a real (small) code change in `inputs_processor.rs` rather than reuse.

We deliberately do **not** reuse `staged_speculative_tokens` itself for storage — it's typed and
named for verify/accept semantics (`SpeculativeProposalDistribution`, device-tensor variant for GPU
draft models) that a reviewer would have to reason about for no reason if grammar-certain tokens rode
along in it. A new, narrowly-scoped `pending_ff_tokens: Vec<u32>` field keeps the two mechanisms
legible as separate concepts while still sharing the GDN dispatch path via `SpeculativeDecode`.

**What's genuinely new here vs. what already exists in the `llguidance` dependency:** `llguidance`
1.4.0 (already vendored, unchanged) has had `Matcher::compute_ff_tokens()` /
`Matcher::consume_ff_tokens()` (`matcher.rs:141-152`) all along — mistralrs's decode loop has simply
never called it. This PR's entire job is wiring an existing, already-tested dependency capability
through mistralrs's own sequence/decode-window/GDN-dispatch plumbing. No changes to `llguidance`
itself, no new grammar semantics — that framing should be the first thing in the PR description, so
the maintainer isn't left wondering whether this is a new grammar feature or a new dependency version.

## Design decisions and why (to state explicitly in the PR, not leave implicit)

1. **Reuse `RecurrentBatchKind::SpeculativeDecode` for FF-widened decode**, rather than adding a new
   enum variant or relaxing `RecurrentBatchKind::Decode`'s single-token invariant. Relaxing `Decode`
   itself is riskier: CUDA graph capture (`pipeline/cuda_graph.rs`) and other call sites may assume
   `Decode` always means `seq_len == 1`. `SpeculativeDecode` already means "more than one
   already-committed-or-proposed token this step" to GDN, which is exactly what we need, at the cost
   of one known, explicitly-flagged caveat (below).
2. **Known caveat to flag, not hide:** on CUDA, `SpeculativeDecode` also engages a
   speculative-checkpoint fast path in `gdn/layer.rs` (`forward_speculative_checkpoints`,
   checkpoint-lane bookkeeping meant for draft/verify rollback) that fast-forward windows don't need
   and this PR does not measure. Mitigated by scoping `supports_grammar_fast_forward` to
   `NormalPipeline` only and defaulting off everywhere (see below) — the PR description should say
   plainly that CUDA interaction with that checkpoint path is unverified and ask the maintainer (or
   whoever enables the flag on CUDA) to check it, rather than asserting it's fine.
3. **Opt-in via env var, not a default flip** — same reasoning sumac's own journal already worked
   out: the payoff is grammar-shape-dependent (near-zero to 6.4x), so shipping it as an unconditional
   default bets on every caller's grammar and hardware. `MISTRALRS_GRAMMAR_FAST_FORWARD=1`, checked
   once via `OnceLock`, matches the existing `perf_flags.rs` convention (see file plan below) rather
   than inventing a new toggle style.
4. **Wire the flag for `NormalPipeline` only** (the only pipeline whose input builder consumes
   `pending_ff_tokens` at all) — the other 6 `GeneralMetadata` construction sites get a hardcoded
   `false`, with a doc comment on the field itself stating that's a correctness requirement (those
   pipelines build their own inputs and never read the field), not just unmeasured caution.

## File-by-file plan

**`mistralrs-core/src/sequence.rs`** — add, right after `staged_speculative_distribution` (line
~783): a `pending_ff_tokens: Vec<u32>` field, initialized to `Vec::new()`, plus four small methods
mirroring the existing staged-speculative accessors exactly in shape
(`active_pending_ff_tokens(&self) -> &[u32]`, `set_pending_ff_tokens`, `take_pending_ff_tokens`) —
no distribution, no device variant, since these tokens need no verification and no CUDA-tensor path
for v1. One-line doc comment distinguishing this from `staged_speculative_tokens`: grammar-certain,
always accepted, no distribution.

**`mistralrs-core/src/perf_flags.rs`** — add a third flag alongside `cuda_graphs_enabled` /
`flashinfer_decode_enabled`, following that file's existing `const ..._ENV` + `static
OnceLock<bool>` + `env_flag()` pattern exactly:
```rust
const GRAMMAR_FAST_FORWARD_ENV: &str = "MISTRALRS_GRAMMAR_FAST_FORWARD";
static GRAMMAR_FAST_FORWARD_ENABLED: OnceLock<bool> = OnceLock::new();
pub(crate) fn grammar_fast_forward_enabled() -> bool {
    *GRAMMAR_FAST_FORWARD_ENABLED.get_or_init(|| env_flag(GRAMMAR_FAST_FORWARD_ENV, false))
}
```
(default `false`, unlike the other two flags which default `true` — this one is unmeasured on this
repo's own CI/hardware matrix, sumac's journal only measured one CPU container and one model).

**`mistralrs-core/src/pipeline/mod.rs`** — add `pub supports_grammar_fast_forward: bool` to
`GeneralMetadata` (after `loaded_for_uqff_write`, matching that field's plain `//` comment style, not
a `///` doc comment — see explored convention). Update all 7 `GeneralMetadata { ... }` literals
(`normal.rs`, `gguf.rs`, `embedding.rs`, `ggml.rs`, `diffusion.rs`, `multimodal.rs`, `speech.rs` —
each has a differently-ordered field list, so each needs its own edit, matched by field name not
position). Six sites get `supports_grammar_fast_forward: false,` with a one-line comment pointing at
the field's own doc comment. `normal.rs`'s site gets
`supports_grammar_fast_forward: crate::perf_flags::grammar_fast_forward_enabled(),`.

**`mistralrs-core/src/pipeline/inputs_processor.rs`**, `text_models_inputs_processor::make_completion_chunk`
(currently lines 1525-1740):
- Add a `pending_ff_batch_width`-style homogeneous-width helper mirroring
  `speculative::staging::staged_batch_width` (all-or-none across the batch; a mixed batch falls back
  to no widening for that step, matching the existing staged-speculative fallback behavior).
- In the per-sequence loop: read `seq.active_pending_ff_tokens()`, extend `ctxt` with it (after the
  existing staged-speculative extension — the two are mutually exclusive per sequence in every
  pipeline that will set either field, but the code shouldn't assume that silently; assert or bail if
  both are non-empty for the same sequence rather than silently picking one).
- Track the **full** widened length separately from the **narrowed** logit-selection length: when a
  sequence's width > 1 comes from `pending_ff` alone (no staged speculative on that sequence),
  `context_lens.push((query_len - 1, 1))` instead of `(0, query_len)` — only the position after the
  forced splice needs a sampled logit. Keep a separate `full_query_lens: Vec<usize>` (full width,
  including the forced splice) for `DecodePagedRows`'s `query_len` (KV-cache writes need every
  position), independent of the narrowed `context_lens` used for hidden-state/lm_head selection.
- **Verify at implementation time, not assumed now:** confirm how `InputMetadata.context_lens`
  tuples are actually consumed downstream (hidden-state gather before lm_head) in the *current*
  forward-pass code — this file had a 1298-line diff since v0.9.2 so the old patch's assumption about
  `(skip, take)` semantics needs re-confirming against current code, not copied blind. If the
  narrowing turns out more invasive than this shape suggests, ship v1 without it (still gets the
  forward-pass-count reduction and GDN dispatch amortization) and land narrowing as a clearly-labeled
  follow-up once confirmed.
- One-line change at the `recurrent_batch_kind_for_input` call site (~line 2760): the
  `has_staged_speculative_batch` argument becomes `staged_batch_width(..).is_some() ||
  pending_ff_batch_width(..).is_some()`.

**`mistralrs-core/src/pipeline/sampling.rs`**:
- `sample_sequence` (currently lines 1381-1536): add a trailing `supports_fast_forward: bool` param.
  After the existing `llg.consume_token(...)` block (lines 1511-1522), when `supports_fast_forward &&
  !llg.is_stopped()`, call `let splice = llg.consume_ff_tokens();` (single call — already
  compute-and-consume in one step, per `llguidance::Matcher`'s existing API, no manual
  `compute_ff_tokens` + `consume_tokens` two-step needed) and `seq.set_pending_ff_tokens(splice)` if
  non-empty.
- New helper modeled directly on the existing `finalize_block_gen` (lines 686-718, which already does
  "replay a Vec<u32> of pre-known tokens through `finish_or_add_toks_to_seq`, stopping early via
  `seq.is_running()`") — same shape, one sequence's splice instead of a per-seq block.
- `sample_and_add_toks_inner` (lines 783-836): capture `seq.take_pending_ff_tokens()` for every
  sequence *before* the `sample_sequence` calls (which may stage a new splice for the *next* step
  onto the same field), then after sampling each next token, replay the captured splice first (via
  the new helper) and skip applying the freshly-sampled token if the replay finished the sequence.
  `metadata.supports_grammar_fast_forward` (already in scope as `metadata`) becomes the new trailing
  arg to `sample_sequence`.
- Other `sample_sequence(` call sites — `speculative/driver.rs:278`, `speculative/verifier.rs:890`
  and `:954`, plus the `#[cfg(test)]` helper at `sampling.rs:1680` — all pass `false` for the new
  param (speculative decoding and tests are out of scope for this mechanism, matching how they
  already pass fixed `false`/`false`/`false` for the other two trailing bools).

## Commit staging (one PR, three commits, in review order)

1. **Plumbing, inert by default**: `sequence.rs` field/methods, `perf_flags.rs` toggle,
   `GeneralMetadata` field + all 7 construction sites. Compiles, changes nothing (flag off
   everywhere, field never populated).
2. **Decode-window + GDN dispatch reuse**: `inputs_processor.rs` changes. Still inert — nothing sets
   `pending_ff_tokens` yet, so this is dead code until commit 3, but reviewable on its own as "this is
   how a forced splice would be windowed, reusing the existing speculative-decode GDN path."
3. **Wire it up**: `sampling.rs` changes — this is the commit that actually makes the flag do
   something.

Keeps each commit's diff readable in isolation and lets the maintainer stop reading after commit 1 or
2 if that's as far as they want to reason about right now.

## Demo (not part of the PR)

Adapt `mistralrs/examples/advanced/llguidance` (existing example, already in the repo, already shows
the grammar-constraint API) into a standalone scratch script — same shape as sumac's own
`ff_bench.py`: a regex grammar matching one fixed string end-to-end, `enable_thinking=False`,
`temperature=0.0`, timed with the env var unset vs. `1`, asserting byte-identical output both ways.
Keep it in the scratchpad, not committed, not added to `examples/` — it's evidence for the PR
description's claim, not part of the diff. Report median wall time and forward-pass call count
(reuse the sumac journal's `FF_PROBE`-style instrumentation approach, temporarily, reverted after
measuring) for both settings.

## Execution steps

1. `git checkout -b grammar-fast-forward` (new local branch off current `master`/HEAD — leaves
   `master` untouched, nothing pushed).
2. Implement commit 1, `cargo check -p mistralrs-core`, commit locally (message describing just that
   slice) — **do not push, do not open a PR** without a fresh, explicit go-ahead per this repo's
   `CLAUDE.md` rule.
3. Implement commit 2, `cargo check -p mistralrs-core`, verify the `context_lens` narrowing question
   above against current forward-pass code before finalizing that part; commit.
4. Implement commit 3, `cargo check -p mistralrs-core` and `cargo clippy -p mistralrs-core --tests --
   -D warnings`; commit.
5. Build a release wheel (or CLI binary) from this branch, run the demo script twice (env var unset /
   `1`) against a small local GGUF model, confirm byte-identical output and report the timing/call-count
   delta.
6. Report results back before drafting any PR description or touching git remotes.

## Verification

- `cargo check -p mistralrs-core` and `cargo clippy --workspace --tests --examples -- -D warnings`
  after each commit (per this repo's own CLAUDE.md testing rule).
- `cargo test -p mistralrs-core` — existing GDN/speculative unit tests (e.g.
  `gdn/backend.rs`'s `causal_conv1d_full` vs `causal_conv1d_update_cpu` equivalence tests,
  `vision_models/qwen3_5/text.rs`'s `SpeculativeDecode`-width tests) should be untouched and still
  pass, since this PR doesn't modify `gdn/backend.rs` at all.
- The demo script's on/off byte-identical-output check is the correctness gate; the timing numbers
  are the "why this is worth merging" evidence for the PR description, kept separate from the diff.
