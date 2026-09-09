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

The goal is not to port the old patch mechanically, but to design something small and legible enough
that a maintainer can review it without re-deriving every assumption behind it. Research against the
current tree (two rounds of Explore-agent research, plus direct verification of the points below)
found:

- **`llguidance` 1.4.0 (the existing, unchanged dependency) already exposes this capability.**
  `Matcher::compute_ff_tokens()` / `Matcher::consume_ff_tokens()` (`matcher.rs:141-152`) have been
  there all along; mistralrs's decode loop simply never calls them. This whole PR is wiring an
  already-tested dependency capability through mistralrs's own sequence/decode-window/GDN-dispatch
  plumbing — no grammar semantics change, no dependency bump. That framing belongs first in the PR
  description so the maintainer isn't left wondering whether this is a new grammar feature.
- **GDN's decode path currently hard-bails on more than one token.** `causal_conv1d`
  (`mistralrs-core/src/gdn/backend.rs:824-840`) bails with "GDN decode expects a single-token query"
  whenever `RecurrentBatchKind::Decode` carries `seq_len != 1`. A fast-forward window needs to feed
  several already-known tokens through decode at once, so this bail has to be relaxed. The fix is
  narrow and already proven safe by existing code: `causal_conv1d_full` (the non-decode branch) is
  already exercised at arbitrary widths by prefill and by real speculative decoding's
  `RecurrentBatchKind::SpeculativeDecode` path, so it's just a matter of letting `Decode` fall through
  to it at width > 1 instead of bailing.
- **There is an existing, adjacent, but importantly different mechanism** —
  `Sequence::staged_speculative_tokens` / `SpeculativeTokens`, consumed generically by
  `inputs_processor.rs`'s `make_completion_chunk`, and a `RecurrentBatchKind::SpeculativeDecode`
  variant added upstream specifically so verify-then-accept speculative decoding could push
  multi-token windows through GDN. It's tempting to route fast-forward through this existing
  machinery to avoid touching `gdn/backend.rs` at all, but that would be borrowing more than it looks
  like: real speculative decoding is propose-then-maybe-reject, and GDN's recurrent state isn't
  cheaply truncatable the way a KV cache is — once a batched call folds N tokens into the SSM state
  you can't cheaply un-fold some of them if they're later rejected. That's why `gdn/layer.rs` has a
  CUDA-side checkpoint-lane mechanism (`forward_speculative_checkpoints`) gated on
  `batch_kind == RecurrentBatchKind::SpeculativeDecode`, snapshotting recurrent state so a rejection
  can roll back. Whether anything keyed on `SpeculativeDecode` expects a matching explicit
  accept/reject signal from `speculative/verifier.rs` before treating a window's state as durably
  committed is not something this investigation fully traced, and fast-forward tokens never go
  through that verifier at all (they're grammar-certain, always accepted, by construction). Rather
  than gamble on that being fine, this plan keeps fast-forward windows labeled
  `RecurrentBatchKind::Decode` (relaxed to allow width > 1) and never touches `SpeculativeDecode` or
  its checkpoint-lane code at all. This is a few explicit lines in `gdn/backend.rs` instead of zero,
  but it stays fully orthogonal to the speculative-decoding subsystem rather than borrowing an
  unverified piece of its protocol — a smaller, more legible correctness surface for a reviewer, even
  though it's a larger diff by line count.
- **Checked, not assumed, that relaxing `Decode`'s single-token invariant is actually safe** for the
  other places in the codebase that read `RecurrentBatchKind`:
  - CUDA graph capture (`pipeline/normal.rs`'s `try_cuda_decode_graph_forward`) already
    double-guards on *shape*, not just kind — it calls `cuda_decode_graph_batch_kind_supported`
    (which already allows both `Decode` and `SpeculativeDecode`) and *separately* checks
    `q_len != 1 || ...`, bailing out to eager execution if so. This exists precisely because
    `SpeculativeDecode` windows are already commonly wider than 1 in production. A `Decode`-kind call
    with `seq_len > 1` is already handled correctly here: eager fallback, exactly like a wide
    `SpeculativeDecode` call gets today.
  - `apply_recurrence_from_convolved` (`gdn/backend.rs:226`) doesn't take `batch_kind` at all — it
    dispatches purely on `seq_len` and device (`seq_len == 1 && cpu` → fast decode kernel, everything
    else → the general path, unconditionally on CUDA). It doesn't care what label the caller used.
  - Qwen3.5's own `Decode`-specific fast path (`vision_models/qwen3_5/text.rs:2345`, the
    `deferred_gdn` condition) already requires `query_len == 1` *and* `batch_kind == Decode`
    together, not `Decode` alone — a widened `Decode` call safely fails this check and falls through
    to the general, correct path.
  - Position-id computation (`vision_models/mod.rs`'s `text_decode_position_ids_from_context`)
    already derives positions generically from `seq_len` for `Decode | SpeculativeDecode` uniformly.
  - The one real, honest gap: `apply_recurrence_from_convolved`'s CUDA branch
    (`recurrence_cuda_from_convolved`) is only exercised at width > 1 today via
    `SpeculativeDecode`-labeled calls in the existing test suite. A `Decode`-labeled call with
    `seq_len > 1` runs the identical code (it doesn't branch on the label) but is a combination
    nothing has tested yet. Fine to ship behind an opt-in, CPU-first-validated flag; not something to
    assert as CUDA-safe without someone actually measuring it there. State this plainly in the PR.

## Design decisions and why (state explicitly in the PR, not leave implicit)

1. **Relax `causal_conv1d`'s bail in `gdn/backend.rs` directly** (`Decode && seq_len == 1` for the
   fast path, everything else falls through to `causal_conv1d_full`) rather than reusing
   `RecurrentBatchKind::SpeculativeDecode`. See Context above for the full reasoning: this keeps
   fast-forward fully independent of the speculative-decoding subsystem's accept/reject-shaped state
   machine instead of silently depending on it being compatible.
2. **New, narrowly-scoped `Sequence::pending_ff_tokens: Vec<u32>` field**, not a reuse of
   `staged_speculative_tokens`/`SpeculativeTokens`. That existing type is shaped for verify/accept
   semantics (`SpeculativeProposalDistribution`, a device-tensor variant for GPU draft models) that a
   reviewer would have to reason about for no reason if grammar-certain tokens rode along in it.
   Keeping the two fields separate keeps the two mechanisms legible as distinct concepts even though
   they now share the same relaxed GDN dispatch path.
3. **Opt-in via env var, not a default flip.** Sumac's own measurements found the payoff
   grammar-shape-dependent (near-zero on mostly-freeform completions, several times faster when a
   grammar forces long literal spans), so shipping it as an unconditional default would bet on every
   caller's grammar and hardware. `MISTRALRS_GRAMMAR_FAST_FORWARD=1`, checked once via `OnceLock`,
   matches this crate's existing `perf_flags.rs` convention rather than inventing a new toggle style.
4. **Wire the flag for `NormalPipeline` only** (the only pipeline whose input builder will consume
   `pending_ff_tokens`) — the other 6 `GeneralMetadata` construction sites get a hardcoded `false`,
   with a doc comment on the field itself stating that's a correctness requirement (those pipelines
   build their own inputs and never read the field), not just unmeasured caution.

## File-by-file plan

**`mistralrs-core/src/gdn/backend.rs`**, `causal_conv1d` (currently lines 824-840):
```rust
pub fn causal_conv1d(
    x: &Tensor,
    conv1d_weight: &Tensor,
    dims: &GdnDims,
    cache: &mut GdnLayerCache,
    batch_kind: RecurrentBatchKind,
) -> Result<Tensor> {
    let (_, seq_len, _) = x.dims3()?;
    // A `Decode` step normally queries one new token, but a grammar fast-forward window
    // (Sequence::pending_ff_tokens) can present several already-known tokens at once.
    // `causal_conv1d_full` already handles arbitrary widths correctly (proven by prefill and by
    // `SpeculativeDecode`'s existing multi-token windows) and doesn't care about the kind label.
    if matches!(batch_kind, RecurrentBatchKind::Decode) && seq_len == 1 {
        causal_conv1d_update(x, conv1d_weight, dims, cache)
    } else {
        causal_conv1d_full(x, conv1d_weight, dims, cache)
    }
}
```
No other `gdn/` change needed: `causal_conv1d_update_cpu`'s own `seq_len != 1` bail is only reached
via the branch above once `seq_len == 1` is already guaranteed, and `apply_recurrence_from_convolved`
doesn't gate on `batch_kind` at all. No change needed to `recurrent_batch_kind_for_input` or its call
site either — a fast-forward-widened window stays classified as plain `RecurrentBatchKind::Decode`;
that function's existing `has_staged_speculative_batch` argument is untouched since
`pending_ff_tokens` is a wholly separate field from `staged_speculative_tokens`.

**`mistralrs-core/src/sequence.rs`** *(already implemented on the `grammar-fast-forward` branch,
uncommitted)* — added, right after `staged_speculative_distribution`: a `pending_ff_tokens: Vec<u32>`
field, initialized to `Vec::new()`, plus three methods mirroring the existing staged-speculative
accessors in shape (`active_pending_ff_tokens(&self) -> &[u32]`, `set_pending_ff_tokens`,
`take_pending_ff_tokens`) — no distribution, no device variant, since these tokens need no
verification and no CUDA-tensor path for v1.

**`mistralrs-core/src/perf_flags.rs`** *(already implemented, uncommitted)* — added a third flag
alongside `cuda_graphs_enabled` / `flashinfer_decode_enabled`, following that file's existing
`const ..._ENV` + `static OnceLock<bool>` + `env_flag()` pattern: `grammar_fast_forward_enabled()`,
reading `MISTRALRS_GRAMMAR_FAST_FORWARD`, defaulting `false` (unlike the other two flags, which
default `true` — this one is unmeasured on this repo's own CI/hardware matrix; sumac's journal only
measured one CPU container and one model).

**`mistralrs-core/src/pipeline/mod.rs`** — add `pub supports_grammar_fast_forward: bool` to
`GeneralMetadata` (after `loaded_for_uqff_write`, matching that field's plain `//` comment style, not
a `///` doc comment). Update all 7 `GeneralMetadata { ... }` literals (`normal.rs`, `gguf.rs`,
`embedding.rs`, `ggml.rs`, `diffusion.rs`, `multimodal.rs`, `speech.rs` — each has a
differently-ordered field list, so each needs its own edit, matched by field name not position). Six
sites get `supports_grammar_fast_forward: false,` with a one-line comment pointing at the field's own
doc comment. `normal.rs`'s site gets
`supports_grammar_fast_forward: crate::perf_flags::grammar_fast_forward_enabled(),`.

**`mistralrs-core/src/pipeline/inputs_processor.rs`**, `text_models_inputs_processor::make_completion_chunk`
(currently lines 1525-1740):
- Add a `pending_ff_batch_width`-style homogeneous-width helper mirroring
  `speculative::staging::staged_batch_width` (all-or-none across the batch; a mixed batch falls back
  to no widening for that step, matching the existing staged-speculative fallback behavior).
- In the per-sequence loop: read `seq.active_pending_ff_tokens()`, extend `ctxt` with it (after the
  existing staged-speculative extension). The two are mutually exclusive per sequence in every
  pipeline that will set either field, but the code shouldn't assume that silently — assert or bail
  if both are non-empty for the same sequence rather than silently picking one.
- Track the **full** widened length separately from the **narrowed** logit-selection length: when a
  sequence's width > 1 comes from `pending_ff` alone (no staged speculative on that sequence),
  `context_lens.push((query_len - 1, 1))` instead of `(0, query_len)` — only the position after the
  forced splice needs a sampled logit; the forced tokens are already known and don't need lm_head
  run over them. Keep a separate `full_query_lens: Vec<usize>` (full width, including the forced
  splice) for `DecodePagedRows`'s `query_len` (KV-cache writes need every position), independent of
  the narrowed `context_lens` used for hidden-state/lm_head selection.
- **Verify at implementation time, not assumed now:** confirm how `InputMetadata.context_lens`
  tuples are actually consumed downstream (hidden-state gather before lm_head) in the *current*
  forward-pass code — this file has had large diffs since v0.9.2, so the `(skip, take)` semantics
  need re-confirming against current code, not copied from the old prototype blind. If the narrowing
  turns out more invasive than this shape suggests, ship v1 without it (still gets the forward-pass-
  count reduction and GDN dispatch amortization) and land narrowing as a clearly-labeled follow-up
  once confirmed.

**`mistralrs-core/src/pipeline/sampling.rs`**:
- `sample_sequence` (currently lines 1381-1536): add a trailing `supports_fast_forward: bool` param.
  After the existing `llg.consume_token(...)` block (lines 1511-1522), when `supports_fast_forward &&
  !llg.is_stopped()`, call `let splice = llg.consume_ff_tokens();` (a single call — already
  compute-and-consume in one step per `llguidance::Matcher`'s existing API) and
  `seq.set_pending_ff_tokens(splice)` if non-empty.
- New helper modeled directly on the existing `finalize_block_gen` (lines 686-718, which already does
  "replay a `Vec<u32>` of pre-known tokens through `finish_or_add_toks_to_seq`, stopping early via
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
   `GeneralMetadata` field + all 7 construction sites, `gdn/backend.rs`'s bail relaxation. Compiles,
   changes nothing observable (flag off everywhere, field never populated, so `causal_conv1d` never
   actually sees `seq_len > 1` under `Decode` yet).
2. **Decode-window widening**: `inputs_processor.rs` changes. Still inert — nothing sets
   `pending_ff_tokens` yet, so this is dead code until commit 3, but reviewable on its own as "this is
   how a forced splice would be windowed."
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

1. `git checkout -b grammar-fast-forward` (new local branch off `master`/HEAD — done; nothing pushed).
2. Implement commit 1 (`gdn/backend.rs`, `sequence.rs` and `perf_flags.rs` already done,
   `pipeline/mod.rs` + 7 sites remaining), `cargo check -p mistralrs-core` — do not commit without a
   separate go-ahead (explicitly requested: nothing gets committed yet, independent of this repo's
   own `CLAUDE.md` rule that nothing leaves the machine without approval, which governs pushes/PRs
   rather than local commits).
3. Implement commit 2 (`inputs_processor.rs`), `cargo check -p mistralrs-core`, verify the
   `context_lens` narrowing question above against current forward-pass code before finalizing that
   part.
4. Implement commit 3 (`sampling.rs`), `cargo check -p mistralrs-core` and
   `cargo clippy -p mistralrs-core --tests -- -D warnings`.
5. Build a release wheel (or CLI binary) from this branch, run the demo script twice (env var unset /
   `1`) against a small local GGUF model, confirm byte-identical output and report the timing/call-count
   delta.
6. Report results back before drafting any PR description or touching git remotes.

## Verification

- `cargo check -p mistralrs-core` and `cargo clippy --workspace --tests --examples -- -D warnings`
  after each implementation step.
- `cargo test -p mistralrs-core` — existing GDN/speculative unit tests (e.g. `gdn/backend.rs`'s
  `causal_conv1d_full` vs `causal_conv1d_update_cpu` equivalence tests, `vision_models/qwen3_5/text.rs`'s
  `SpeculativeDecode`-width tests) should be untouched and still pass.
- The demo script's on/off byte-identical-output check is the correctness gate; the timing numbers
  are the "why this is worth merging" evidence for the PR description, kept separate from the diff.

## Current implementation state (uncommitted, on local branch `grammar-fast-forward`)

- Done: `sequence.rs` field/accessors, `perf_flags.rs` toggle.
- Not started: `gdn/backend.rs` bail relaxation, `pipeline/mod.rs` + 7 call sites,
  `inputs_processor.rs`, `sampling.rs`.
- Nothing committed. Nothing pushed. No PR opened.
