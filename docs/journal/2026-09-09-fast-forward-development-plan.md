# 2026-09-09: Grammar fast-forward development plan

A development entry. It converts the findings of
`2026-09-09-fast-forward-comprehensiveness-research.md` into ordered, self-contained tasks, and
prescribes for each one the file, the change, and the check that proves it. It implements nothing.
A separate agent does the editing.

Scope: the `grammar-fast-forward` branch through `3b37de65e`, on top of v0.9.3 `d5ae0f18f`.

Every file, line number and code quotation below was re-read against `3b37de65e` while writing this
entry, not copied from the research entry. Where re-reading changed the prescription the research
entry implies, the task says so.

Three task classes:

- **Implementable** — the change is fully determined here; the container cannot build, so
  verification is by `cargo test -p mistralrs-core` and `cargo clippy` on a machine that can.
- **Blocked** — the fix depends on an answer only a CUDA or Metal build produces. These are named
  and specified, and no code lands for them in this pass.
- **Deferred** — real, sized, and deliberately not in this pass.

---

## Long form

### The one structural decision, made here

Six findings (F1, F2, F3, plus the dormant halves of F6 and F9, plus the CUDA-graph question in
R3) are downstream of a single fact: a fast-forward window arrives at every model labelled
`RecurrentBatchKind::Decode` with `seq_len > 1`, a combination that could not previously occur
(`pipeline/mod.rs:733-743`). Two designs close that:

- **(a) A fourth variant.** Add `RecurrentBatchKind::FastForward` and extend
  `recurrent_batch_kind_for_input` with a third predicate, exactly as `bea02b2c4` added
  `SpeculativeDecode`.
- **(b) Width-derived reclassification.** Leave the enum alone and reclassify at the pipeline
  boundary, as `MultimodalPipeline::recurrent_batch_kind` (`pipeline/multimodal.rs:1834-1857`)
  already does: it takes a `Decode` label, reads `input_ids.dim(1)`, and returns `Prefill` when
  `seq_len != 1` on a paged non-first chunk.

**Take (a).** Reason: (b) is already implemented and already does not apply here.
`MultimodalPipeline` has that reclassifier; `NormalPipeline` has no `fn recurrent_batch_kind` at
all (confirmed by grep over `pipeline/normal.rs`), and `NormalPipeline` is the only pipeline where
`supports_grammar_fast_forward` is ever true. Extending (b) to `NormalPipeline` would mean
relabelling a fast-forward window `Prefill`, which is wrong in the other direction: a prefill label
tells LFM2 to reset its conv state (`models/lfm2.rs:1283-1284`) and tells Granite's
`forward_full` a different cache contract. A named variant states the actual situation — a decode
step whose window is wider than one token because the grammar already knew the tokens — and makes
every existing `== RecurrentBatchKind::Decode` comparison fail loudly-by-default rather than
silently-by-accident.

The cost of (a) is that it is not a one-line change: every existing comparison against `Decode` has
to be classified. Task 1 does exactly that, with the full site list. Nothing else in this plan
depends on which design is chosen except Task 1 and Task 2; if the decision is revisited, only
those two are rewritten.

### Task 1: add `RecurrentBatchKind::FastForward` and classify every existing comparison

**Addresses** F1. **Blocks** Tasks 2, 3 and the R3 investigation. **Implementable.**

Add the variant to `pipeline/mod.rs:727-731` and a third parameter to
`recurrent_batch_kind_for_input` (`pipeline/mod.rs:733-743`):

```rust
pub(crate) fn recurrent_batch_kind_for_input(
    is_prompt: bool,
    has_staged_speculative_batch: bool,
    has_pending_ff_batch: bool,
) -> RecurrentBatchKind
```

Precedence: `Prefill` > `SpeculativeDecode` > `FastForward` > `Decode`. Speculative outranks
fast-forward because `make_completion_chunk` already bails when a sequence carries both
(`pipeline/inputs_processor.rs:1624-1629`), so the combination is an error rather than a case to
order.

Call sites of `recurrent_batch_kind_for_input`, all of which need the new argument:

| Site | Pass |
|---|---|
| `pipeline/inputs_processor.rs:2830` | the real predicate (from Task 2's shared helper) |
| `pipeline/inputs_processor.rs:2734` (X-LoRA arm, hardcoded `RecurrentBatchKind::Decode`) | leave `Decode` — Task 4 makes the flag unreachable under X-LoRA |
| `vision_models/gemma4/inputs_processor.rs:1689` | `false` |
| `vision_models/qwen2_5_vl/inputs_processor.rs:414` | `false` |
| `vision_models/qwen2vl/inputs_processor.rs:979` | `false` |
| `vision_models/qwen3_vl/inputs_processor.rs:1080`, `:1657` | `false` |
| `pipeline/mod.rs:3104`, `:3108`, `:3112` (unit tests) | `false`, plus one new case asserting `FastForward` |

Every existing comparison against `RecurrentBatchKind::Decode`, and what the new variant must do at
each. This list is exhaustive as of `3b37de65e` (`git grep -n "batch_kind" -- mistralrs-core/src`
filtered to `Decode` comparisons):

| Site | Today | Under `FastForward` |
|---|---|---|
| `gdn/backend.rs:836` | `matches!(kind, Decode) && seq_len == 1` → `causal_conv1d_update`, else `causal_conv1d_full` | Falls to `causal_conv1d_full`. **Revert the branch's `&& seq_len == 1` relaxation** and restore the `bail!` under a true `Decode`: with the variant in place, a `Decode` window with `seq_len != 1` is once again impossible, and the guard is worth keeping as an assertion. |
| `gdn/layer.rs:544` | `deferred_decode_batch_supported(..) && kind == Decode && seq_len == 1 && ...` | Falls through to the unaccelerated path. Correct, no edit. |
| `models/granite.rs:942-944` | `matches!(kind, Decode)` → `bail!("Mamba decode expects a single-token query.")` when `seq_len != 1` | Falls to `forward_full`. **This is the F2 fix and it lands for free**, provided Task 3's verification of `forward_full`'s cache contract passes. No edit at this line. |
| `models/lfm2.rs:768` | `matches!(kind, Decode) && seq_len == 1` → `decode_conv`, else `use_existing_state` branch | Falls to `cached_prefill_conv` **only if `use_existing_state` is true** — see the next row. No edit at this line. |
| `models/lfm2.rs:1283-1284` | `use_existing_state = kind == Decode \|\| !ctx.is_first_prompt_chunk()` | **Must become `matches!(kind, Decode \| FastForward) \|\| ...`.** Without this edit the variant silently flips LFM2 from continuing its conv state to resetting it (`models/lfm2.rs:770-778` takes the `prefill_state` arm) — a silent corruption that the current branch does *not* have, because today the label stays `Decode`. This is the single most dangerous line in Task 1. |
| `models/lfm2.rs:956` | `if batch_kind != Prefill` | Unchanged behaviour, no edit. |
| `models/granite.rs:2213`, `:2240`, `:2322`, `models/qwen3_next.rs:640`, `:1006` | all compare against `Prefill` | Unchanged behaviour, no edit. |
| `vision_models/qwen3_5/text.rs:2345` | `deferred_gdn` requires `kind == Decode` (and `query_len == 1`, which already excludes a splice) | Unchanged behaviour: `deferred_gdn` was already false for a splice window via `query_len == 1`. This does **not** close F3 — see Blocked B1. |
| `pipeline/multimodal.rs:1840`, `:2326` | `kind != Decode` / `kind == Decode` | Multimodal never sets the flag. No edit. |
| `pipeline/normal.rs:2079` | `rollback_live_state && kind == Decode && supports_recurrent_speculative_transitions() && ...` | CUDA graph capture path. Leave as `Decode`-only: a fast-forward window is not a speculative rollback. No edit, but note it under R3. |
| `pipeline/cuda_graph.rs:2379` | `cuda_decode_graph_batch_kind_supported` accepts `Decode \| SpeculativeDecode` | **Leave `FastForward` out of the accepted set** for this pass, and record why in R3. |
| `pipeline/cuda_graph.rs:3965-3971`, `:4011-4075`, `:4244`, `:4932` | tests and key construction | Add no `FastForward` case; they exercise the two accepted kinds. |

Verification: `cargo test -p mistralrs-core` plus a new `pipeline/mod.rs` unit test asserting
`recurrent_batch_kind_for_input(false, false, true) == RecurrentBatchKind::FastForward` and that
`(false, true, true)` yields `SpeculativeDecode`.

### Task 2: move splice resolution into the engine, and account for it there

**Addresses** F4 (structure), F10 (accounting), and the prompt-step gap named below. **Depends on**
Task 1. **Implementable.**

Today `resolve_pending_ff_batch` runs inside one of twenty input processors
(`pipeline/inputs_processor.rs:2668-2670`, under `if !is_prompt`), and the engine's token
accounting one layer above it knows nothing about splices (`engine/mod.rs:1825-1842`). Both are the
same omission seen from two sides, and one insertion point fixes both.

`engine/mod.rs:1815-1820` holds `guards_mut: Vec<&mut Sequence>` — a mutable batch — and `is_prompt`
in scope, immediately before `staged_width` is computed. Insert there:

1. On a prompt step (`is_prompt == true`), call `discard_pending_ff_tokens()` on every sequence in
   the batch. A splice cannot survive a step that rebuilds the window from `seq.get_toks()`, and
   the current `if !is_prompt` guard leaves it staged instead of dropping it. Whether a sequence
   carrying a splice can actually be scheduled into a prompt chunk is not settled by reading —
   `output.scheduled_prompt_chunks` (`engine/mod.rs:1820`) shows mixed prefill/decode batches are
   expressible — so the discard is defensive, and the drop counter from Task 7 will say whether it
   ever fires.
2. On a decode step, call the existing all-or-none resolution over `guards_mut`, then compute
   `pending_ff_width` from the surviving state.
3. Add the accounting term the research entry identifies as missing, beside the speculative one:

```rust
let staged = staged_width
    .map(|_| seq.active_staged_speculative_len())
    .unwrap_or_default();
let pending_ff = pending_ff_width
    .map(|_| seq.active_pending_ff_tokens().len())
    .unwrap_or_default();
seq.num_uncomputed_tokens()
    .saturating_add(staged)
    .saturating_add(pending_ff)
```

The `.map(|_| ...)` shape is deliberate and mirrors the speculative term exactly: the width option
is `Some` only when the batch is homogeneous, so the per-sequence length is added only on steps
where the splice actually reaches the window.

Ordering matters and is the reason resolution moves rather than being duplicated: the engine
computes `scheduled_token_counts` *before* the forward pass, and today's resolution happens
*inside* it. Reading the widths before resolution would count splices that are about to be
discarded. Resolving first makes the two agree by construction.

Then:

- Delete the `resolve_pending_ff_batch` call at `pipeline/inputs_processor.rs:2668-2670`. Keep the
  function and `pending_ff_batch_width` (`:1526-1556`); `make_completion_chunk` still reads the
  width at `:1593`.
- Move both helpers, plus the width predicate, into one crate-internal module so the engine and the
  input processor share a single definition. `speculative/staging.rs` already hosts the generic
  `staged_batch_state_from_widths` the branch made `pub(crate)` (`speculative/staging.rs:14-19`);
  putting the fast-forward pair beside it also retires the "pub(crate), not private" comment that
  Task 9 removes.
- Add to `make_completion_chunk`, after the width read at `:1593`, an invariant check: if
  `pending_ff_batch_width(input_seqs)` is `None` while any sequence has a non-empty
  `active_pending_ff_tokens()`, return an error naming unresolved splices. That converts the
  call-site discipline the doc comment at `:1538-1546` describes into an enforced invariant, and it
  is what makes the nineteen `vision_models/` processors safe to reach `make_completion_chunk`
  (F4's live half) without each one having to call the resolver.

Verification: the two existing tests at `pipeline/inputs_processor.rs:3268-3315` move with the
function. Add a test that a batch with one splice and one bare sequence produces
`scheduled_token_counts` equal to the flag-off counts, and one that a homogeneous batch of width
`K` adds `K` per sequence.

### Task 3: verify Granite's `forward_full` accepts a mid-sequence multi-token window

**Addresses** F2. **Depends on** Task 1. **Implementable, but verification-first.**

Task 1 routes Granite's Mamba layer to `forward_full` (`models/granite.rs:955`) instead of
bailing, which is the fix only if `forward_full` continues from the cached recurrent state rather
than starting from zero. `gdn/backend.rs:836`'s equivalent claim is asserted in a comment on the
branch — "`causal_conv1d_full` already handles arbitrary widths correctly (proven by prefill and by
`SpeculativeDecode`'s existing multi-token windows)" — and was exercised by the CPU benchmark for
GDN. Granite's Mamba path has no such evidence.

Read `Mamba::forward_full` and its cache writes before landing Task 1's Granite row, and record the
answer in this entry:

- If `forward_full` seeds from `cache` and writes back the final state, Task 1 closes F2 with no
  edit in `granite.rs`.
- If it assumes a fresh state, `granite.rs:942-955` needs a third arm for `FastForward` that
  continues from the cache, and F2 becomes a real code change rather than a fallthrough.

Until this is answered, no claim that F2 is closed appears in the PR description.

### Task 4: make `no_kv_cache` and X-LoRA unable to stage a splice

**Addresses** F7, F8. **Independent of Tasks 1-3.** **Implementable, two lines.**

Both live correctness holes have the same shape — a code path builds its window from
`seq.get_toks()`, which does not contain the staged splice, while `sample_and_add_toks_inner`
replays the splice unconditionally — and both have the same cheapest correct answer: never stage a
splice on a pipeline where either holds.

`GeneralMetadata` carries `no_kv_cache` and `is_xlora` as `pub` fields (`pipeline/mod.rs:1299`,
`:1307`), and both are in scope at the one site that sets the flag (`pipeline/normal.rs:300`,
inside the same struct literal). Change `pipeline/normal.rs:300` to:

```rust
supports_grammar_fast_forward: crate::perf_flags::grammar_fast_forward_enabled()
    && !no_kv_cache
    && !is_xlora,
```

This is preferred over gating in `sample_and_add_toks_inner` (where `metadata.no_kv_cache` and
`metadata.is_xlora` are equally reachable) because it keeps one place that answers "does this
pipeline do fast-forward", which is the same place Task 6 edits and the same place the field's doc
comment describes.

Consequence to state in the PR description rather than hide: fast-forward and X-LoRA are mutually
exclusive, and fast-forward requires a KV cache.

Verification: no unit test reaches these flags today; assert by reading, and add a
`#[test]` in `pipeline/normal.rs` only if a pipeline-construction test harness already exists there
(it does not as of `3b37de65e`, so the check is a code-reading one).

### Task 5: clear the splice on sequence reallocation and on the block-diffusion window

**Addresses** F6, F9. **Independent.** **Implementable.**

Both are dormant only because multimodal holds the flag off, and both become live the moment Task 6
is taken further than this pass takes it.

- `sequence.rs:1375-1384`: `set_toks_and_reallocate` resets `num_computed_tokens` to zero and calls
  `clear_staged_speculative_tokens()`. Add `self.discard_pending_ff_tokens()` beside it — the
  discarding form, not `take_`, so the llguidance matcher is rolled back rather than left advanced
  past tokens the rebuilt sequence never emits.
- `pipeline/inputs_processor.rs:2455` (`make_completion_prefill_chunk`) reads no
  `pending_ff_tokens`. Task 2's invariant check does not cover it, because it is a different
  function. Add the same check at its head: error if any input sequence carries a non-empty splice.
  Its only caller is `vision_models/gemma4/inputs_processor.rs:1562`, so the check is unreachable
  today and exists to make it a loud failure the day it is not.

### Task 6: extend coverage to the GGUF and GGML legacy pipelines, and correct the field comment

**Addresses** F5, and the half of F4 that is a coverage gap rather than a safety gap.
**Depends on** Task 2 (whose invariant check is what makes widening coverage safe). **Implementable.**

`pipeline/mod.rs:1315-1318` asserts that leaving the flag false outside `NormalPipeline` is "a
correctness requirement, not just an optimization opt-out", listing "Vision/embedding/diffusion/speech".
`GGUFPipeline` (`pipeline/gguf.rs:1415`) and `GGMLPipeline` (`pipeline/ggml.rs:400`) are neither,
and both take their inputs processor from `BasicProcessor` → `TextInputsProcessor`
(`pipeline/processing.rs:204-209`, `pipeline/mod.rs:1342`) — the same processor that builds the
widened window.

Two changes:

1. Set the flag from the env flag at `pipeline/gguf.rs:1415` and `pipeline/ggml.rs:400`, with the
   same `&& !no_kv_cache && !is_xlora` conjunction as Task 4. Both structs have `self.no_kv_cache`
   and `is_xlora` in scope in the same literal (verified at `gguf.rs:1399,1404` and
   `ggml.rs:384,389`).
2. Rewrite the field comment at `pipeline/mod.rs:1315-1318` to state the contract rather than a
   justification: which field the decode path reads, and that a pipeline whose inputs processor is
   not `TextInputsProcessor` leaves it false. Task 9's rules apply to the rewrite.

Multimodal stays false in this pass. Enabling nineteen input processors is Deferred D2, and the
honest line for the PR description is that multimodal tool-calling — one of the most common uses of
constrained decoding — gets nothing from this feature yet.

### Task 7: register the counters that give the drop counter a denominator

**Addresses** N2. **Independent.** **Implementable.**

`mistralrs_grammar_ff_splice_drops_total` (`sequence.rs:1287`) is the only metric the branch
registers, and `pipeline/normal.rs:289-299`'s doc comment directs operators to watch it. A thousand
drops means nothing without knowing whether ten thousand splices were staged.

Register, mirroring the five-counter family speculative decoding uses for exactly this reason
(`docs/src/content/docs/guides/deploy/observability.mdx:111-115`, divided against each other in the
PromQL block at `:124-129`):

| Metric | Incremented where |
|---|---|
| `mistralrs_grammar_ff_splices_staged_total` | `pipeline/sampling.rs:1643` (`set_pending_ff_tokens`), once per splice |
| `mistralrs_grammar_ff_tokens_staged_total` | same site, by `splice.len()` |
| `mistralrs_grammar_ff_splices_fed_total` | Task 2's engine site, once per sequence whose splice reaches the window |
| `mistralrs_grammar_ff_tokens_fed_total` | same site, by the splice length |
| `mistralrs_grammar_ff_splice_drops_total` | unchanged (`sequence.rs:1287`) |

Add a `reason` label to the drop counter, matching how `mistralrs_speculative_staged_drops_total`
is documented ("preemption, batch-shape mismatch, or a step that cannot verify") and how
`mistralrs_prefix_cache_evictions_total` and the encoder-cache counters carry `reason`
(`observability.mdx:73-80`). Values: `batch_shape`, `preemption`, `prompt_step`, `realloc`. Pass it
as an argument to `discard_pending_ff_tokens`, whose four call sites after this plan are Task 2's
two, `paged_attention/scheduler.rs:1385`, and Task 5's reallocation site.

Forward passes skipped is *not* a separate counter: it equals `..._tokens_fed_total`, and the
Task 8 docs say so rather than the code duplicating it.

### Task 8: document the feature

**Addresses** N1. **Depends on** Tasks 4, 6, 7 for accuracy. **Implementable.**

The branch touches no file outside `mistralrs-core/src/` — 17 files, 457 insertions, all in the
crate. Both merged features the research entry compares against (`7ed0e8441` for a perf flag,
`bea02b2c4` for a widened decode window) carried docs in the same commit. Four files:

1. `docs/src/content/docs/reference/environment-variables.md` — a row for
   `MISTRALRS_GRAMMAR_FAST_FORWARD`. **Not** in the "CUDA acceleration" table where the two other
   `perf_flags.rs` flags live (`:69`, `:72`); the feature is hardware-independent. It goes in the
   general table (`:57-64`), stating: off by default, opt in with `1`/`true`/`yes`/`on`, applies to
   grammar-constrained requests only, no effect under `--no-kv-cache` or X-LoRA (Task 4), and
   currently near-baseline for concurrent grammar requests (F12).
2. `docs/src/content/docs/guides/deploy/observability.mdx` — a "Grammar fast-forward" subsection
   after "Speculative decoding (MTP)" (`:106-116`) with the five counters from Task 7, plus a
   PromQL line in "Useful PromQL" (`:118-135`) computing the fed/staged ratio. Also amend the
   `mistralrs_decode_tokens_processed_total` row (`:72`) — it currently reads "With speculative
   decoding only verified tokens count", and after Task 2 fast-forward tokens do count.
3. `docs/src/content/docs/guides/serve/structured-output.mdx` — a subsection under "Grammar
   constraints" (`:137`): what a splice is, which grammar shapes pay off (long forced literal
   spans, tool-call scaffolds) and which do not (mostly-freeform completions), and how to read the
   counters.
4. `docs/src/content/docs/guides/perf/throughput-tuning.mdx` — one cross-reference in the
   "Concurrency: max running sequences" section (`:24`), because F12 is a throughput-under-
   concurrency property and that is where a reader looking for it will be.

No new page and no `docs/astro.config.mjs` sidebar entry. The research entry's comparison to
`7ed0e8441`'s dedicated `guides/perf/use-cuda-graphs.md` no longer applies: that page is now a
redirect (`docs/astro.config.mjs:53`), the perf guides have been consolidated to five files, and a
grammar-scoped flag belongs in the grammar guide.

### Task 9: rewrite the comments to describe the code

**Addresses** C1-C4. **Do last** — every earlier task changes what the comments must say.
**Implementable.**

Line counts on the branch: `pipeline/sampling.rs` 162 added lines of which 32 begin a comment;
`pipeline/inputs_processor.rs` 148/26; `sequence.rs` 43/15; `paged_attention/scheduler.rs` 34/8;
`pipeline/normal.rs` 12/11; `speculative/staging.rs` 6/3; `gdn/backend.rs` 5/4. `sampling.rs` holds
32 comment lines on `master` and 64 on the branch.

The rule to apply, which C4 establishes is the file's own norm rather than an imported standard:
rationale comments are within norms; comments whose subject is the branch's review history are not.
Upstream's comment immediately above the branch's insertion point
(`pipeline/inputs_processor.rs:1582-1586`) is itself rationale-bearing and carries a temporal marker
— "in this first batched implementation" — so the divergence is one of subject, not of presence.

Five specific rewrites:

- `pipeline/normal.rs:289-299` — eleven of the twelve lines this file adds. Delete the sentence
  beginning "The only prior validation (byte-identical output vs. unpatched decode) ran a single
  sequential CPU request...": it describes the state of a review, and stops being true the moment
  someone runs a validation. Delete the F12 paragraph too; after Task 8 it lives in
  `structured-output.mdx`, where it can be updated without a code change. What survives is two
  lines: off by default, opt in via the env var, payoff is grammar-shape-dependent.
- `speculative/staging.rs:14-16` — "pub(crate), not private: also reused by..." describes the
  visibility change the diff makes and reads as a non-sequitur once the diff is history. Task 2
  moves the fast-forward helpers into this module, at which point the sentence is also false.
  Replace with what the function computes.
- `sequence.rs:1272-1283` — "Not reachable in practice: ..." through "Fail the sequence instead of
  continuing on a corrupted matcher" argues for a decision. Keep one sentence of it: a matcher left
  advanced past tokens the sequence never emitted answers every later mask at the wrong grammar
  position. Drop the reachability argument and the deliberation.
- `pipeline/inputs_processor.rs:1644-1650` — "(the bail above already rules out both being active
  at once; this repeats the check on the raw per-sequence state rather than relying on that being
  the only path here)" records a deliberation about whether a redundant check is worth keeping.
  Drop the parenthetical; keep the reason the narrowing is safe.
- `pipeline/sampling.rs:738-740` — explains a field's value by naming the value it replaced
  ("instead of going out as `None`"). State the contract: a replayed token carries the same
  `bytes` and `top_logprobs` shape as a sampled one, because `finish_or_add_toks_to_seq` unwraps
  both unconditionally in its Done-state handling.

Also in scope, and cutting the other way: `pipeline/inputs_processor.rs:1538-1546` states a
counterfactual defect ("without this, a sequence whose splice was staged but not fed would grow by
`splice_len + 1` tokens against a KV cache that only grew by 1 position"). Task 2 moves this
function and converts its "must run before `make_completion_chunk`" clause into an enforced
invariant, so the counterfactual becomes the invariant check's error message — where a
counterfactual is exactly the right thing to say — and the doc comment states the contract.

---

## Blocked

Neither of these produces code in this pass. Both need a CUDA or Metal build; this container
produces no `cargo` invocation at all.

### B1: does Qwen3.5's deferred GDN state flush on every splice step, and what does it cost?

**F3, Open.** `vision_models/qwen3_5/text.rs:2340-2345` computes `deferred_gdn` with `query_len == 1`
as its first conjunct, so a fast-forward window makes it false, and `:2346-2354` then calls
`flush_deferred_recurrent_state` and bails with "Qwen3.5 deferred recurrent state cannot be
materialized" if that fails. Task 1 does not change this: the `query_len == 1` term already
excluded splice windows before the label changed.

What a build settles: whether the flush succeeds, and what it costs per splice.
`crate::cuda::gdn::deferred_decode_batch_supported` is CUDA-only, so a CPU measurement cannot see
either. This is the finding most likely to turn the measured 6.1-6.2x into something else on the
hardware people serve from, because Qwen3.5 is both the model the benchmark used and a GDN hybrid.

Measurement: the `ff_bench.py` configuration on a CUDA build, flag on and flag off, with
`RUST_LOG` at debug to count flushes.

### B2: does the paged allocator under-reserve once the computed-token lag opens?

**F10, the Open half.** Task 2 closes the accounting lag itself, which is the fix. What Task 2 does
not settle is whether the lag was reachable as a panic before it was closed, which determines
whether this is a fix or a regression-fix in the PR description.

The site: `paged_attention/scheduler.rs:1185-1195` picks `seq_guard.len()` when
`num_uncomputed_tokens() > 0` and `seq_guard.len() + 1` otherwise. With the lag open and no splice
staged on a given step, the `len()` branch is taken where `len() + 1` used to be, and
`inputs_processor.rs:1691` panics with "Block table is too small (completion)!" if the slot is
short. Invisible on CPU: `paged_attn_supported()` is a compile-time `const fn` returning `false`
off CUDA and Metal (`utils/mod.rs:297-305`).

Run this before Task 2 lands, on the pre-Task-2 code, or the answer is no longer obtainable.

---

## Deferred

### D1: ragged-width decode windows

**F12.** `resolve_pending_ff_batch` discards every splice in the batch unless every sequence holds
a splice of identical length, and splice length tracks each sequence's own grammar position, so two
concurrent grammar-constrained requests ordinarily agree on nothing. Each one still pays
`Matcher::consume_ff_tokens` to compute the splice and `Matcher::rollback` to undo it. The feature
is a no-op with overhead under concurrent grammar load.

The blocking piece is `make_completion_chunk`'s rejection of rows with differing `query_len`
(`pipeline/inputs_processor.rs:1664-1670`); padded positions would have to be kept out of the KV
cache and out of `slot_mapping`. `LogitsSelection::from_context_lens` already accepts a per-row
`start` at a uniform `len` (`pipeline/mod.rs:1135-1149`), so logit selection needs no change.

Sizing it is a product call, and the honest input to that call is a drop-rate measurement under
real concurrent grammar load — which needs Task 7's staged/fed counters to exist first. That
ordering is the reason D1 is deferred rather than merely unscheduled.

### D2: multimodal input processors

**F4, the coverage half.** Nineteen files under `mistralrs-core/src/vision_models/` implement their
own `process_inputs` and reach `make_completion_chunk` via `get_completion_input`. After Task 2 they
are safe (the invariant check errors rather than silently widening) but still excluded
(`pipeline/multimodal.rs:1531` holds the flag false). Enabling them means answering, per pipeline,
whether its window builder and its recurrent path tolerate `FastForward`, and it is the natural
follow-up PR rather than a task in this one.

`bea02b2c4`'s uniform two-line change to each of the 21 files under `mistralrs-core/src/models/` is
the shape D2 would take.

### D3: the `pub` API break

**N3.** `sample_sequence` is `pub` (`pipeline/sampling.rs:1494`) and gains an eleventh positional
`bool`; `GeneralMetadata::supports_grammar_fast_forward` (`pipeline/mod.rs:1319`) is a new `pub`
field on a `pub` struct with no `Default` impl. Both are source-breaking for any out-of-tree crate
that constructs `GeneralMetadata` or calls `sample_sequence`.

Not a defect and not worth a refactor into a parameter struct in this PR. It goes in the PR
description under a "Breaking changes" heading, which is the entire task.

---

## Investigate before the next pass

### R3: CUDA decode graph capture against variable splice widths

Not in the research entry; found while building Task 1's audit table, and stated here as a
hypothesis to test rather than a finding.

`CudaDecodeGraphKey` (`pipeline/cuda_graph.rs:920-930`) includes `input_shape: Vec<usize>` and
`recurrent_batch_kind`. A fast-forward window therefore produces a distinct key per splice width, so
there is no risk of replaying a width-1 graph against a wider window — that much is safe by
construction. The hypothesis is the other failure mode: the cache is capacity-bounded with an
eviction metric (`pipeline/cuda_graph.rs:61-62`, `:326-344`,
`mistralrs_cuda_graph_evictions_total`), splice widths are data-dependent and unbounded, and capture
is expensive. Variable-width fast-forward windows could evict the width-1 decode graph that every
non-splice step depends on.

Task 1 keeps `FastForward` out of `cuda_decode_graph_batch_kind_supported`
(`pipeline/cuda_graph.rs:2375-2380`) for this reason, which makes the hypothesis moot for the
merge and testable afterwards: measure `mistralrs_cuda_graph_evictions_total` and
`mistralrs_cuda_graph_resident_entries` with `FastForward` added to the accepted set, against the
same run with it excluded.

---

## Validation matrix

The measured configuration (`ff_bench.py`: `Which.GGUF` on `unsloth/Qwen3.5-4B-GGUF`, one request
at a time, `temperature=0.0`, `enable_thinking=False`, a regex forcing one exact passage, 20-core
CPU container, no GPU) exercised 6 of the branch's 17 changed files, and none of
`paged_attention/scheduler.rs`, `speculative/*`, or the six pipelines where the flag is off (M1).
Four configurations close the gap; each names the findings it would have caught.

| Configuration | Reaches | Findings it exercises |
|---|---|---|
| Two concurrent grammar requests, same model, CPU | `StagedBatchState::Mixed`, `discard_pending_ff_tokens`, `Matcher::rollback`, Task 7's drop counter with `reason=batch_shape` | F12, Task 2's accounting, Task 7 |
| One request, `temperature > 0` with a repetition or DRY penalty and `return_logprobs=true` | the sampling reorder at `pipeline/sampling.rs:856-880`, the `bytes`/`TopLogprob` shapes at `:737-762` | the two paths the greedy run could not observe |
| CUDA build, same single-request regex, flag on and off | PagedAttention slot accounting, `full_query_lens`, both CUDA-only GDN paths | B1, B2, R3 |
| Granite with Mamba layers, a grammar, flag on | `models/granite.rs:942` | F2, Task 3 |

Byte-identical output against a flag-off run remains the acceptance criterion in every
configuration, as it was for the original measurement.

---

## Task index

| Task | Findings | Files | Depends on |
|---|---|---|---|
| 1 | F1, F2 (partly) | `pipeline/mod.rs`, `gdn/backend.rs`, `models/lfm2.rs`, `pipeline/cuda_graph.rs`, 5 vision inputs processors | — |
| 2 | F4 (safety), F10 | `engine/mod.rs`, `pipeline/inputs_processor.rs`, `speculative/staging.rs` | 1 |
| 3 | F2 | `models/granite.rs` (possibly no edit) | 1 |
| 4 | F7, F8 | `pipeline/normal.rs` | — |
| 5 | F6, F9 | `sequence.rs`, `pipeline/inputs_processor.rs` | 2 |
| 6 | F5, F4 (coverage) | `pipeline/gguf.rs`, `pipeline/ggml.rs`, `pipeline/mod.rs` | 2 |
| 7 | N2 | `sequence.rs`, `pipeline/sampling.rs`, `engine/mod.rs` | 2 |
| 8 | N1 | 4 files under `docs/src/content/docs/` | 4, 6, 7 |
| 9 | C1-C4 | 5 files under `mistralrs-core/src/` | all |
| B1 | F3 | — | CUDA build |
| B2 | F10 (open) | — | CUDA build, run before Task 2 |
| D1 | F12 | — | Task 7's counters |
| D2 | F4 (coverage) | — | Task 2 |
| D3 | N3 | PR description | — |

---

## Current State

- The plan assigns nine implementable tasks, two blocked investigations, three deferred items and
  one hypothesis to test, covering every finding in
  `2026-09-09-fast-forward-comprehensiveness-research.md` except the two the research entry itself
  records as refuted (M2, M3) and the third refuted finding F11.
- Task 1 is the only task that edits model files, and its audit table lists every comparison against
  `RecurrentBatchKind::Decode` in `mistralrs-core/src` as of `3b37de65e`.
- Task 4 closes both live correctness holes (F7, F8) in a single struct literal at
  `pipeline/normal.rs:300`, which is a smaller change than either finding's description implies.
- Task 2 closes F4's safety half and F10 at one insertion point in `engine/mod.rs:1815-1820`,
  because the engine's accounting and the resolver's placement are the same omission seen from two
  sides.

## Missing

- An answer to whether `Mamba::forward_full` (`models/granite.rs:955`) continues from the cached
  recurrent state — Task 3 blocks Task 1's Granite row on reading it.
- A `cargo` invocation of any kind in this container, so every task's verification is specified and
  none is performed.
- Any measurement on CUDA or Metal hardware, so B1, B2 and R3 stay open.

## Divergence

- `pipeline/mod.rs:1315-1318` states that a false `supports_grammar_fast_forward` outside
  `NormalPipeline` is "a correctness requirement", and `GGUFPipeline` and `GGMLPipeline` use the
  same `TextInputsProcessor` that consumes the field — Task 6 corrects the comment and the
  exclusion together.
- `pipeline/normal.rs:289-299` directs operators to `mistralrs_grammar_ff_splice_drops_total`, which
  has no denominator until Task 7 registers the staged and fed counters.
- `observability.mdx:72` documents `mistralrs_decode_tokens_processed_total` as generated tokens
  processed, and the counter omits every fast-forward token until Task 2 lands.

## Not decided here

- Whether the branch merges as one PR or splits at Task 8. The task graph supports either.
- Whether D1 is worth its size — that stays a product call with Task 7's drop rate as its input.
- The order of Tasks 4, 5, 6 and 7 among themselves; only their dependencies are fixed.
