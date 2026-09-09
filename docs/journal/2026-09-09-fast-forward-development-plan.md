# 2026-09-09: Grammar fast-forward development plan

A development entry. It converts the findings of
`2026-09-09-fast-forward-comprehensiveness-research.md` into ordered, self-contained tasks, and
prescribes for each one the file, the change, and the check that proves it. It implements nothing.
A separate agent does the editing.

Scope: the `grammar-fast-forward` branch through `3b37de65e`, on top of v0.9.3 `d5ae0f18f`.

This entry was written in two passes. The first mapped findings to tasks. The second steelmanned
each finding by re-reading the code and tracing the mechanism through a decode step; four findings
did not survive that pass, and the tasks they justified are withdrawn. The withdrawals are recorded
below with their traces, not deleted, because the research entry still asserts them.

Task classes:

- **Implementable** — the change is fully determined here; the container cannot build, so
  verification is by `cargo test -p mistralrs-core` and `cargo clippy` on a machine that can.
- **Blocked** — the fix depends on an answer only a CUDA or Metal build produces.
- **Deferred** — real, sized, and deliberately not in this pass.
- **Withdrawn** — asserted by the research entry, contradicted by the steelman trace.

---

## Withdrawn by the steelman pass

Read this section before the tasks. It removes roughly half the work the first pass proposed.

### W1: the computed-token lag does not accumulate, and the throughput metric does not under-report

**Research entry F10, the Confirmed half.** The claim: the engine advances `num_computed_tokens` by
1 for a `1+K` window, so the counter "falls `K` further behind `Sequence::len()` on every splice
fed, cumulatively, for the life of the sequence", and
`mistralrs_decode_tokens_processed_total` omits every fast-forward token.

The trace, with `L = seq.len()` and `C = num_computed_tokens` at the start of a step, and `K` the
splice length. `sample_and_add_toks` runs inside `pipeline.step()` (`engine/mod.rs:1436`, `:1506`),
which precedes the advance at `engine/mod.rs:2100`, so the splice is already appended to `len()` by
the time the counter moves.

- **Step n** (splice staged, window is `1+K` wide): `C = L-1`, so `scheduled = 1`. The forward
  computes positions `L-1 ..= L+K-1`. `advance(1)` → `C = L`. Sampling then appends the `K` splice
  tokens and 1 sampled token → `len = L+K+1`. Lag opens: `num_uncomputed_tokens() = K+1`.
- **Step n+1** (no splice): `scheduled = num_uncomputed_tokens() = K+1`. The window is one token
  wide, and `seqlen_offsets` comes from `ctxt.len()`, not from `C`
  (`pipeline/inputs_processor.rs:1640`), so the model reads the correct position regardless.
  `advance(K+1)` → `C = L+K+1`, which is exactly the number of positions now computed. Sampling
  appends 1 → `num_uncomputed_tokens() = 1`. **The lag is closed.**

So the lag is one step deep and self-correcting, not cumulative. Back-to-back splices keep it
bounded at the most recent splice length rather than summing.

The metric follows: step n counts 1 and step n+1 counts `K+1`, totalling `K+2` — identical to what
a correct accounting counts (`1+K` then `1`). `mistralrs_decode_tokens_processed_total` is
phase-shifted by one step, not short by `K`. The `observability.mdx:70-72` identity holds.

`set_num_computed_tokens` clamps to `len()` (`sequence.rs:1453-1455`), which is what makes the
self-correction safe rather than lucky.

**What survives:** `num_uncomputed_tokens()` transiently reads `K+1` for a pure decode sequence.
That feeds `PagedAttentionScheduler::completion_token_cost` (`paged_attention/scheduler.rs:540-545`),
which over-estimates the cost for one step — the safe direction. Task 2 still adds the accounting
term, because a four-line change that makes an invariant true is worth having, but it is tidiness
and must not be described in the PR as fixing a metric bug.

**One real caveat, no task.** The CUDA async decode-completion path advances *before* sampling
appends (`engine/mod.rs:1985-1993`), where the `min(len())` clamp would bind and genuinely lose the
`K`. No splice-carrying sequence reaches it: `cuda_token_sampling_plan` returns `None` for any
sequence whose recognizer is not `SequenceRecognizer::None` (`pipeline/sampling.rs:1036-1043`),
which is research finding F11. Worth one sentence in the PR description.

### W2: the paged allocator does not under-reserve

**Research entry F10, the Open half; first pass B2.** The claim: with the lag open,
`allocate_slots` takes `seq_guard.len()` where `len() + 1` used to be
(`paged_attention/scheduler.rs:1189-1197`), possibly reaching
`panic!("Block table is too small (completion)!")`.

Continuing the W1 trace against the four arms as they stand on the branch:

- **Step n** (`pending_ff = K`): the second arm fires — `len() + pending_ff` = `L + K` — reserving
  positions `0 ..= L+K-1`, exactly the window. Correct. The branch added this arm for this reason.
- **Step n+1** (`pending_ff = 0`, `num_uncomputed_tokens() = K+1 > 0`): the third arm fires —
  `len()` = `L+K+1` — reserving positions `0 ..= L+K`, which is the token sampled at the end of step
  n plus everything before it. Correct.

The `len()` branch is not under-reserving, because `len()` has itself grown by `K+1` in the interim.
The research entry compared the branch taken against the branch that used to be taken without
accounting for `len()` having moved. **No build is needed to settle this, and B2 is withdrawn.**

### W3: the `RecurrentBatchKind::FastForward` variant is not worth adding

**Research entry F1; first pass Task 1.** The first pass proposed a fourth enum variant, justified
as making every existing `== RecurrentBatchKind::Decode` comparison "fail loudly-by-default rather
than silently-by-accident". That justification is wrong on inspection:

- **There is no compiler enforcement to gain.** No site matches exhaustively on
  `RecurrentBatchKind`; every one of the twelve sites is `==`, `!=` or `matches!`
  (`git grep "batch_kind" -- mistralrs-core/src`). A new variant makes `kind == Decode` silently
  false at each, which is the opposite of failing loudly. The only compile errors would come from
  the signature change to `recurrent_batch_kind_for_input`.
- **It introduces a silent corruption the branch does not currently have.**
  `models/lfm2.rs:1283-1284` computes `use_existing_state = kind == Decode || !ctx.is_first_prompt_chunk()`.
  A new variant flips that to false and sends LFM2 down the `prefill_state` arm
  (`models/lfm2.rs:770-778`), resetting conv state mid-sequence. Today, with the label staying
  `Decode`, LFM2 is correct by accident — but it is correct.
- **The one genuinely broken site is fixable in two lines** (W4).

The variant would be the right design for a feature that widens decode windows broadly. This one
is off by default, gated to `NormalPipeline`, and touches one model badly. **Task 1 is withdrawn**;
Task 1' replaces it.

### W4: Granite's fix is the two-line change already applied to GDN, and its verification is answered

**Research entry F2; first pass Task 3.** `models/granite.rs:942-944` bails with "Mamba decode
expects a single-token query." on exactly the input this feature creates. The first pass made this
a fallthrough of the enum refactor and gated it on an unread question: does
`Mamba::forward_full` continue from the cached recurrent state, or assume a fresh one?

Read it. `forward_full` (`models/granite.rs:1170-1340`) takes `prior_state = cache.conv_state.clone()`,
concatenates it ahead of the new tokens, and writes the trailing `conv_kernel_size` window back
(`:1189-1194`); the SSM loop seeds `ssm_state` from `cache.ssm_state` and writes it back
(`:1284-1322`). It is a cache-continuing multi-token path, the same contract as
`causal_conv1d_full`.

So F2 is real, and its fix is the identical relaxation the branch already applied at
`gdn/backend.rs:829-839`, with no enum and no open question. That is Task 1'.

---

## Long form

### Task 1': relax Granite's Mamba decode guard

**Addresses** F1 (the only site where it bites), F2. **Independent.** **Implementable.**

`models/granite.rs:942-944`:

```rust
let y = if matches!(batch_kind, RecurrentBatchKind::Decode) {
    if seq_len != 1 {
        candle_core::bail!("Mamba decode expects a single-token query.");
    }
```

becomes the shape `gdn/backend.rs:836` already uses:

```rust
let y = if matches!(batch_kind, RecurrentBatchKind::Decode) && seq_len == 1 {
```

with the `bail!` deleted, so a wider window falls to `forward_full` at `:1195`. W4 establishes
`forward_full` continues from the cache.

Carry the justification comment across from `gdn/backend.rs:830-833` in the form Task 8 prescribes:
what the code does, not what it used to do.

The other eleven `Decode` comparisons need no edit, and the audit that establishes that is the
table in W3 plus these three: `gdn/layer.rs:544` and `vision_models/qwen3_5/text.rs:2345` each
already carry `seq_len == 1` / `query_len == 1` and fall through safely;
`models/lfm2.rs:768` falls to `cached_prefill_conv` with `use_existing_state` true, which is the
continuing path.

### Task 2: move splice resolution into the engine, and add the accounting term

**Addresses** F4 (the safety half), F10 (as tidiness — see W1). **Independent.** **Implementable.**

Today `resolve_pending_ff_batch` runs inside one of twenty input processors
(`pipeline/inputs_processor.rs:2668-2670`), so the invariant "splices are resolved before the window
is built" is enforced by call-site discipline in one file. The nineteen `vision_models/` processors
reach `make_completion_chunk` without it; only `supports_grammar_fast_forward: false` keeps them
safe.

`engine/mod.rs:1815-1820` holds `guards_mut: Vec<&mut Sequence>` — a mutable batch — immediately
before the speculative width is computed. Insert there:

1. Resolve the batch (the existing all-or-none rule) and hold the resulting `pending_ff_width`.
2. Add the accounting term beside the speculative one at `engine/mod.rs:1832-1842`:

```rust
let pending_ff = pending_ff_width
    .map(|_| seq.active_pending_ff_tokens().len())
    .unwrap_or_default();
seq.num_uncomputed_tokens()
    .saturating_add(staged)
    .saturating_add(pending_ff)
```

The `.map(|_| ...)` shape mirrors the speculative term: the width is `Some` only when the batch is
homogeneous, so the length is added only on steps where the splice reaches the window. Resolving
before reading is what makes the two agree — the reverse order would count splices about to be
discarded.

Then:

- Delete the `resolve_pending_ff_batch` call at `pipeline/inputs_processor.rs:2668-2670`. Keep the
  function and `pending_ff_batch_width` (`:1526-1556`); `make_completion_chunk` still reads the
  width at `:1593`.
- Move both helpers into `speculative/staging.rs`, which already hosts the generic
  `staged_batch_state_from_widths` the branch made `pub(crate)` (`:14-19`), so the engine and the
  input processor share one definition. This also retires the "pub(crate), not private" comment
  Task 8 removes.
- Add to `make_completion_chunk`, after the width read at `:1593`, an invariant check: error if
  `pending_ff_batch_width` is `None` while any sequence carries a non-empty
  `active_pending_ff_tokens()`. That is what makes the nineteen multimodal processors structurally
  safe rather than safe-by-flag.

**Do not add a prompt-step discard.** The first pass proposed one defensively. A splice is staged
only in `sample_sequence` (`pipeline/sampling.rs:1626-1653`), so only a sequence that has already
sampled can carry one; such a sequence is `RunningCompletion`, and the one route back to a prompt
step is preemption, which already discards (`paged_attention/scheduler.rs:1385`). It would be dead
code.

Verification: the two existing tests at `pipeline/inputs_processor.rs:3268-3315` move with the
function. Add one test that a batch mixing a splice-carrying sequence and a bare one produces
`scheduled_token_counts` equal to the flag-off counts.

### Task 3: make `no_kv_cache` and X-LoRA unable to stage a splice

**Addresses** F7, F8 — both live correctness holes. **Independent.** **Implementable, two lines.**

Both have the same shape: a code path builds its window from `seq.get_toks()`, which does not
contain the staged splice, while `sample_and_add_toks_inner` replays the splice unconditionally
(`pipeline/sampling.rs:856-868`).

- `no_kv_cache`: `get_completion_input` returns `get_prompt_input` (`pipeline/inputs_processor.rs:2536-2547`).
  The splice tokens do eventually get context, since a `no_kv_cache` step recomputes everything from
  `get_toks()` — but the token sampled at the end of the splice step was sampled from the logits row
  for the position *preceding* the splice, and is then appended after it. One wrong token per splice.
- X-LoRA: the full pass comes from `get_prompt_input` over `seq.get_toks()` (no splice), the scaled
  pass from `make_completion_chunk` (splice included) (`pipeline/inputs_processor.rs:2671-2698`).
  The two disagree on `query_len`.

`GeneralMetadata` carries both flags (`pipeline/mod.rs:1299`, `:1307`), and both are in scope at the
one site that sets the feature flag (`pipeline/normal.rs:300`, same struct literal, with
`no_kv_cache` at `:273` and `is_xlora` at `:278`):

```rust
supports_grammar_fast_forward: crate::perf_flags::grammar_fast_forward_enabled()
    && !no_kv_cache
    && !is_xlora,
```

Preferred over gating in `sample_and_add_toks_inner` because it keeps one place that answers "does
this pipeline do fast-forward" — the same place Task 5 edits and the field's doc comment describes.

State in the PR description: fast-forward requires a KV cache and is mutually exclusive with X-LoRA.

### Task 4: clear the splice on sequence reallocation and on the block-diffusion window

**Addresses** F6, F9. **Depends on** Task 2. **Implementable.**

Both are dormant only because multimodal holds the flag off.

- `sequence.rs:1375-1384`: `set_toks_and_reallocate` resets `num_computed_tokens` and calls
  `clear_staged_speculative_tokens()` with no counterpart for `pending_ff_tokens`. Add
  `self.discard_pending_ff_tokens()` — the discarding form, so the matcher is rolled back rather
  than left advanced past tokens the rebuilt sequence never emits.
- `pipeline/inputs_processor.rs:2455` (`make_completion_prefill_chunk`) reads no
  `pending_ff_tokens`, and Task 2's invariant check does not cover it. Add the same check at its
  head. Its only caller is `vision_models/gemma4/inputs_processor.rs:1562`, so it is unreachable
  today and exists to be loud the day it is not.

### Task 5: extend coverage to GGUF and GGML, and correct the field comment

**Addresses** F5, and the coverage half of F4. **Depends on** Task 2. **Implementable.**

`pipeline/mod.rs:1315-1318` asserts that a false flag outside `NormalPipeline` is "a correctness
requirement, not just an optimization opt-out", listing "Vision/embedding/diffusion/speech".
`GGUFPipeline` (`pipeline/gguf.rs:1415`) and `GGMLPipeline` (`pipeline/ggml.rs:400`) are neither,
and both take their inputs processor from `BasicProcessor` → `TextInputsProcessor`
(`pipeline/processing.rs:204-209`, `pipeline/mod.rs:1342`) — the processor that builds the widened
window.

1. Set the flag from the env flag at `pipeline/gguf.rs:1415` and `pipeline/ggml.rs:400`, with the
   same conjunction as Task 3 (`no_kv_cache` and `is_xlora` are in scope in both literals:
   `gguf.rs:1399,1404`, `ggml.rs:384,389`).
2. Rewrite the comment at `pipeline/mod.rs:1315-1318` to state the contract — which field the decode
   path reads, and that a pipeline whose inputs processor is not `TextInputsProcessor` leaves it
   false. Task 8's rules apply to the rewrite.

Multimodal stays false; that is D2.

### Task 6: give the drop counter a denominator

**Addresses** N2. **Independent.** **Implementable.**

`mistralrs_grammar_ff_splice_drops_total` (`sequence.rs:1287`) is the only metric the branch
registers, and `pipeline/normal.rs:289-299` directs operators to watch it. A drop count without a
denominator answers nothing.

Three counters, not five — the first pass proposed staged/fed pairs in both splices and tokens, and
two of those four are derivable:

| Metric | Incremented where |
|---|---|
| `mistralrs_grammar_ff_splices_staged_total` | `pipeline/sampling.rs:1643`, once per splice staged |
| `mistralrs_grammar_ff_tokens_fed_total` | Task 2's engine site, by splice length per sequence fed |
| `mistralrs_grammar_ff_splice_drops_total` | unchanged site, plus a `reason` label |

Drops over staged gives the drop rate; tokens fed is the payoff, and equals forward passes skipped
(the docs say so rather than a fourth counter existing). `reason` values: `batch_shape`,
`preemption`, `realloc` — matching how `mistralrs_speculative_staged_drops_total` and
`mistralrs_prefix_cache_evictions_total` are labelled (`observability.mdx:73-80`, `:115`). Pass it
as an argument to `discard_pending_ff_tokens`, whose three call sites after this plan are Task 2's,
`paged_attention/scheduler.rs:1385`, and Task 4's.

### Task 7: document the feature

**Addresses** N1. **Depends on** Tasks 3, 5, 6 for accuracy. **Implementable.**

The branch touches no file outside `mistralrs-core/src/` — 17 files, 457 insertions. Both merged
features the research entry compares against (`7ed0e8441`, `bea02b2c4`) carried docs in the same
commit. Four files:

1. `docs/src/content/docs/reference/environment-variables.md` — a row for
   `MISTRALRS_GRAMMAR_FAST_FORWARD` in the general table (`:57-64`), **not** the "CUDA acceleration"
   table where the other two `perf_flags.rs` flags live (`:69`, `:72`): the feature is
   hardware-independent. State: off by default; grammar-constrained requests only; no effect under
   `--no-kv-cache` or X-LoRA (Task 3); near-baseline under concurrent grammar load (F12).
2. `docs/src/content/docs/guides/deploy/observability.mdx` — a "Grammar fast-forward" subsection
   after "Speculative decoding (MTP)" (`:106-116`) with Task 6's three counters, and one PromQL line
   in `:118-135` for the drop rate. Leave the
   `mistralrs_decode_tokens_processed_total` row (`:72`) alone — W1 establishes it is accurate.
3. `docs/src/content/docs/guides/serve/structured-output.mdx` — a subsection under "Grammar
   constraints" (`:137`): what a splice is, which grammar shapes pay off (long forced literal spans,
   tool-call scaffolds) and which do not, how to read the counters.
4. `docs/src/content/docs/guides/perf/throughput-tuning.mdx` — one cross-reference in "Concurrency:
   max running sequences" (`:24`), because F12 is a throughput-under-concurrency property and that
   is where a reader looks for it.

No new page and no `docs/astro.config.mjs` sidebar entry. The research entry's comparison to
`7ed0e8441`'s dedicated `guides/perf/use-cuda-graphs.md` no longer applies: that page is a redirect
(`docs/astro.config.mjs:53`), the perf guides are consolidated to five files, and a grammar-scoped
flag belongs in the grammar guide.

### Task 8: rewrite the comments to describe the code

**Addresses** C1-C4. **Do last** — every earlier task changes what the comments must say.
**Implementable.**

Added lines, of which lines beginning a comment: `pipeline/sampling.rs` 162/32,
`pipeline/inputs_processor.rs` 148/26, `sequence.rs` 43/15, `paged_attention/scheduler.rs` 34/8,
`pipeline/normal.rs` 12/11, `speculative/staging.rs` 6/3, `gdn/backend.rs` 5/4. `sampling.rs` holds
32 comment lines on `master` and 64 on the branch.

The rule, which C4 establishes is the file's own norm rather than an imported standard: rationale
comments are within norms; comments whose subject is the branch's review history are not. Upstream's
comment immediately above the branch's insertion point (`pipeline/inputs_processor.rs:1582-1586`) is
itself rationale-bearing and carries a temporal marker — "in this first batched implementation".

- `pipeline/normal.rs:289-299`, eleven of the twelve lines this file adds. Delete the sentence
  beginning "The only prior validation (byte-identical output vs. unpatched decode) ran a single
  sequential CPU request...": it describes the state of a review and stops being true the moment
  someone runs a validation. Delete the F12 paragraph too — after Task 7 it lives in
  `structured-output.mdx`, where updating it is not a code change. Two lines survive: off by
  default, opt in via the env var, payoff is grammar-shape-dependent.
- `speculative/staging.rs:14-16` — "pub(crate), not private: also reused by..." describes the
  visibility change the diff makes, and Task 2 makes it false as well as stale. Replace with what
  the function computes.
- `sequence.rs:1272-1283` — "Not reachable in practice: ..." through "Fail the sequence instead of
  continuing on a corrupted matcher" argues for a decision. Keep one sentence: a matcher left
  advanced past tokens the sequence never emitted answers every later mask at the wrong grammar
  position. Drop the reachability argument.
- `pipeline/inputs_processor.rs:1644-1650` — drop the parenthetical recording whether a redundant
  check was worth keeping; keep why the narrowing is safe.
- `pipeline/sampling.rs:738-740` — explains a field by naming the value it replaced ("instead of
  going out as `None`"). State the contract: a replayed token carries the same `bytes` and
  `top_logprobs` shape as a sampled one, because `finish_or_add_toks_to_seq` unwraps both
  unconditionally.
- `pipeline/inputs_processor.rs:1538-1546` states a counterfactual defect. Task 2 converts its
  "must run before `make_completion_chunk`" clause into an enforced invariant, so the counterfactual
  becomes that check's error message — where a counterfactual is the right thing to say — and the
  doc comment states the contract.

---

## Blocked

### B1: does Qwen3.5's deferred GDN state flush on every splice step, and what does it cost?

**F3, Open.** `vision_models/qwen3_5/text.rs:2340-2345` computes `deferred_gdn` with `query_len == 1`
as its first conjunct, so a fast-forward window makes it false, and `:2346-2354` calls
`flush_deferred_recurrent_state`, bailing with "Qwen3.5 deferred recurrent state cannot be
materialized" if it fails. Task 1' does not change this.

`crate::cuda::gdn::deferred_decode_batch_supported` is CUDA-only, so a CPU measurement sees neither
whether the flush succeeds nor what it costs. This is the finding most likely to turn the measured
6.1-6.2x into something else on serving hardware, because Qwen3.5 is both the benchmarked model and
a GDN hybrid.

Measurement: the `ff_bench.py` configuration on a CUDA build, flag on and off, `RUST_LOG=debug` to
count flushes.

This is now the only blocked item; W2 retired the other.

---

## Deferred

### D1: ragged-width decode windows

**F12.** `resolve_pending_ff_batch` discards every splice unless every sequence in the batch holds
one of identical length, and splice length tracks each sequence's own grammar position, so two
concurrent grammar-constrained requests ordinarily agree on nothing. Each still pays
`Matcher::consume_ff_tokens` and `Matcher::rollback` — CPU-side matcher work, cheap next to a
forward pass, so the honest claim is "no-op with small overhead", not "pays for it".

The blocking piece is `make_completion_chunk`'s rejection of rows with differing `query_len`
(`pipeline/inputs_processor.rs:1664-1670`); padded positions would have to be kept out of the KV
cache and out of `slot_mapping`. `LogitsSelection::from_context_lens` already accepts a per-row
`start` at a uniform `len` (`pipeline/mod.rs:1135-1149`), so logit selection needs no change.

Sizing it is a product call whose honest input is a drop rate under real concurrent grammar load,
which needs Task 6's counters first. That ordering is why D1 is deferred rather than unscheduled.

### D2: multimodal input processors

**F4, coverage half.** Nineteen files under `mistralrs-core/src/vision_models/` implement their own
`process_inputs` and reach `make_completion_chunk`. After Task 2 they are safe (the invariant check
errors rather than silently widening) but still excluded (`pipeline/multimodal.rs:1531`). Enabling
them means answering, per pipeline, whether its window builder and recurrent path tolerate a wide
decode window — a follow-up PR, shaped like `bea02b2c4`'s uniform two-line change to each of the 21
files under `mistralrs-core/src/models/`.

The line for the PR description: multimodal tool-calling, one of the most common uses of constrained
decoding, gets nothing from this feature yet.

### D3: the `pub` API break

**N3.** `sample_sequence` is `pub` (`pipeline/sampling.rs:1494`) and gains an eleventh positional
`bool`; `GeneralMetadata::supports_grammar_fast_forward` (`pipeline/mod.rs:1319`) is a new `pub`
field on a `pub` struct with no `Default` impl. Both are source-breaking for out-of-tree crates. Not
a defect, not worth a parameter-struct refactor here — it goes under a "Breaking changes" heading in
the PR description, which is the whole task.

---

## Investigate later

### R1: CUDA decode graph capture against variable splice widths

Not in the research entry; found while auditing `Decode` comparisons, and stated as a hypothesis.

`CudaDecodeGraphKey` (`pipeline/cuda_graph.rs:920-930`) includes `input_shape`, so a fast-forward
window produces a distinct key per splice width and cannot replay a width-1 graph against a wider
window — safe by construction. The hypothesis is the other failure mode: the cache is
capacity-bounded with an eviction metric (`pipeline/cuda_graph.rs:61-62`, `:326-344`), splice widths
are data-dependent and unbounded, and capture is expensive, so variable-width windows could evict
the width-1 decode graph every non-splice step depends on.

`cuda_decode_graph_batch_kind_supported` accepts `Decode | SpeculativeDecode`
(`pipeline/cuda_graph.rs:2375-2380`), and with W3 withdrawing the enum variant a fast-forward window
is labelled `Decode` and is therefore eligible. Measure `mistralrs_cuda_graph_evictions_total` and
`mistralrs_cuda_graph_resident_entries` on a CUDA build with the flag on and off, alongside B1's run.
No code change in this pass.

---

## Validation matrix

The measured configuration (`ff_bench.py`: `Which.GGUF` on `unsloth/Qwen3.5-4B-GGUF`, one request at
a time, `temperature=0.0`, `enable_thinking=False`, a regex forcing one exact passage, 20-core CPU
container, no GPU) exercised 6 of 17 changed files, and none of `paged_attention/scheduler.rs`,
`speculative/*`, or the six pipelines where the flag is off (M1).

| Configuration | Reaches | Findings |
|---|---|---|
| Two concurrent grammar requests, CPU | `StagedBatchState::Mixed`, `discard_pending_ff_tokens`, `Matcher::rollback`, the drop counter with `reason=batch_shape` | F12, Task 2, Task 6 |
| One request, `temperature > 0` with a repetition or DRY penalty, `return_logprobs=true` | the sampling reorder (`pipeline/sampling.rs:856-880`), the `bytes`/`TopLogprob` shapes (`:737-762`) | the two paths the greedy run could not observe |
| CUDA build, same single-request regex, flag on and off | PagedAttention slot accounting, `full_query_lens`, both CUDA-only GDN paths | B1, R1 |
| Granite with Mamba layers, a grammar, flag on | `models/granite.rs:942` | F2, Task 1' |

Byte-identical output against a flag-off run is the acceptance criterion in every configuration, as
it was for the original measurement.

---

## Task index

| Task | Findings | Files | Depends on |
|---|---|---|---|
| 1' | F1, F2 | `models/granite.rs` | — |
| 2 | F4 (safety), F10 (tidiness) | `engine/mod.rs`, `pipeline/inputs_processor.rs`, `speculative/staging.rs` | — |
| 3 | F7, F8 | `pipeline/normal.rs` | — |
| 4 | F6, F9 | `sequence.rs`, `pipeline/inputs_processor.rs` | 2 |
| 5 | F5, F4 (coverage) | `pipeline/gguf.rs`, `pipeline/ggml.rs`, `pipeline/mod.rs` | 2 |
| 6 | N2 | `sequence.rs`, `pipeline/sampling.rs`, `engine/mod.rs` | 2 |
| 7 | N1 | 4 files under `docs/src/content/docs/` | 3, 5, 6 |
| 8 | C1-C4 | 5 files under `mistralrs-core/src/` | all |
| B1 | F3 | — | CUDA build |
| D1 | F12 | — | Task 6's counters |
| D2 | F4 (coverage) | — | Task 2 |
| D3 | N3 | PR description | — |

Tasks 1', 2 and 3 are independent of each other and of everything else; three agents could take them
in parallel. Everything else funnels through Task 2.

---

## Current State

- Eight implementable tasks, one blocked investigation, three deferred items, one hypothesis, and
  four withdrawn findings cover every finding in
  `2026-09-09-fast-forward-comprehensiveness-research.md` except the three it records as refuted
  (M2, M3, F11).
- Task 3 closes both live correctness holes (F7, F8) in a single struct literal at
  `pipeline/normal.rs:300`.
- Task 1' closes F2 in two lines, and W4 answers the cache-contract question the first pass left
  open.
- No task edits `RecurrentBatchKind` or any file under `mistralrs-core/src/models/` other than
  `granite.rs`.

## Missing

- Any `cargo` invocation in this container, so every task's verification is specified and none is
  performed.
- Any measurement on CUDA or Metal hardware, so B1 and R1 stay open.

## Divergence

- `pipeline/mod.rs:1315-1318` calls a false `supports_grammar_fast_forward` outside `NormalPipeline`
  "a correctness requirement", and `GGUFPipeline` and `GGMLPipeline` use the same
  `TextInputsProcessor` that consumes the field — Task 5 corrects the comment and the exclusion
  together.
- `pipeline/normal.rs:289-299` directs operators to `mistralrs_grammar_ff_splice_drops_total`, which
  has no denominator until Task 6 registers the staged counter.
- The research entry states that `mistralrs_decode_tokens_processed_total` omits every fast-forward
  token and that the computed-token lag accumulates for the life of the sequence — W1 traces both to
  ground and withdraws them.

## Not decided here

- Whether the branch merges as one PR or splits at Task 7. The task graph supports either.
- Whether D1 is worth its size — a product call with Task 6's drop rate as its input.
- The order of Tasks 4, 5 and 6 among themselves; only their dependencies are fixed.

---

## Resolved

All eight implementable tasks landed on `grammar-fast-forward` in ten commits, `3f651cc86` through
`8e4e21606`, one commit per task plus one out-of-plan comment trim. The branch now touches four
files outside `mistralrs-core/src/`, which was finding N1's substance. Net across the ten commits:
16 files, 241 insertions, 166 deletions.

**No commit was compiled.** The container that made them carries no Rust toolchain — no `cargo`, no
`rustc` — so every change was verified by reading, and no claim below rests on a build.

### Landed

| Task | Commit | Files | Effect |
|---|---|---|---|
| 1' | `3f651cc86` | `models/granite.rs` | `matches!(batch_kind, Decode)` gains `&& seq_len == 1` and the `bail!("Mamba decode expects a single-token query.")` is deleted, so a splice window falls to `forward_full` |
| 2 | `a3851faba` | `engine/mod.rs`, `pipeline/inputs_processor.rs`, `speculative/staging.rs` | `pending_ff_batch_width` and `resolve_pending_ff_batch` move to `speculative/staging.rs`; the engine resolves the batch at `engine/mod.rs:1817` before any width is read and adds the `pending_ff` term to `scheduled_token_counts`; `make_completion_chunk` errors on an unresolved splice |
| 3 | `e6e9f8f40` | `pipeline/normal.rs` | `supports_grammar_fast_forward` gains `&& !no_kv_cache && !is_xlora`; the field's comment drops from 11 lines to 3 |
| 4 | `2c061c838` | `sequence.rs`, `pipeline/inputs_processor.rs` | `set_toks_and_reallocate` calls `discard_pending_ff_tokens`; `make_completion_prefill_chunk` errors on any splice, with no homogeneous-width escape |
| 5 | `2dd39c871` | `pipeline/gguf.rs`, `pipeline/ggml.rs`, `pipeline/mod.rs` | both legacy text pipelines read the env flag; the `GeneralMetadata` field comment states the `TextInputsProcessor` contract instead of a correctness claim |
| 6 | `83deb86ad` | `sequence.rs`, `pipeline/sampling.rs`, `engine/mod.rs`, `speculative/staging.rs`, `paged_attention/scheduler.rs` | `mistralrs_grammar_ff_splices_staged_total` and `mistralrs_grammar_ff_tokens_fed_total` registered; `discard_pending_ff_tokens` takes `reason: &'static str`, passed as `batch_shape`, `preemption`, `realloc` |
| 7 | `b3a8bb732` | 4 files under `docs/src/content/docs/` | env-var row, three-counter table plus drop-rate PromQL, a "Grammar fast-forward" subsection under "Grammar constraints", one cross-reference in throughput tuning |
| 8 | `174cd4dde` | `sequence.rs`, `pipeline/inputs_processor.rs`, `pipeline/sampling.rs`, `speculative/staging.rs` | four comment rewrites; the `splice_len + 1` counterfactual moves from a doc comment to the invariant check it describes |

`mistralrs_grammar_ff_tokens_fed_total` increments only where `pending_ff_width` is `Some`, and
`resolve_pending_ff_batch` has already counted every splice in a batch where it is `None` as a drop
— the two are mutually exclusive per step, so `drops / staged` is a rate rather than two unrelated
series divided.

### Deviations from the plan as written

- Task 8's `speculative/staging.rs` rewrite landed early, in `a3851faba` — moving the helpers into
  that module made the "pub(crate), not private: also reused by inputs_processor.rs" sentence false
  as well as stale.
- Task 8's `pipeline/normal.rs` trim landed in `e6e9f8f40` and the `GeneralMetadata` field comment
  in `2dd39c871`, each beside the code change that made the old text wrong. `174cd4dde` covers the
  four remaining sites.
- Task 2's suggested engine-level test, asserting `scheduled_token_counts` for a batch mixing a
  splice-carrying and a bare sequence equals the flag-off counts, is not written. `engine/mod.rs`
  carries no harness that constructs a scheduled batch. The width logic is covered by
  `staged_batch_state_from_widths([15, 0]) == Mixed` and by the two `resolve_pending_ff_batch` tests
  that moved to `speculative/staging.rs`.
- Task 5 writes `!self.no_kv_cache` at `pipeline/gguf.rs:1415` and `pipeline/ggml.rs:400`, not the
  `!no_kv_cache` the plan quoted from Task 3: `no_kv_cache` is a struct field at those two sites and
  a local binding only in `build_normal_pipeline`. `is_xlora` is a local at all three
  (`gguf.rs:1319`, `ggml.rs:310`, `normal.rs:170`).
- `8e4e21606` is not a plan task. The comment above `grammar_fast_forward_enabled`
  (`perf_flags.rs:33-34`) read "near-zero on mostly-freeform completions, several times faster when
  a grammar forces long literal spans"; the magnitude generalises the single CPU configuration in
  `RESULTS.md`, and the comment now states the dependency shape without it.
- `docs/src/content/docs/guides/serve/structured-output.mdx` says "Span length tracks each request's
  own position" — "splice" is the internal term and reaches user-facing prose only inside the metric
  names.
- `pipeline/sampling.rs:912-914` ("Rare enough ... isn't worth the extra row-selection bookkeeping")
  stays. It states why the CUDA batched-sampling path is skipped on a step where a splice finished a
  sequence, which is rationale for a design choice rather than review narration.

### Divergences closed

- `pipeline/mod.rs:1315-1318` no longer calls a false `supports_grammar_fast_forward` outside
  `NormalPipeline` a correctness requirement, and `GGUFPipeline` and `GGMLPipeline` no longer hold
  it false (`2dd39c871`).
- `pipeline/normal.rs` no longer directs operators to a drop counter with no denominator: the
  sentence is deleted (`e6e9f8f40`) and the denominator exists (`83deb86ad`).
- The research entry's claims that `mistralrs_decode_tokens_processed_total` omits fast-forward
  tokens and that the computed-token lag accumulates for the life of the sequence stand withdrawn by
  W1. `observability.mdx:72` is unedited, and `b3a8bb732` adds no fast-forward clause to it.

### Not landed

- **B1** (F3, Qwen3.5 deferred GDN flush per splice step) and **R1** (CUDA decode-graph eviction
  under variable splice widths) need a CUDA or Metal build. Neither has one.
- **D1** (ragged-width windows), **D2** (multimodal input processors) and **D3** (the `pub` API
  break, which belongs in a PR description) are unchanged and unstarted.
- **No `cargo build`, `cargo test -p mistralrs-core` or `cargo clippy` has run against any of the ten
  commits.** The two changes most likely to break mechanically are the `ff_test_sequence` helper,
  which moved into `speculative/staging.rs`'s test module with its imports re-derived rather than
  compile-checked, and the `Sequence::discard_pending_ff_tokens` signature change across its three
  call sites.
