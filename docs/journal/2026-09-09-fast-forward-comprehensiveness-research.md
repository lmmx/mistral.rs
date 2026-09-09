# 2026-09-09: Is grammar fast-forward a finished feature?

A research entry, not a development entry. It asks one question — **is the
`grammar-fast-forward` branch shaped like a feature that gets merged into mistral.rs, or is it a
working core mechanism that hasn't been wired into the rest of the system yet?** — and answers it
by reading code. It changes nothing and prescribes no fix. A development entry comes later, and a
separate agent does the editing.

Scope: the branch through `3b37de65e`, on top of v0.9.3 `d5ae0f18f`.

Every conclusion here comes from reading. No `cargo` invocation runs in this container, so nothing
below was executed, and findings are labelled accordingly:

- **Confirmed** — established by code quoted with file and line.
- **Refuted** — a claim made in an earlier review session that the code contradicts.
- **Open** — the mechanism is identified, the consequence is not. These need a build to settle.

---

## The short version

The core mechanism works, and the previous two journal entries establish that its five known
defects are closed. The problem is everything around it.

1. **The feature is undocumented.** `MISTRALRS_GRAMMAR_FAST_FORWARD` has no row in the environment
   variable reference, which every other `MISTRALRS_*` flag has. There is no perf guide page, and
   no observability entry for the metric it registers.
2. **The telemetry can't answer "is this working?"** There is one counter, and it counts failures.
   Nothing counts splices staged, splices fed, or forward passes skipped, so the drop count has no
   denominator. Speculative decoding registers five counters precisely so a rate is computable.
3. **The benchmark validated a narrow slice of the branch.** CPU, one request at a time, greedy
   sampling, a regex forcing the whole completion. Large parts of the diff — every PagedAttention
   change, every CUDA path, the mixed-batch discard, the sampling reorder, the logprob work — ran
   zero times.
4. **Granite looks like it hard-errors.** Granite's Mamba path bails with "Mamba decode expects a
   single-token query" on exactly the input this feature creates. It is the same site the branch
   fixed for GDN, and it was not fixed here.
5. **The engine's token accounting doesn't know fast-forward exists.** Feed one sampled token plus
   a four-token splice and the engine computes KV for five positions but advances its counter by
   one. The gap accumulates for the life of the sequence. Speculative decoding has a term at that
   exact line; fast-forward doesn't.
6. **So the token throughput metric under-reports.** The same omission flows into
   `mistralrs_decode_tokens_processed_total`, which the docs describe as generated tokens
   processed.
7. **Concurrency turns the feature off.** Splices only get fed when every sequence in the batch has
   a splice of the *same* length. Splice length tracks each request's own grammar position, so two
   concurrent grammar-constrained requests ordinarily agree on nothing and every splice is thrown
   away.
8. **Several paths were never designed for a splice.** `no_kv_cache`, X-LoRA, all nineteen
   multimodal input processors, sequence reallocation, and block diffusion. Some are dormant
   because the flag is off for those pipelines; two are live today.
9. **Two worries from the earlier review session don't survive contact with the code.** The GGUF
   loader gate and the CUDA resident-decode path are both fine, for reasons recorded below.
10. **The comments narrate the branch's own review history.** Eleven of the twelve lines added to
    `normal.rs` are comment, including a sentence describing what has and hasn't been re-validated.

Points 1, 2, 5, 6 and 7 are what separate this from a merged feature. Point 4 is a probable crash.
Point 8 contains the two live correctness questions.

---

## What the feature does, in one paragraph

When a grammar leaves only one legal continuation for the next several tokens — the closing
scaffold of a tool call, a literal span in a regex — llguidance can hand back all of them at once.
Rather than run a forward pass per token to "discover" tokens the grammar already determined, the
branch stashes them on the sequence (`Sequence::pending_ff_tokens`) and feeds them into the *next*
decode window in one go. That window is therefore wider than one token, which is the source of
nearly everything below: the rest of the engine was built on the assumption that a decode step
processes exactly one new token per sequence, and the only prior exception — speculative decoding —
taught the engine about itself in a dozen places that fast-forward has not.

---

## How I judged "shaped like a merged feature"

Rather than assert a standard, I read two upstream commits that are already merged and that touch
the same files:

- **`7ed0e8441`** — "feat(cuda): implement cuda graphs and various optimizations (#2180)". This is
  the commit that created `mistralrs-core/src/perf_flags.rs` and the `MISTRALRS_CUDA_GRAPHS` flag,
  so it is the closest precedent for adding a perf flag to that file.
- **`bea02b2c4`** — "feat(gemma4): support MTP speculative decoding! (#2159)". This is the commit
  that created `Sequence::staged_speculative_tokens` and `StagedBatchState`, both of which the
  fast-forward branch reuses. It is the closest precedent for widening a decode window.

What they carried, in the same commit as the mechanism:

`7ed0e8441` added `docs/src/content/docs/reference/environment-variables.md`, a dedicated
`docs/src/content/docs/guides/perf/use-cuda-graphs.md`, an entry in
`docs/src/content/docs/guides/perf/index.md`, `docs/src/content/docs/reference/cli.md`, a
troubleshooting entry, `README.md`, both install scripts and `.github/workflows/ci_cuda.yaml`.

`bea02b2c4` added 38 lines to `mistralrs-cli/src/args/mod.rs`, changes to four CLI command files,
two perf guide pages, a supported-models entry, several Python examples, and — the telling part — a
uniform two-line change to **each of the 21 files under `mistralrs-core/src/models/`**.

Two further conventions hold across the repository: every `MISTRALRS_*` runtime flag has a row in
`environment-variables.md` (including both flags already living in `perf_flags.rs`, at lines 69 and
72), and every per-mechanism counter has a row in
`docs/src/content/docs/guides/deploy/observability.mdx`.

**The fast-forward branch touches no file outside `mistralrs-core/src/`.** 17 files, 457
insertions, all in the crate. That is finding **N1 — Confirmed**, and it is the frame for the rest.

---

## A. The benchmark validated a narrow slice

`ff_bench.py` runs `Which.GGUF` on `unsloth/Qwen3.5-4B-GGUF`, one request at a time,
`temperature=0.0`, `enable_thinking=False`, a regex forcing one exact passage, on a 20-core CPU
container with no GPU (`RESULTS.md`). The 6.1–6.2x number is real and it is not in question here.
What is in question is coverage.

Working through what that configuration can and cannot reach:

**PagedAttention never ran.** `paged_attn_supported()` is a compile-time `const fn` that returns
`true` only under `all(feature = "cuda", target_family = "unix")` or `feature = "metal"`, and
`false` otherwise (utils/mod.rs:297-305). The measured wheel is a CPU build. Every change the branch
makes under `paged_attention/scheduler.rs` — the slot-cost term at :543, the allocation branch at
:1188-1192, the preemption discard at :1388 — executed zero times.

**Flash attention never ran either**, so `full_query_lens` — the variable that exists specifically
to keep the true window width separate from the narrowed logit selection — fed neither of its two
consumers. Its consumers are the `paged_single_token_decode` test (inputs_processor.rs:1732) and
the paged block-table `query_len` (inputs_processor.rs:1750-1752), and both are gated behind
PagedAttention or flash metadata (inputs_processor.rs:1734). The subtlest piece of the design is
entirely unexercised.

**Mixed batches never happened.** One request at a time means `staged_batch_state_from_widths` runs
over a one-element iterator and returns `Homogeneous` whenever a splice exists. So
`StagedBatchState::Mixed`, `Sequence::discard_pending_ff_tokens` and `Matcher::rollback` — the
correctness linchpin of the whole design — never fired.

**The sampling reorder was unobservable.** With `temperature=0.0` and no repetition, frequency,
presence or DRY penalty, the penalty context that the reorder at sampling.rs:856-880 exists to
correct is never read.

**The logprob work was unexercised.** The request sets no `return_logprobs`, so the decoded `bytes`
and single-entry `TopLogprob` vector that `apply_pending_ff_tokens` builds (sampling.rs:737-762)
were never populated.

**Preemption never happened**, on two independent grounds: one sequence can't be preempted for
another, and `DefaultScheduler` moves no running sequence to `SequenceState::Waiting` outside its
own tests (scheduler/default_scheduler.rs:527).

**No CUDA anything** — graph replay, batched sampling and resident decode are all
`#[cfg(feature = "cuda")]`.

**What did run:** Qwen3.5 is a GDN hybrid — `vision_models/qwen3_5/text.rs` imports the `gdn` and
`packed_gdn` modules at lines 21 and 27 — so `causal_conv1d` with a `Decode` batch kind and
`seq_len > 1` (gdn/backend.rs:836) is the one model-level change on the branch the measurement
genuinely exercised. That is worth something: it is the change most likely to be silently wrong,
and it isn't.

**Finding M1 — Confirmed.** Of 17 changed files, the measurement exercised `perf_flags.rs`,
`sequence.rs` (staging and taking, never discarding), `sampling.rs` (staging and replay, never the
logprob shapes, never the finished-sequence branch), `inputs_processor.rs` (window widening and the
`context_lens` narrowing, never `full_query_lens`, never the discard arm), `normal.rs` and
`gdn/backend.rs`. It exercised none of `paged_attention/scheduler.rs`, none of `speculative/*`, and
none of the six pipelines where the flag is hardcoded off.

---

## B. Where the feature isn't wired in

### B1. The engine has a designated place to say "this window is wide", and fast-forward doesn't use it

**Finding F1 — Confirmed.** `recurrent_batch_kind_for_input(is_prompt, has_staged_speculative_batch)`
(pipeline/mod.rs:732-743) is the one function that tells a recurrent model *why* a decode window is
wider than one token. `bea02b2c4` added the `SpeculativeDecode` variant to it for exactly that
purpose. The fast-forward branch does not extend it: inputs_processor.rs:2830-2833 still calls it
with only the staged-speculative predicate.

The consequence is that a fast-forward window arrives at every recurrent model labelled
`RecurrentBatchKind::Decode` **with `seq_len > 1`** — a combination that could not previously occur.
Every site that branches on that label now has to tolerate it independently. There are four:

| Site | State |
|---|---|
| `gdn/backend.rs:836` | Relaxed by this branch |
| `models/lfm2.rs:768` | Already carried `&& seq_len == 1` before the branch |
| `models/granite.rs:942-944` | **Bails** |
| `vision_models/qwen3_5/text.rs:2340-2345` | Silently changes behaviour, see B3 |

This is the single structural finding. Everything in this section is downstream of it.

### B2. Granite hard-errors

**Finding F2 — Confirmed.**

```rust
let y = if matches!(batch_kind, RecurrentBatchKind::Decode) {
    if seq_len != 1 {
        candle_core::bail!("Mamba decode expects a single-token query.");
    }
```

That is `models/granite.rs:942-944`. Granite loads through `NormalLoader`, which builds its
pipeline with `build_normal_pipeline` (normal.rs:300) and therefore reads the env flag. A Granite
model with Mamba layers, a grammar, and `MISTRALRS_GRAMMAR_FAST_FORWARD=1` hits this on the first
step that feeds a splice. It is the same shape as the site the branch fixed at gdn/backend.rs:836
and the same shape as the guard lfm2.rs:768 already had. It is a loud failure rather than a silent
one, which makes it cheap to fix and embarrassing to ship.

### B3. Qwen3.5's deferred GDN state gets flushed on every splice step

**Finding F3 — Open.** `vision_models/qwen3_5/text.rs:2340-2345` computes `deferred_gdn` as
`query_len == 1 && ... && metadata.batch_kind() == RecurrentBatchKind::Decode && ...`. A
fast-forward window has `query_len > 1`, so `deferred_gdn` goes false, and the block immediately
below (text.rs:2346-2354) then calls `flush_deferred_recurrent_state` and bails with "Qwen3.5
deferred recurrent state cannot be materialized" if that fails.

So on a CUDA build, every splice step forces a flush. Whether the flush succeeds and what it costs
per splice cannot be settled by reading — `deferred_decode_batch_supported` is CUDA-only and the
measurement is a CPU build. This is the finding most likely to turn "6x faster" into something else
on the hardware people actually serve from.

### B4. The safety check lives inside one of twenty input processors

**Finding F4 — Confirmed.** `resolve_pending_ff_batch` — the function that decides which splices
survive and discards the rest — is called from exactly one place: `TextInputsProcessor::process_inputs`,
under `if !is_prompt` (inputs_processor.rs:2668-2670).

Its own doc comment explains why it can't live closer to the code that needs it:
`make_completion_chunk` holds `&[&mut Sequence]` and can't discard through a shared reference
(inputs_processor.rs:1538-1546). That's a real constraint, honestly recorded. But the result is
that the invariant "splices are resolved before the window is built" is enforced by *call-site
discipline in one file*, not by the type system or by structure.

Nineteen files under `mistralrs-core/src/vision_models/` implement their own `process_inputs` and
call `get_completion_input`, so they reach `make_completion_chunk` and *would* widen the window for
a staged splice. None of them calls `resolve_pending_ff_batch`. The only thing preventing that today
is `supports_grammar_fast_forward: false` at multimodal.rs:1531 — which is also why multimodal
tool-calling, one of the most common uses of constrained decoding, gets nothing from this feature.

### B5. The field comment says the opposite of the truth for two pipelines

**Finding F5 — Confirmed.** pipeline/mod.rs:1315-1318 reads:

> "Vision/embedding/diffusion/speech pipelines build their own inputs and never consume that field,
> so leaving this false there is a correctness requirement, not just an optimization opt-out."

`GGUFPipeline` (gguf.rs:1415) and `GGMLPipeline` (ggml.rs:400) are none of vision, embedding,
diffusion or speech. Both take their inputs processor from `BasicProcessor`, which returns
`TextInputsProcessor` (processing.rs:204-209, pipeline/mod.rs:1342) — the *same* processor that
calls `resolve_pending_ff_batch`. For those two the `false` is a coverage exclusion, and the comment
asserts it is a correctness requirement. A future maintainer reading that comment would not know
they are allowed to turn it on.

### B6. Sequence reallocation forgets the splice

**Finding F6 — Confirmed, dormant.** `Sequence::set_toks_and_reallocate` (sequence.rs:1375-1384)
replaces the token list, resets `num_computed_tokens` to zero, and calls
`clear_staged_speculative_tokens()` — with no counterpart for `pending_ff_tokens`. All 30 call sites
are under `vision_models/`, so it is unreachable while multimodal holds the flag off. It becomes
reachable at the exact moment B4 is addressed, which is the reason to record it now rather than
discover it then.

### B7. `no_kv_cache` samples from the wrong position

**Finding F7 — Confirmed by reading, unverified by execution.** This is a live path today.

`get_completion_input` returns `get_prompt_input(...)` when `no_kv_cache` holds
(inputs_processor.rs:2536-2547). `get_prompt_input` reads `seq.get_toks()`, has no
`pending_ff_tokens` handling, and a staged splice is not yet in `seq.get_toks()`. Meanwhile
`resolve_pending_ff_batch` happily keeps the splice — a single-sequence batch is always
`Homogeneous` — and `sample_and_add_toks_inner` replays it unconditionally before sampling
(sampling.rs:856-868).

So the splice tokens are appended to the sequence, and then a token is appended after them that was
sampled from the logits row for the position *preceding* the splice. The model is conditioned on
the wrong context. `no_kv_cache` is a user-facing setting (`with_no_kv_cache`,
mistralrs_for_server_builder.rs:506) that reaches `build_normal_pipeline` models, so this is not a
synthetic configuration.

### B8. X-LoRA builds two passes that disagree on width

**Finding F8 — Confirmed by reading, unverified by execution.** Also live today. Under
`is_xlora && !is_prompt` (inputs_processor.rs:2671-2698), the "full" pass comes from
`get_prompt_input` over `seq.get_toks()` — no splice — while the scaled pass comes from
`get_completion_input` → `make_completion_chunk` — splice included. The two passes then disagree on
`query_len` and on which positions they cover, for any sequence carrying a splice. X-LoRA normal
models are built by `build_normal_pipeline`, so the flag is live for them.

### B9. Block diffusion's window builder has no splice handling

**Finding F9 — Confirmed, dormant.** `make_completion_prefill_chunk` (inputs_processor.rs:2455)
reads no `pending_ff_tokens`, and `get_completion_input_windowed` (inputs_processor.rs:2567) routes
to it whenever paged metadata is present. Its only caller is
`vision_models/gemma4/inputs_processor.rs:1562`, a multimodal pipeline with the flag off. Dormant
for the same reason as B6, and live under the same conditions.

---

## C. Accounting and telemetry

### C1. The engine advances its token counter by one for a five-token window

**Finding F10 — Confirmed; one consequence Open.** This is the most consequential finding after B2.

`engine/mod.rs:1825-1842` builds `scheduled_token_counts` per sequence as:

```rust
let staged = staged_width
    .map(|_| seq.active_staged_speculative_len())
    .unwrap_or_default();
seq.num_uncomputed_tokens().saturating_add(staged)
```

That `staged` term is precisely what `bea02b2c4` added so the engine would account correctly for
windows widened by speculative decoding. There is no `active_pending_ff_tokens()` term beside it.
`engine/mod.rs:1991` and `:2100` then call `advance_num_computed_tokens(scheduled)`.

So a step that computed KV for `1 + K` positions advances `num_computed_tokens` by 1, and
`Sequence::num_computed_tokens` falls `K` further behind `Sequence::len()` on every splice fed,
cumulatively, for the life of the sequence.

Three things follow. Two are established:

- **The throughput metric under-reports.** `engine/mod.rs:2123` sums `scheduled_token_counts` into
  `total_processed_tokens`, so `mistralrs_decode_tokens_processed_total` omits every fast-forward
  token. `observability.mdx:70-72` documents that counter as generated tokens processed and as one
  of two halves that always sum to `mistralrs_tokens_processed_total`; the identity survives only
  because the tokens are missing from both sides. Ironically this means the branch's own reported
  tok/s *understates* its win.
- **`num_uncomputed_tokens()` becomes non-zero for a pure decode sequence** once the lag opens,
  which `PagedAttentionScheduler::completion_token_cost` reads (paged_attention/scheduler.rs:540-545).

The third is **Open, and it is the one that could bite**: the `allocate_slots` call site
(paged_attention/scheduler.rs:1185-1195) picks `seq_guard.len()` when `num_uncomputed_tokens() > 0`
and `seq_guard.len() + 1` otherwise. With the lag open and no splice staged on a given step, the
`len()` branch is taken where `len() + 1` used to be. Whether that under-reserves the slot for the
token sampled on that step — and therefore whether it can reach
`panic!("Block table is too small (completion)!")` at inputs_processor.rs:1691 — cannot be settled
by reading. It needs a CUDA or Metal build.

Note that this whole finding is invisible on CPU: it only starts to matter where PagedAttention
runs, which is exactly where the benchmark couldn't go.

### C2. One counter, and it counts failures

**Finding N2 — Confirmed.** `mistralrs_grammar_ff_splice_drops_total` (sequence.rs:1287) is the only
metric the branch registers. Nothing counts splices staged, splices fed, or forward passes skipped.

So the flag's own doc comment tells operators to watch the drop counter (normal.rs:295), and the
drop counter has no denominator — a thousand drops is meaningless without knowing whether ten
thousand splices were staged or a thousand were. Compare speculative decoding's five counters at
observability.mdx:111-115, which exist as a family specifically so that an acceptance rate is
computable, and which the docs then show being divided against each other at observability.mdx:124-129.

### C3. The public surface changed

**Finding N3 — Confirmed.** `sample_sequence` is `pub` (sampling.rs:1494) and gains an eleventh
positional `bool`. `GeneralMetadata::supports_grammar_fast_forward` (pipeline/mod.rs:1319) is a
`pub` field on a `pub` struct with no `Default` impl. Both are source-breaking for any out-of-tree
crate that constructs `GeneralMetadata` or calls `sample_sequence`. Not a defect — worth stating in
a PR description rather than having a downstream user discover it.

### C4. Nothing is documented

**Finding N1 — Confirmed**, stated above and restated here for completeness: no
`environment-variables.md` row, no `observability.mdx` row, no `guides/perf/` page, no example, no
CI change. Both comparable merged features carried all of these in the same commit.

---

## D. Concurrency: the feature is a no-op under load

**Finding F12 — Confirmed.** Restated from the remaining-work entry and unchanged at `3b37de65e`,
because it bears directly on what the PR can claim.

`resolve_pending_ff_batch` discards **every** splice in the batch unless every sequence holds a
splice of identical length. Splice length tracks each sequence's own grammar position, so two
requests agree only when their grammars *and* their positions coincide. Two or more concurrent
grammar-constrained requests therefore ordinarily feed no splice at all — and still pay for it:
each one pays `Matcher::consume_ff_tokens` to compute the splice and `Matcher::rollback` to undo it.

The single-request benchmark cannot see this, because a one-element batch is always homogeneous.

Ragged-width support is what would fix it. The blocking piece is `make_completion_chunk`'s rejection
of rows with differing `query_len` (inputs_processor.rs:1664-1670); the padded positions would have
to be kept out of the KV cache and out of `slot_mapping`. `LogitsSelection::from_context_lens`
already accepts a per-row `start` at a uniform `len` (pipeline/mod.rs:1135-1149), so logit selection
needs no change.

---

## E. Two earlier worries that don't hold

Both come from the first review session and both are worth recording as refuted so nobody spends
time on them again.

**Finding M2 — Refuted: the benchmark did not exercise PagedAttention.** The claim was that it did,
because the Python `Runner` left `no_paged_attn` at its default. But `paged_attn_supported()` is a
compile-time `const fn` returning `false` on a non-CUDA, non-Metal build (utils/mod.rs:297-305). No
runtime argument can enable it on the measured wheel. This matters in the opposite direction from
how it was originally raised: it means the paged accounting question in C1 has *not* been shown
benign by the measured run, as was concluded at the time.

**Finding M3 — Refuted: the GGUF loader does reach the feature.** The claim was that
`supports_grammar_fast_forward: false` at gguf.rs:1415 meant `ff_bench.py` never touched the branch's
code. `GGUFLoader::load_native_normal` (gguf.rs:566) delegates to `NormalLoaderBuilder` (gguf.rs:699)
for any architecture with a native adapter, so Qwen3.5 runs under `build_normal_pipeline`
(normal.rs:300) and reads the env flag. The `false` at gguf.rs:1415 governs only the legacy
`GGUFPipeline` fallback. The 6x number stands.

**Finding F11 — Refuted: CUDA resident decode is already guarded.**
`cuda_token_sampling_plan` (sampling.rs:1036-1043) returns `None` when the sequence's recognizer is
anything other than `SequenceRecognizer::None`, and a splice is only ever staged inside the
`SequenceRecognizer::Llguidance` arm of `sample_sequence` (sampling.rs:1626-1653). No splice-carrying
sequence can reach the resident decode or batched sampling path. The function does carry an explicit
`active_staged_speculative_len() != 0` term with no `active_pending_ff_tokens()` counterpart, but
that asymmetry is correct: staged speculative tokens can exist on a sequence with no recognizer, and
a splice cannot.

---

## F. The comments describe the review, not the code

Counting added lines that begin a comment, against all added lines:

| File | added | comment |
|---|---:|---:|
| pipeline/sampling.rs | 162 | 32 |
| pipeline/inputs_processor.rs | 148 | 26 |
| sequence.rs | 43 | 15 |
| paged_attention/scheduler.rs | 34 | 8 |
| pipeline/normal.rs | 12 | 11 |
| speculative/staging.rs | 6 | 3 |
| gdn/backend.rs | 5 | 4 |

`sampling.rs` holds 32 comment lines on `master` and 64 on the branch: the branch doubles the file's
comment count while adding roughly a quarter to its length.

**Finding C1 — Confirmed.** normal.rs:289-299 puts the branch's review history into a shipped doc
comment:

> "The only prior validation (byte-identical output vs. unpatched decode) ran a single sequential
> CPU request under greedy sampling with a fully-forcing regex, a configuration that couldn't
> exercise mixed-width batching, PagedAttention slot accounting, or sampling-penalty ordering --
> those paths have since been reviewed and fixed, but not re-validated end to end."

Eleven of the twelve lines this file adds are comment. The sentence describes the state of a review,
not a property of the code, and it stops being true the moment someone runs a validation.

**Finding C2 — Confirmed.** Four comments describe what the code *is not*, or *was*, rather than
what it does:

- speculative/staging.rs:14-16 opens "pub(crate), not private: also reused by ..." — a statement
  about the visibility change the diff makes, which will read as a non-sequitur once the diff is
  history.
- sequence.rs:1272-1283 opens "Not reachable in practice: ..." and closes "Fail the sequence instead
  of continuing on a corrupted matcher" — arguing for a decision rather than describing the branch.
- inputs_processor.rs:1644-1650 closes "(the bail above already rules out both being active at once;
  this repeats the check on the raw per-sequence state rather than relying on that being the only
  path here)" — recording a deliberation about whether a redundant check is worth keeping.
- sampling.rs:738-740 explains a field's value by naming the wrong value it replaced: "so a replayed
  token's `bytes` matches a sampled token's shape instead of going out as `None`".

**Finding C3 — Confirmed.** inputs_processor.rs:1538-1546 states a counterfactual defect rather than
the function's contract: "without this, a sequence whose splice was staged but not fed would grow by
`splice_len + 1` tokens against a KV cache that only grew by 1 position."

**Finding C4 — Confirmed, and it cuts the other way.** The upstream comment immediately above the
branch's insertion point (inputs_processor.rs:1582-1586) is itself rationale-bearing and carries a
temporal marker — "in this first batched implementation". So rationale comments are within this
file's norms, and the divergence is one of *degree* and of *subject*: the upstream comment narrates
the code's constraints, and the branch's comments narrate the branch's review.

---

## Findings index

| ID | Verdict | One line | Primary site |
|---|---|---|---|
| N1 | Confirmed | No docs of any kind; branch touches nothing outside `mistralrs-core/src/` | — |
| N2 | Confirmed | One counter, counts failures, no denominator | sequence.rs:1287 |
| N3 | Confirmed | `pub` API surface changed source-breakingly | sampling.rs:1494, pipeline/mod.rs:1319 |
| M1 | Confirmed | Benchmark exercised 6 of 17 changed files | `RESULTS.md` |
| M2 | Refuted | Benchmark did *not* run PagedAttention (CPU build) | utils/mod.rs:297-305 |
| M3 | Refuted | GGUF loader *does* reach the feature; 6x stands | gguf.rs:566, 699 |
| F1 | Confirmed | `recurrent_batch_kind_for_input` carries no fast-forward term | pipeline/mod.rs:732-743 |
| F2 | Confirmed | Granite bails on a splice window | models/granite.rs:942-944 |
| F3 | Open | Qwen3.5 deferred GDN flushes every splice step on CUDA | qwen3_5/text.rs:2340-2354 |
| F4 | Confirmed | Splice resolution in 1 of 20 input processors; multimodal excluded | inputs_processor.rs:2668-2670 |
| F5 | Confirmed | Field comment misdescribes the GGUF/GGML exclusions | pipeline/mod.rs:1315-1318 |
| F6 | Confirmed, dormant | `set_toks_and_reallocate` doesn't clear the splice | sequence.rs:1375-1384 |
| F7 | Confirmed by reading | `no_kv_cache` samples from the pre-splice position | inputs_processor.rs:2536-2547 |
| F8 | Confirmed by reading | X-LoRA's two passes disagree on window width | inputs_processor.rs:2671-2698 |
| F9 | Confirmed, dormant | Block diffusion's window builder ignores splices | inputs_processor.rs:2455 |
| F10 | Confirmed / Open | Engine advances computed-tokens by 1 for a `1+K` window | engine/mod.rs:1825-1842 |
| F11 | Refuted | CUDA resident decode already excludes grammar sequences | sampling.rs:1036-1043 |
| F12 | Confirmed | Concurrent grammar requests feed no splice at all | inputs_processor.rs:1538-1546 |
| C1–C4 | Confirmed | Comments narrate the branch's review history | normal.rs:289-299 |

## What this entry does not decide

- Whether F3, and the Open half of F10, are real. Both need a CUDA or Metal build, which this
  container cannot produce.
- Whether ragged-width support (D) is worth its size. That is a product call, and the honest input
  to it is a drop-rate measurement under real concurrent grammar load — which needs the staged/fed
  counters from N2 to exist first.
- Any ordering, sizing or assignment of the work. That belongs in the development entry.
