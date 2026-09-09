# 2026-09-09: Grammar fast-forward — second-round research

Scope: a second, independent read of `grammar-fast-forward` at `8850efa93` (on top of v0.9.3
`d5ae0f18f`), run without contact with the first-pass review or the round-one research and plan
entries on this branch. Two purposes: test whether an independent reader reaches the same
conclusions, and find what a reader who starts from a different question finds. Self-contained.
No build ran; no `cargo` invocation completes in this container.

## Summary

The mechanism works. Nothing below says it is fundamentally broken, and nothing below reopens a
defect the first-pass review closed. What this entry says is that fast-forward does not yet
integrate cleanly with four systems that already existed — recurrent-model dispatch, batch
scheduling, MoE routing and the metrics — and that one of those four can change output rather than
just speed.

In order of how much they matter:

1. **AnyMoE output can differ with the flag on.** This is the only finding that touches correctness
   rather than performance, and it is unverified. An AnyMoE model inherits the feature without
   anyone having decided that, and its router picks one expert per *window* rather than per *token*
   — so a widened window routes differently. Fast-forward is defended as an optimisation that cannot
   change output; on AnyMoE it can. (New D.)
2. **Hybrid recurrent models take the slow path on every fast-forward step.** Nine sites enable
   multi-token recurrent handling only for speculative decoding, by name, and a fast-forward window
   is not labelled that way. Each falls through to the general path, which is conservative and
   probably correct but forfeits exactly the saving the feature exists for. Needs a CUDA build to
   settle. (New A.)
3. **The scheduler composes batches without knowing whether their splices can be used**, so it
   admits combinations that are then discarded wholesale. Not a correctness bug — the discard is
   what makes it safe — but it means the concurrency limitation is not located solely where the
   prior entries place it. (New B.)
4. **The published drop-rate metric cannot balance.** Some splices are neither fed nor dropped, and
   they sit in the denominator of the ratio the observability docs tell operators to watch. (New E.)

One further section (New C) is not a problem found in the code; it is an option for addressing (3)
that the prior entries did not consider — shortening splices to a common length instead of
discarding them.

Everything else in this entry either corroborates a round-one finding independently, or withdraws a
suspicion this pass raised and then disproved.

## Resolving the open questions

The three unresolved items below each name a model class rather than a model. This section names
models, sizes and the check that settles each, so none of them reads as "needs a CUDA build".

Two corrections to note first, because both change the cost by an order of magnitude:

- **AnyMoE is not tied to Mistral-7B.** The repo's own examples use
  `mistralai/Mistral-7B-Instruct-v0.1` as base with `HuggingFaceH4/zephyr-7b-beta` as expert, about
  40 GB for the pair, and the LoRA-expert example points at `typeof/zephyr-7b-beta-lora`, which no
  longer resolves. Nothing requires either. `create_anymoe_layers` is implemented by fourteen
  architectures, `models/qwen3.rs`, `models/qwen2.rs` and `models/smollm3.rs` among them, so base
  and expert can both be Qwen3-0.6B. `AnyMoeConfig::hidden_size` (amoe/mod.rs:144) is read straight
  into `linear(config.hidden_size, n_experts, vb)` (amoe/mod.rs:207), so it is the base model's own
  `hidden_size` from its `config.json`, not the 4096 the shipped TOML hardcodes.
- **`ibm-granite/granite-4.0-micro` exercises no Mamba code.** `models/granite.rs` builds a
  `MambaLayer` only for layers whose `layer_types` entry is `GraniteLayerType::Mamba`
  (granite.rs:55, 94). The dense Granite 4 variants carry no such entry. Any Granite candidate has
  to be checked by reading `layer_types` out of its `config.json` before downloading the weights;
  the hybrid line (`-h-` in the repo name) is where to look, and the 350M member of it is the
  smallest, at well under a gigabyte. The exact repo ids are unverified here.

### Question 1: does AnyMoE routing actually diverge? (New D)

The cheapest of the three, and the only one that needs no GPU at all: `AnyMoeLoader` warns and
disables PagedAttention (pipeline/amoe.rs:66-70), so this path never wanted a paged build.

- Base and expert: Qwen3-0.6B for both, roughly 2.5 GB on disk for the pair.
- Gate training data: `examples/amoe.json` ships in the repo, ten rows, and is enough to produce a
  gate that discriminates. Cut `layers` to `[0, 1, 2]` and `epochs` to 25 to keep the fitting pass
  to minutes.
- `hidden_size`: take it from the base model's `config.json`.
- The check: one prompt, `temperature=0.0`, a *partially* forcing grammar such as a small JSON
  schema — not the fully forcing regex `ff_bench.py` uses, which pins both runs to the same string
  by construction and can therefore never fail. Run with `MISTRALRS_GRAMMAR_FAST_FORWARD` unset and
  set, and assert token-for-token equality.
- What makes it diagnostic rather than just pass/fail: log the `topk(1)` index chosen at
  amoe/mod.rs:266 per forward. Equal outputs with differing expert indices is a different result
  from equal outputs with equal indices, and only the second retires the finding.

### Question 2: what do the nine `SpeculativeDecode` sites cost? (New A)

Split this in two. "Which branch is taken" costs nothing and needs no GPU. "Is the branch it takes
numerically right, and how much does it cost" needs CUDA, because `gdn/layer.rs:532` and the
Qwen3.5 `transition_gdn` and `deferred_gdn` paths are CUDA-gated.

The first half is a `tracing::debug!` at each of the nine sites recording which arm ran, then one
CPU run per architecture family with the flag on and a forcing grammar. That alone confirms or
refutes the claim in New A, which is a claim about control flow.

Model per recurrent family, smallest first:

| Family | Code exercised | Smallest checkpoint | Approx. size | CPU? |
|---|---|---|---|---|
| Short convolution | `models/lfm2.rs` | `LiquidAI/LFM2.5-230M` | 0.46 GB | yes |
| Gated delta net | `gdn/backend.rs`, `gdn/layer.rs`, `vision_models/qwen3_5/text.rs` | `Qwen/Qwen3.5-0.8B` | ~1.5 GB | yes, tight |
| Mamba SSM | `models/granite.rs` | Granite 4 hybrid, 350M member | <1 GB | yes |

Three notes on that table. Qwen3.5-0.8B replaces the 4B the demo used and reaches the same GDN
code, so the GDN question does not need the larger download. `Qwen/Qwen3-Next-80B-A3B-Instruct` is
the only released Qwen3-Next size at roughly 163 GB, and it shares `gdn/` with Qwen3.5, so
Qwen3.5-0.8B covers `models/qwen3_next.rs`'s recurrent behaviour without it. LFM2 is the family
round one's W3 argument turns on (`models/lfm2.rs:1283-1284`), so LFM2.5-230M is also the model
that would catch a regression if anyone revisits the enum-variant decision.

### Question 3: what is the batch_shape drop rate under real load? (New B, and the sizing input for New C and round one's D1)

Model choice is irrelevant here — the question is about scheduling, not about any model's numerics.
Use whichever of the above is already downloaded.

- The scheduler half needs PagedAttention, so it needs a CUDA or Metal build:
  `paged_attn_supported()` is a compile-time `const fn` that returns `false` on a CPU build
  (utils/mod.rs:297-305).
- Drive N concurrent requests carrying *different* JSON schemas, which is the shape that produces
  differing splice widths, and read
  `mistralrs_grammar_ff_splice_drops_total{reason="batch_shape"}` against
  `mistralrs_grammar_ff_splices_staged_total`.
- **This measurement depends on New E being fixed first.** Splices staged on a step whose sampled
  token ends the sequence are counted in the denominator and can never appear in the numerator, so
  the ratio reads low until that is corrected. Fixing the counter is a prerequisite for sizing
  D1 and New C, not an independent piece of tidying.

### Harness

`ff_bench.py` on this branch is the starting point: it builds a `Runner` over `Which.GGUF` and times
repeats around a fixed grammar. Questions 1 and 2 need `Which.Plain` instead, since none of the
models above is being fetched as GGUF here, and need an equality assertion rather than a timing
loop. Question 3 needs concurrent request submission, which `ff_bench.py` has no shape for today.

---

## Long form

### Why an entry that partly repeats an existing one is worth having

`2026-09-09-fast-forward-splice-review.md`, `-remaining-work.md`, `-comprehensiveness-research.md`
and `-development-plan.md` were on this branch before this pass began and were not read until after
its findings were written down. Everything in the "Corroborated" list below was therefore
re-derived from the diff and the surrounding code rather than restated, which is the only reason
it is worth writing at all: a finding two readers reach separately from the same source, without
one seeing the other, is a finding about the code rather than about a reader.

The reverse also holds, and matters more. Round one asked "what does this branch break?" and
audited the sites that could break. This pass asked "what does the widened decode window silently
*change*, and who decides the batch it lands in?", and that question reaches a different set of
files. Four findings below came out of it, and none of them appear in the four prior entries.

### The mechanism, in plain terms

When a grammar leaves exactly one legal continuation for several tokens in a row — the tail of a
forced literal, the closing brace and comma of a tool-call scaffold — llguidance can hand back the
whole run at once. The branch takes that run (a "splice"), parks it on the sequence
(`Sequence::pending_ff_tokens`, sequence.rs:786), and on the next decode step appends it to the
input window so those positions are computed in one forward pass instead of one each. After the
forward pass the splice is replayed into the sequence token by token through the ordinary
completion path, and the token sampled at the position after the splice is applied on top.

The whole thing is off unless `MISTRALRS_GRAMMAR_FAST_FORWARD` is set (perf_flags.rs:33-35), and it
is refused for pipelines that never learned to consume the field — every `GeneralMetadata`
construction site outside normal/GGUF/GGML passes `false`.

The demo it was measured on (`RESULTS.md`) is Qwen3.5-4B GGUF on a 20-core CPU container: one
request at a time, no GPU, no PagedAttention, no flash attention, no CUDA graphs, no speculative
proposer, greedy sampling, and a regex that forces 100% of the completion. That is a narrow slice
of the code the branch changed, and round one already anatomised how narrow. This entry does not
re-do that count; it uses it only to say which of the findings below the 6.1-6.2x measurement
could not have caught. The answer is all of them.

### Corroborated without contact

Re-derived here independently, and already recorded in the round-one entries. No new detail is
added and nothing below turns on them:

- `recurrent_batch_kind_for_input` (pipeline/mod.rs:733-744) carries no fast-forward term, so a
  splice window is labelled `RecurrentBatchKind::Decode` at a width greater than one — a
  combination that could not occur before this branch (round one F1).
- `models/granite.rs:944` and `gdn/backend.rs:834` are the two sites that met that combination with
  a `bail!`, and the branch relaxes both to `&& seq_len == 1`; `models/lfm2.rs:768` already had that
  shape (round one F2, Task 1').
- `vision_models/qwen3_5/text.rs:2340-2346` gates `deferred_gdn` on `query_len == 1`, so every
  splice step on a CUDA Qwen3.5 build falls to `flush_deferred_recurrent_state` (round one F3, open,
  needs a CUDA build).
- The branch touches no file outside `mistralrs-core/src/`, where the closest merged precedent
  (`bea02b2c4`, MTP speculative decoding, 146 files) carried CLI args, pyo3 bindings, the server
  builder, the Rust builder surface, docs, examples and a uniform two-line edit to every file under
  `models/` (round one N1). Independently reached here from the same commit.
- `mistralrs-core` has no `tests/` directory and the norm is inline `#[test]` modules; the branch
  adds three tests, and none to `sampling.rs`, `inputs_processor.rs` or `sequence.rs`, which hold
  11, 17 and 42 existing ones (round one Task 6/Task 3 discussion).

### New A: nobody has audited the sites that test for `SpeculativeDecode`

Round one's W3 audited the twelve `== RecurrentBatchKind::Decode` comparisons and asked whether a
wider window breaks any of them. It concluded — correctly, and the LFM2 argument at
`models/lfm2.rs:1283-1284` is the decisive part of it — that adding a fourth enum variant would
introduce a silent corruption the branch does not have. Nothing here disputes W3.

The complementary set was not audited. Nine sites test for `RecurrentBatchKind::SpeculativeDecode`
specifically, and each of them exists to *enable* a path that only makes sense when the decode
window is wider than one token:

- `gdn/layer.rs:532` — the CUDA speculative-checkpoint path in `forward_speculative_checkpoints`.
- `vision_models/qwen3_5/text.rs:850` — `should_stash_gdn_replay`.
- `vision_models/qwen3_5/text.rs:2321-2324` — `speculative_gdn`, which gates `transition_gdn`.
- `pipeline/normal.rs:2210` — `transitions_supported` in `snapshot_hybrid_recurrent_checkpoints`.
- `pipeline/multimodal.rs:1874`, `:2233`; `vision_models/mod.rs:184`, `:243`;
  `pipeline/cuda_graph.rs:2379`.

A fast-forward window reaches every one of them labelled `Decode`, so every one takes its other
branch. In each case the other branch is the conservative one — flush the deferred state, snapshot
the checkpoints, apply the pending transitions eagerly — so the failure mode is not corruption, it
is that a hybrid-recurrent model on CUDA pays the general path on every splice step while the
feature is supposed to be saving forward passes. Round one's F3 is one instance of this shape; the
other eight have not been looked at, and F3's CUDA-build blocker blocks them too.

This *strengthens* W3 rather than reopening it. A `FastForward` variant would not make any of these
nine fire either, because they test for `SpeculativeDecode` by name. Making them fire means editing
them one at a time with the recurrent contract of each in hand, which is a different and larger
piece of work than an enum change, and it is the piece that decides whether this feature is a win
or a wash on hybrid models.

### New B: the batch is composed before anyone knows what splices it holds

`PagedAttentionScheduler::select_completion_batch` (paged_attention/scheduler.rs:621-632) picks the
decode batch through `completion_batch_indices` (scheduler.rs:547-579). That function reads the
splice-carrying width of the row at the cursor as `active_staged_speculative_len()` and skips every
row whose `active_staged_speculative_len()` differs. It applies no equivalent test on
`active_pending_ff_tokens().len()`.

With fast-forward on and no speculative proposer configured, `active_staged_speculative_len()` is 0
for every row, so the filter is a no-op and the scheduler admits rows on token budget alone —
including rows whose splice widths differ, and rows carrying no splice at all. One step later
`resolve_pending_ff_batch` (speculative/staging.rs:59-68) sees a non-homogeneous batch and discards
every splice in it, rolling each matcher back.

This is not a correctness bug: round one's Defect 1 closed the correctness hole, and the discard is
the mechanism that closes it. It is an architectural gap. Batch composition and splice viability
are decided in two different places, the first with no knowledge of the second, and the second able
only to say no. The round-one entries locate the concurrency limitation entirely in
`make_completion_chunk`'s rejection of rows with differing `query_len`
(inputs_processor.rs:1664-1670), which is where D1 (ragged widths) would fix it. That is the right
place for the *general* fix, but it is not the only lever, and naming the window builder as the sole
blocker understates the problem: even with ragged windows implemented, the scheduler's round-robin
cursor will keep mixing splice-carrying rows with splice-less ones, and a splice-less row pins the
useful width to zero unless the padding scheme handles it.

The splice review's constraint 7 rules out grouping completion batches on splice *width*, and the
reasoning is sound — splice lengths are data-dependent, so grouping on width serializes the batch to
one sequence per step. It does not rule out the weaker predicate: preferring to co-schedule rows
that carry a splice at all, which is a partition into two groups rather than into one group per
observed width. Whether that is worth anything depends on the drop rate under real load, which is
what round one's Task 6 counters were added to measure.

### New C: discarding the whole splice is not the only alternative to ragged windows

The round-one entries present two options for a non-homogeneous batch: drop everything (what the
branch does), or implement ragged-width windows with padding kept out of the KV cache and out of
`slot_mapping` (D1, deferred). There is a third that sits between them and neither entry considers.

`Matcher::rollback` takes a token count. A splice of length K can be shortened to length m by
rolling back `K - m` tokens; the first m tokens stay committed and stay valid, because they were
forced by the grammar independently of what follows them.
`Sequence::discard_pending_ff_tokens` (sequence.rs:1261-1279) already performs the full-length case
of exactly this operation.

So a batch in which every sequence carries a splice — the ordinary shape of a structured-output
endpoint under load, two concurrent JSON-schema requests at different positions in their own
grammars — can be made homogeneous by truncating every splice to the batch minimum instead of
discarding all of them. Today that batch feeds nothing. Under truncation it feeds
`min(K_1..K_n)` forced tokens per sequence, at the cost of one extra partial rollback per sequence
per step and no change to the window builder, the scheduler, or the KV accounting.

The case it does not help is a batch mixing grammar-constrained and unconstrained requests, where
the minimum is zero. That case needs ragged widths, or New B's scheduling preference, or both.

This is stated as an option that was not evaluated, not as a recommendation. Its cost is a partial
rollback per sequence per step on batches that currently pay a full rollback per sequence per step,
so the arithmetic is unlikely to be the deciding factor; the question is whether a shortened splice
is worth the code.

### New D: AnyMoE inherits the flag, and a widened window changes which expert runs

`AnyMoePipeline::get_metadata` forwards the wrapped pipeline's `GeneralMetadata` unchanged
(pipeline/amoe.rs:247-249), so an AnyMoE model built on a normal, GGUF or GGML pipeline inherits
`supports_grammar_fast_forward: true` whenever the flag is set. No entry in the branch or in the
round-one documents records a decision to include AnyMoE.

`MoeMlp::forward` (amoe/mod.rs:258-284) takes hidden states shaped `[b, s, h]`, computes the gate,
then reduces it with `gate.mean(1)` — the mean across the sequence dimension — and selects a single
expert per batch row with `topk(1)` for the whole window. The shapes are correct for any `s`; there
is no bail and no panic.

The behaviour is not the same. At `s == 1`, which is every decode step without this feature, each
generated token selects its own expert. At `s == 1 + K`, one expert serves all `1 + K` positions,
chosen from the mean gate over a window that mixes the previously sampled token with K
grammar-forced ones. Flag on and flag off therefore produce different expert routing, and can
produce different output, on an AnyMoE model — silently, with no error and no counter.

Two qualifications. First, this is not introduced by this branch: a staged speculative proposal
widens the same window the same way, so AnyMoE plus MTP has the same property today. Second,
`gate.mean(1)` is also what prefill does, so the widened decode window is adopting prefill's
granularity rather than inventing a third behaviour. Neither qualification removes the consequence
for this feature specifically: fast-forward is defended as an optimisation that cannot change
output, and on AnyMoE it can, without a speculative proposer being configured and without anything
in the gating chain saying so.

### New E: the staged counter has no matching feed or drop at end of sequence

`mistralrs_grammar_ff_splices_staged_total` is incremented when a splice is staged
(sampling.rs:1630). `mistralrs_grammar_ff_tokens_fed_total` is incremented when one reaches a decode
window (engine/mod.rs:1851). `mistralrs_grammar_ff_splice_drops_total` is incremented on discard
(sequence.rs:1276).

Within one call to `sample_and_add_toks_inner`, a splice is staged inside `sample_sequence`, and the
sampled token is applied *after* that, by `finish_or_add_toks_to_seq`. If that token ends the
sequence — a length cap or a stop string, not EOS, since EOS sets `ends_turn` and suppresses staging
at sampling.rs:1619 — the sequence is finished holding a splice that is never fed and never
discarded.

`observability.mdx` documents the drop rate as
`rate(…splice_drops_total) / rate(…splices_staged_total)`. That ratio has a denominator containing
splices that were never eligible to be dropped, so it reads low by one splice per request that ends
on a length cap or a stop string. On short grammar-constrained completions — the tool-call shape the
feature is aimed at — that is not a rounding error.

### Withdrawn after checking

Two items written during this pass were checked against the code and do not hold:

- A `return_raw_logits` request cannot interact with the `(query_len - 1, 1)` logit narrowing at
  inputs_processor.rs:1625-1632. A raw-logits step returns from `send_raw_responses`
  (pipeline/mod.rs:2683-2697) before reaching the `should_sample_step` gate at
  pipeline/mod.rs:2719, so `sample_sequence` never runs for such a sequence and no splice is ever
  staged on it. The engine's batch-uniform `return_raw_logits` assertion (engine/mod.rs:1928-1935)
  makes the mixed case impossible as well.
- A splice cannot survive into a prompt step. Round one's remaining-work entry already established
  that the only `process_inputs` call passing a subset of the batch is prompt chunking, where
  `is_prompt` holds; independently, a sequence in prompt phase has never sampled, and a preempted
  sequence has its splice discarded at paged_attention/scheduler.rs:1386 before returning to
  `Waiting`. The engine computing `pending_ff_batch_width` unconditionally (engine/mod.rs:1817-1826)
  while resolving only under `!is_prompt` is therefore safe by construction rather than by check.

---

## Current State

- `GeneralMetadata::supports_grammar_fast_forward` (pipeline/mod.rs:1316) is set at all seven
  construction sites: from `perf_flags::grammar_fast_forward_enabled() && !no_kv_cache && !is_xlora`
  at normal.rs:289, gguf.rs:1415 and ggml.rs:400, and `false` at multimodal.rs:1531, speech.rs:330,
  diffusion.rs:252 and embedding.rs:706.
- `perf_flags::grammar_fast_forward_enabled` (perf_flags.rs:33-35) reads
  `MISTRALRS_GRAMMAR_FAST_FORWARD` through a `OnceLock` defaulting to `false`, and each pipeline
  reads it once at load, so the flag is a process-wide load-time constant with no per-request,
  per-model or runtime override.
- `AnyMoePipeline::get_metadata` returns the wrapped pipeline's `GeneralMetadata` unchanged
  (pipeline/amoe.rs:247-249), so an AnyMoE model over a normal, GGUF or GGML pipeline reports
  `supports_grammar_fast_forward: true` when the flag is set.
- `MoeMlp::forward` reduces the expert gate with `gate.mean(1)` across the sequence dimension and
  selects one expert per batch row with `topk(1)` for the whole window (amoe/mod.rs:258-284).
- `sampling.rs:1620` is the only `Matcher::consume_token` call on a sequence recognizer in the
  crate, and `sampling.rs:1625` the only `consume_ff_tokens` call, so staging happens at exactly one
  site, guarded by `!llg.is_stopped() && !ends_turn` (sampling.rs:1619) and then by
  `supports_fast_forward && !llg.is_stopped()` and `!llg.is_error()` (sampling.rs:1624-1631).
- `Sequence::discard_pending_ff_tokens` drops the whole splice and calls
  `Matcher::rollback(splice.len())`, setting `SequenceState::Error` when the rollback fails
  (sequence.rs:1261-1279).
- `PagedAttentionScheduler::select_completion_batch` composes the decode batch through
  `completion_batch_indices`, which filters rows on `active_staged_speculative_len()` and reads no
  splice length (paged_attention/scheduler.rs:547-579, 621-632).
- `DefaultScheduler` admits sequences by count under `DefaultSchedulerMethod::Fixed(n)`
  (scheduler/default_scheduler.rs:301-304), holds no token budget and has no preemption path, so it
  needs no splice accounting — and it is the scheduler the CPU GGUF demo ran under, not the paged
  one whose accounting the branch changed.
- Nine sites test for `RecurrentBatchKind::SpeculativeDecode` by name and take their other branch
  for a fast-forward window: gdn/layer.rs:532, vision_models/qwen3_5/text.rs:850 and :2321-2324,
  pipeline/normal.rs:2210, pipeline/multimodal.rs:1874 and :2233, vision_models/mod.rs:184 and :243,
  pipeline/cuda_graph.rs:2379.
- A raw-logits step returns at `send_raw_responses` (pipeline/mod.rs:2683-2697) before the
  `should_sample_step` gate (pipeline/mod.rs:2719), so `sample_sequence` never runs for a sequence
  with `return_raw_logits` set and no splice is staged on one.
- The branch adds three `#[test]` functions:
  `completion_batches_reserve_slots_for_pending_fast_forward_splices`
  (paged_attention/scheduler.rs:4118), `resolve_pending_ff_batch_discards_mismatched_splice_widths`
  (speculative/staging.rs:150) and `resolve_pending_ff_batch_keeps_equal_width_splices`
  (speculative/staging.rs:163).

## Missing

- No site makes any of the nine `RecurrentBatchKind::SpeculativeDecode` tests fire for a
  fast-forward window, so a hybrid-recurrent model takes the general recurrent path on every splice
  step: `snapshot_hybrid_recurrent_checkpoints` snapshots rather than returning `Ok(None)`
  (pipeline/normal.rs:2210-2216), and `vision_models/qwen3_5/text.rs:2325-2340` applies pending
  recurrent transitions eagerly rather than through `transition_gdn`.
- `completion_batch_indices` applies no splice-width or carries-a-splice predicate
  (paged_attention/scheduler.rs:547-579), so batch composition admits rows that
  `resolve_pending_ff_batch` then forces into a whole-batch discard
  (speculative/staging.rs:59-68).
- No code shortens a splice: `Sequence::discard_pending_ff_tokens` rolls back `splice.len()` and
  takes the whole vector (sequence.rs:1261-1263), and `Matcher::rollback` accepts any count, so a
  partial rollback to the batch-minimum width has no caller.
- No test covers `apply_pending_ff_tokens` (pipeline/sampling.rs:724), the splice branch of
  `make_completion_chunk` (inputs_processor.rs:1597-1632), the `full_query_lens` substitutions
  (inputs_processor.rs:1706-1725), either `anyhow::bail!` guard (inputs_processor.rs:1564-1571,
  2440-2451), or the `Matcher::rollback` half of `discard_pending_ff_tokens` (sequence.rs:1261-1279).
- No test asserts that flag-on and flag-off produce identical tokens for a fixed grammar and seed;
  `ff_bench.py` pins both runs to the same string by construction under a fully forcing regex at
  `temperature=0.0` (branch `ff-demo-artifacts`, `ff_bench.py`, `RESULTS.md`).
- `MISTRALRS_GRAMMAR_FAST_FORWARD` has no counterpart in `mistralrs-cli/src/args/mod.rs`,
  `mistralrs-pyo3/mistralrs.pyi`, `mistralrs-server-core/src/mistralrs_for_server_builder.rs` or the
  `mistralrs/src/` builder surface, all of which `bea02b2c4` extended for its decode-shape change.

## Divergence

- `structured-output.mdx` states fast-forward "is not available under X-LoRA or for multimodal
  pipelines" and names `--no-kv-cache`; an AnyMoE model over a normal, GGUF or GGML pipeline is
  available to it through `AnyMoePipeline::get_metadata` (pipeline/amoe.rs:247-249), and
  `MoeMlp::forward` selects one expert for a whole widened window (amoe/mod.rs:263-267) where a
  width-1 window selects one per token.
- `observability.mdx` documents the drop rate as
  `sum(rate(mistralrs_grammar_ff_splice_drops_total[5m])) / sum(rate(mistralrs_grammar_ff_splices_staged_total[5m]))`;
  a splice staged at sampling.rs:1630 on the step whose sampled token finishes the sequence through
  a length cap or a stop string is neither fed nor discarded, so the denominator counts splices the
  numerator can never count.
- `structured-output.mdx` states "Span length tracks each request's own position in its own grammar,
  so two concurrent grammar-constrained requests ordinarily agree on nothing, and the batch falls
  back to the flag-off baseline for that step", presenting the whole-batch discard as a property of
  splice lengths; the discard is also a property of `completion_batch_indices` composing the batch
  without reading splice lengths (paged_attention/scheduler.rs:547-579) and of
  `discard_pending_ff_tokens` having no partial-rollback caller (sequence.rs:1261-1263).
- `throughput-tuning.mdx:163` describes the `perf_flags.rs` env switches as existing "only for
  debugging and benchmarking comparisons"; `MISTRALRS_GRAMMAR_FAST_FORWARD` is the only entry in
  that file that defaults to off and is the only way to turn a behaviour on.
- `environment-variables.md:64` lists `MISTRALRS_GRAMMAR_FAST_FORWARD` in the "Server and UI" table
  and `structured-output.mdx` links to `/reference/environment-variables/#server-and-ui`, while the
  two existing `perf_flags.rs` entries are listed under "CUDA acceleration"
  (environment-variables.md:70, :73).

## Unverified

- Whether the general recurrent path that the nine `SpeculativeDecode` sites fall through to is
  numerically correct for a fast-forward window as well as slower. Established here is which branch
  is taken, not what it computes. Needs a CUDA build, the same blocker as round one's F3 and R1.
- Whether AnyMoE expert routing actually diverges in output between flag on and flag off, as opposed
  to selecting the same expert in practice because the gate is dominated by the sampled token.
  Established here is that the selection granularity differs. Needs an AnyMoE model, a grammar and
  the flag.
- Whether truncating splices to the batch minimum recovers a useful fraction of the feature under
  concurrent grammar load. Needs `mistralrs_grammar_ff_splice_drops_total{reason="batch_shape"}`
  under real traffic, which is round one's Task 6 dependency for D1 as well.
