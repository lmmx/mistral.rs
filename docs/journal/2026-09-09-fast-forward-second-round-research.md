# 2026-09-09: Grammar Fast-Forward — Second-Round Comprehensiveness Research

Research entry. Records what the `grammar-fast-forward` branch touches, what comparable merged
features touched, and which code paths the branch's decode-window widening reaches that the
Qwen3.5-4B GGUF CPU demo (`RESULTS.md`) does not exercise. No development plan, no edits.

## Method

- This round was run without reading `2026-09-09-fast-forward-comprehensiveness-research.md`,
  `2026-09-09-fast-forward-development-plan.md`, `2026-09-09-fast-forward-remaining-work.md` or
  `2026-09-09-fast-forward-splice-review.md`, so its findings are independent of round one and
  agreement between the two rounds carries evidential weight.
- Branch under review is `grammar-fast-forward` at `8850efa93`, diffed against its merge base
  with `master`, `d5ae0f18f` (v0.9.3): 23 files, +465/-44.
- Demo coverage is `unsloth/Qwen3.5-4B-GGUF` Q4_K_M on a 20-core CPU container with no GPU
  (`RESULTS.md`) — no CUDA, no paged attention, no FlashInfer, no CUDA graphs, no speculative
  proposer, single request at a time.

## Merged-feature norms (prior art)

- `feat(gemma4): support MTP speculative decoding! (#2159)` (`bea02b2c4`) is the closest merged
  analogue — it is the change that introduced the staged-token decode-window widening the
  fast-forward branch reuses (`mistralrs-core/src/speculative/staging.rs` was added by it).
- `bea02b2c4` spans 146 files and reaches every user-facing surface: `mistralrs-cli/src/args/mod.rs`
  plus `commands/{bench,config,run,serve}.rs`, `mistralrs-pyo3/{mistralrs.pyi,src/lib.rs}`,
  `mistralrs-server-core/src/mistralrs_for_server_builder.rs`, the Rust builder surface
  (`mistralrs/src/{builder_macros.rs,model_builder_trait.rs,text_model.rs,multimodal_model.rs,auto_model.rs}`),
  `docs/src/content/docs/guides/perf/` and `docs/src/content/docs/reference/`, and `examples/`.
- `bea02b2c4` added a per-model trait method to all 24 files under `mistralrs-core/src/models/` and
  all 7 under `mistralrs-core/src/xlora_models/` in the same commit — the norm for a decode-shape
  change is to visit every model file rather than the ones a demo model exercises.
- `MISTRALRS_CUDA_GRAPHS` and `MISTRALRS_FLASHINFER_DECODE`, the only other entries in
  `mistralrs-core/src/perf_flags.rs`, both default to `true` and are documented as existing "only
  for debugging and benchmarking comparisons"
  (`docs/src/content/docs/guides/perf/throughput-tuning.mdx:163`) — both name a path that is on by
  default and switchable off, neither is the mechanism by which a user opts a feature in.
- `mistralrs-core` has no `tests/` directory; `mistralrs-flash-attn`, `mistralrs-code-exec`,
  `mistralrs-quant`, `mistralrs-paged-attn`, `mistralrs-vision` and `mistralrs-sandbox` do. Core
  coverage is inline `#[test]` modules: 101 in `paged_attention/scheduler.rs`, 42 in `sequence.rs`,
  17 in `pipeline/inputs_processor.rs`, 11 in `pipeline/sampling.rs`.

## Current State

- `GeneralMetadata` carries `supports_grammar_fast_forward` (`pipeline/mod.rs:1316`) and every one
  of the 7 construction sites sets it, so pipeline enumeration is exhaustive by struct definition:
  `normal.rs:289`, `gguf.rs:1415`, `ggml.rs:400` compute it from
  `perf_flags::grammar_fast_forward_enabled() && !no_kv_cache && !is_xlora`; `multimodal.rs:1531`,
  `speech.rs:330`, `diffusion.rs:252`, `embedding.rs:706` hardcode `false`.
- `perf_flags::grammar_fast_forward_enabled()` (`perf_flags.rs:33-35`) reads
  `MISTRALRS_GRAMMAR_FAST_FORWARD` with default `false`, memoised in a `OnceLock`, and is read once
  per pipeline at load time — the flag is a process-wide, load-time constant with no per-request,
  per-model or runtime override.
- `sample_sequence` stages a splice only inside the `SequenceRecognizer::Llguidance` arm, guarded by
  `!llg.is_stopped() && !ends_turn` and then `supports_fast_forward && !llg.is_stopped()`
  (`pipeline/sampling.rs:1615-1637`); `sampling.rs:1620` is the only `consume_token` call on a
  sequence recognizer in the crate, so there is exactly one staging site.
- `apply_pending_ff_tokens` (`pipeline/sampling.rs:724-761`) replays each staged token through
  `finish_or_add_toks_to_seq`, which recomputes `completion_bytes` from
  `tok_trie().decode_ext(&[token], include_special)` itself (`sampling.rs:205-207`) — the replayed
  token's contribution to `seq.completion_bytes()` goes through the same decoder as a sampled token.
- `resolve_pending_ff_batch` and `pending_ff_batch_width` (`speculative/staging.rs:47-68`) reuse
  `staged_batch_state_from_widths`, whose `Homogeneous` arm requires every sequence in the batch to
  carry a non-empty splice of identical length (`staging.rs:15-37`) — one concurrent request without
  a splice makes the whole step fall back and drops every splice with reason `batch_shape`.
- `make_completion_chunk` appends the splice to the decode window and narrows logit selection to
  `(query_len - 1, 1)` while keeping the full window width in `full_query_lens` for flash-attn
  metadata and paged slot mapping (`pipeline/inputs_processor.rs:1597-1632`, `1706-1725`).
- Two `anyhow::bail!` guards reject a splice that reaches a window builder that does not consume it:
  `inputs_processor.rs:1564-1571` (unresolved splice in `make_completion_chunk`) and
  `inputs_processor.rs:2440-2451` (`make_completion_prefill_chunk`).
- `make_completion_prefill_chunk` is reached only from `get_completion_input_windowed` when
  `paged_attn_metadata.is_some()` (`inputs_processor.rs:2578-2591`), and the only configuration
  setting `decode_window` above 1 is `loaders/multimodal_loaders.rs:9371`
  (`decode_window: Some(cfg.canvas_length)`), on a loader whose pipeline sets
  `supports_grammar_fast_forward: false` — the guard at `inputs_processor.rs:2440` is defensive
  rather than reachable under the current gating.
- `PagedAttentionScheduler::completion_token_cost` (`paged_attention/scheduler.rs:540-545`) and the
  block-reservation loop (`scheduler.rs:1185-1194`) add the splice length; `_preempt`
  (`scheduler.rs:1386`) discards it with reason `preemption`; `Sequence::reset_and_reallocate`
  discards it with reason `realloc` (`sequence.rs:1373`).
- `DefaultScheduler` (`scheduler/default_scheduler.rs`) admits sequences by count
  (`DefaultSchedulerMethod::Fixed(n)`, `default_scheduler.rs:301-304`), holds no token budget and has
  no preemption path, so it needs no splice accounting — the demo's CPU GGUF run used this scheduler,
  not the paged one whose accounting the branch changed.
- Speculative decoding passes `supports_fast_forward: false` at all three `sample_sequence` call
  sites outside `sample_and_add_toks_inner` (`speculative/driver.rs:289`,
  `speculative/verifier.rs:901`, `verifier.rs:966`), and `inputs_processor.rs:1602-1607` bails if a
  sequence carries both — the two mechanisms are mutually exclusive by construction.
- The branch adds 3 `#[test]` functions: `completion_batches_reserve_slots_for_pending_fast_forward_splices`
  (`paged_attention/scheduler.rs:4118`) and two `resolve_pending_ff_batch` cases
  (`speculative/staging.rs:150`, `staging.rs:163`).

## Missing

- `recurrent_batch_kind_for_input(is_prompt, has_staged_speculative_batch)` (`pipeline/mod.rs:733-744`)
  takes no fast-forward argument, and none of its 7 call sites
  (`pipeline/inputs_processor.rs:2813`, `vision_models/gemma4/inputs_processor.rs:1689`,
  `qwen2_5_vl/inputs_processor.rs:414`, `qwen2vl/inputs_processor.rs:979`,
  `qwen3_vl/inputs_processor.rs:1080` and `:1657`) pass one — a fast-forward decode window of width
  greater than 1 is labelled `RecurrentBatchKind::Decode`, the same label a width-1 decode carries.
- Two `RecurrentBatchKind::Decode` sites were changed to tolerate `seq_len > 1` by relaxing a bail
  into a width test (`gdn/backend.rs:834`, `models/granite.rs:944`), while `models/lfm2.rs:768`
  already carried the same `&& seq_len == 1` shape before the branch — the branch fixes two of the
  three recurrent single-token assumptions reachable under the `Decode` label and adds no test for
  either.
- Sites that admit a multi-token recurrent window only under `RecurrentBatchKind::SpeculativeDecode`
  are unreached by a fast-forward window: `gdn/layer.rs:532` (CUDA speculative checkpoint path),
  `vision_models/qwen3_5/text.rs:850` (`should_stash_gdn_replay`), `qwen3_5/text.rs:2321-2324`
  (`speculative_gdn`, gating `transition_gdn`), `pipeline/normal.rs:2210`
  (`snapshot_hybrid_recurrent_checkpoints`, `transitions_supported`), `pipeline/multimodal.rs:1874`
  and `:2233`, `vision_models/mod.rs:184` and `:243`.
- `qwen3_5/text.rs:2340-2346` gates `deferred_gdn` on `query_len == 1` and
  `batch_kind() == RecurrentBatchKind::Decode`; a width-`n` fast-forward window on a hybrid GDN model
  fails that test and falls to `flush_deferred_recurrent_state`, with `candle_core::bail!("Qwen3.5
  deferred recurrent state cannot be materialized")` on failure — the path is unreachable on the
  demo's CPU build.
- `PagedAttentionScheduler::completion_batch_indices` (`paged_attention/scheduler.rs:547-579`) selects
  a decode batch by requiring `active_staged_speculative_len() == staged_width` and applies no
  equivalent filter on splice width, so the scheduler co-schedules sequences whose splice widths
  differ; `resolve_pending_ff_batch` then discards every splice in that step
  (`speculative/staging.rs:59-68`).
- `MISTRALRS_GRAMMAR_FAST_FORWARD` has no counterpart in `mistralrs-cli/src/args/mod.rs`,
  `mistralrs-pyo3/mistralrs.pyi`, `mistralrs-server-core/src/mistralrs_for_server_builder.rs` or the
  `mistralrs/src/` builder surface, all of which `bea02b2c4` extended for its decode-behaviour change.
- No test covers `apply_pending_ff_tokens` (`pipeline/sampling.rs:724`), the splice branch of
  `make_completion_chunk` (`inputs_processor.rs:1597-1632`), the `full_query_lens` substitutions
  (`inputs_processor.rs:1706-1725`), either `anyhow::bail!` guard, or the matcher rollback in
  `discard_pending_ff_tokens` (`sequence.rs:1261-1279`); `sampling.rs`, `inputs_processor.rs` and
  `sequence.rs` gain 0 tests between them while carrying 11, 17 and 42 existing ones.
- No test asserts the property `RESULTS.md` measures by hand — that flag-on and flag-off produce
  byte-identical completions for a fixed grammar and seed.

## Divergence

- `docs/src/content/docs/guides/serve/structured-output.mdx` states fast-forward "is not available
  under X-LoRA or for multimodal pipelines" and names `--no-kv-cache`; it does not state that speech,
  diffusion and embedding pipelines set the flag `false`, nor that speculative decoding disables it.
- `docs/src/content/docs/reference/environment-variables.md:64` lists
  `MISTRALRS_GRAMMAR_FAST_FORWARD` in the "Server and UI" table, and
  `structured-output.mdx` links to `/reference/environment-variables/#server-and-ui`, while the two
  existing `perf_flags.rs` entries are listed under "CUDA acceleration" (`environment-variables.md:70`,
  `:73`).
- `throughput-tuning.mdx:163` describes the `perf_flags.rs` env switches as existing "only for
  debugging and benchmarking comparisons"; `MISTRALRS_GRAMMAR_FAST_FORWARD` is the sole entry in that
  file that defaults to off and is the only way to turn a behaviour on.
- `RESULTS.md` reports 6.1-6.2x on a grammar where 100% of the completion is forced and states the
  tool-call case (~3%) is "not reproduced here"; `structured-output.mdx` and
  `throughput-tuning.mdx:67` carry the qualitative shape of that result but the branch carries no
  measurement of the mixed-grammar case.

## Unverified

- Whether a width-`n` fast-forward window labelled `RecurrentBatchKind::Decode` produces correct
  recurrent state on a hybrid GDN model (Qwen3.5 non-GGUF, Qwen3-Next, LFM2, Granite Mamba), or only
  a slower path — established here is that the label differs from the speculative case, not that the
  numerics differ.
- Whether `CudaDecodeGraphKey` (`pipeline/cuda_graph.rs:1074-1133`) distinguishes a width-`n`
  fast-forward window from a width-`n` speculative one: `input_shape` differs from a width-1 decode,
  and `recurrent_batch_kind` is part of the key, so a shape collision is not established;
  `DecodePagedRows::one_token_continuation` requires `q_len == 1` (`cuda_graph.rs:596-611`) and skips
  fast-forward windows.
- Whether `AnyMoePipeline` inherits `supports_grammar_fast_forward: true` in practice — it forwards
  the wrapped pipeline's `GeneralMetadata` unchanged (`pipeline/amoe.rs:247-249`), so a wrapped
  normal/GGUF/GGML pipeline enables it, and no AnyMoE gating layer was inspected for multi-token
  decode-window handling.
- Whether a `return_raw_logits` request can carry a grammar: `make_completion_chunk` takes no
  `return_raw_logits` parameter and narrows `context_lens` to `(query_len - 1, 1)` whenever a splice
  is present (`inputs_processor.rs:1625-1632`), while the engine asserts batch-uniform
  `return_raw_logits` (`engine/mod.rs:1928-1935`).
- Whether `pending_ff_batch_width` can be non-`None` on a prompt step: the engine computes it
  unconditionally and calls `resolve_pending_ff_batch` only when `!is_prompt`
  (`engine/mod.rs:1817-1826`), so a splice surviving into a prompt step would add to
  `scheduled_token_counts` while `get_prompt_input` ignores it.
- Whether the staged/fed/dropped counters balance: a splice staged at `sampling.rs:1631` on a step
  whose sampled token finishes the sequence is neither fed nor discarded, so
  `mistralrs_grammar_ff_splices_staged_total` can exceed fed-plus-dropped by one per finished
  grammar-constrained request.
- Whether the `any_finished` bifurcation in `sample_and_add_toks_inner`
  (`pipeline/sampling.rs:872-925`) is reachable with the CUDA batched sampler:
  `cuda_token_sampling_plan` returns `None` for any sequence with a non-`None` recognizer
  (`sampling.rs:1025-1032`), so a batch containing a grammar-constrained sequence never takes
  `try_sample_batch_cuda`.
