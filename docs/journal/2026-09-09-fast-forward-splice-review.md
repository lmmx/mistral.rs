# 2026-09-09: Grammar fast-forward splice review

Review of branch `grammar-fast-forward` (commits `e7bba90be`, `e9b0eac07`, `d217eb32b` on top of
v0.9.3 `d5ae0f18f`) against the layering that SGLang, llguidance and vLLM V1 converged on for
grammar jump-forward decoding. Reviewed by Claude Opus 5; branch authored by Louis Maddox with
Claude Sonnet. Build status on `d217eb32b` is unverified: build scripts cannot execute in
the review container, so no `cargo` invocation in the review session completed.

Three defects block correctness outside the single-sequence CPU configuration that `ff_bench.py`
measures. Sections "Current State" through "Divergence" carry the machine-checkable statements.
The long-form part below carries the reasoning a fix needs, plus an explicit list of code that a
fix must leave alone.

---

## Long form

### What the branch implements

`Sequence::pending_ff_tokens` (sequence.rs:788) holds a grammar-forced token splice between two
decode steps. `sample_sequence` stages a splice by calling `Matcher::consume_ff_tokens`
(sampling.rs:1574) after committing the sampled token to the matcher, and writes the splice with
`Sequence::set_pending_ff_tokens` (sampling.rs:1577). On the following step,
`make_completion_chunk` appends the splice to the decode window (inputs_processor.rs:1597-1601),
widening `query_len` from 1 to `1 + splice_len`. After the forward pass,
`sample_and_add_toks_inner` removes the splice with `Sequence::take_pending_ff_tokens`
(sampling.rs:835) and replays each splice token through `finish_or_add_toks_to_seq` via
`apply_pending_ff_tokens` (sampling.rs:727-747), then commits the token sampled at the position
after the splice.

The mechanism is gated by `GeneralMetadata::supports_grammar_fast_forward` (pipeline/mod.rs:1319),
set from `perf_flags::grammar_fast_forward_enabled` (perf_flags.rs:35-37) for the normal pipeline
only (normal.rs:292) and hardcoded `false` at the six other `GeneralMetadata` construction sites.
`MISTRALRS_GRAMMAR_FAST_FORWARD` defaults to off.

### Architectural fit

The branch keeps fast-forward outside the speculative decoding accept/reject path, which matches
llguidance's own framing in `docs/fast_forward.md` ("similar to speculative decoding, except that
the speculation is 100% correct") and SGLang's compressed-FSM implementation. No acceptance
sampler runs over forced tokens, no `SpeculativeProposalDistribution` is constructed for a splice,
and `RecurrentBatchKind::SpeculativeDecode` is not reused for the widened window. The two plans on
this branch (`plans/declarative-forging-flask.md`, `plans/declarative-forging-flask-v1-specdec-reuse.md`)
differ only on which `RecurrentBatchKind` reaches `causal_conv1d`, not on whether a verify path
runs, so both plans already sat on the correct side of the architectural question.

### Defect 1: staging and consumption are decided independently

`make_completion_chunk` feeds a splice into the decode window only when
`pending_ff_batch_width` returns `Some` (inputs_processor.rs:1573), which happens only for
`StagedBatchState::Homogeneous` - every sequence in the batch carries a splice and every splice has
the same length (speculative/staging.rs:14-38). `sample_and_add_toks_inner` calls
`Sequence::take_pending_ff_tokens` unconditionally (sampling.rs:835) and replays the splice
unconditionally (sampling.rs:872-876).

A batch where one sequence carries a splice and another does not, or where two sequences carry
splices of different lengths, yields `StagedBatchState::Mixed`. No splice reaches the decode
window, and every splice is still appended to its `Sequence`. A sequence holding a splice of
length K then grows by K+1 tokens while its KV cache grows by 1 position.

Splice lengths are data-dependent and vary per sequence, unlike the fixed
`num_speculative_tokens` width that makes `StagedBatchState::Homogeneous` the common case for
staged speculative proposals, so `StagedBatchState::Mixed` is the expected outcome for any batch
of two or more grammar-constrained sequences.

Consequences by cache backend:

- Default cache: the next step computes `start_pos` as `ctxt.len() - decode_window`
  (inputs_processor.rs:1608) from a token list that is K entries longer than the KV cache, so
  `seqlen_offsets` stays K ahead of the true cache length for the rest of the sequence. Rotary
  positions are wrong from that point on and K positions of KV are permanently absent. No error is
  raised.
- PagedAttention: `block_end` is computed as `block_start + query_len`
  (inputs_processor.rs:1666-1667) and reaches `panic!("Block table is too small (completion)!")`
  (inputs_processor.rs:1669) once the accumulated drift pushes `block_pos` past the allocated
  table.

The staged speculative path avoids the same defect because the paged scheduler enforces
homogeneity upstream by grouping completion batches on `active_staged_speculative_len`
(paged_attention/scheduler.rs:557, 565), and because the speculative driver clears stale proposals
when the fallback fires, as recorded in the comment at inputs_processor.rs:1560-1564. The
fast-forward path reuses `staged_batch_state_from_widths` (the downstream predicate) without
either the upstream grouping or the clear-on-fallback.

A fix must also roll the llguidance matcher back whenever a splice is discarded.
`Matcher::consume_ff_tokens` (llguidance-1.4.0 matcher.rs:146) computes and *consumes* the splice,
so the matcher is already advanced past the splice at stage time. Discarding a splice without
calling `Matcher::rollback` (llguidance-1.4.0 matcher.rs:93) leaves the matcher ahead of the
committed token sequence, and every subsequent token mask is computed for the wrong grammar
position.

Note that `make_completion_chunk` receives `input_seqs: &[&mut Sequence]` and cannot mutate a
`Sequence` through a shared reference, so the site that declines a splice is not currently able to
clear it. Any fix has to move the decision, the marker, or the mutability.

### Defect 2: PagedAttention reserves no slots for the widened window

`PagedAttentionScheduler::completion_token_cost` adds `active_staged_speculative_len` to
`num_uncomputed_tokens` (paged_attention/scheduler.rs:540-544) and the `allocate_slots` call site
computes `num_tokens` as `seq.len() + staged_speculative` or `seq.len() + 1`
(paged_attention/scheduler.rs:1186-1190). Neither expression includes
`active_pending_ff_tokens().len()`.

A step that feeds a splice of length K writes `1 + K` slot mappings
(inputs_processor.rs:1663-1686) against an allocation sized for 1. The write succeeds while all
`1 + K` positions fall inside the last already-allocated block and reaches
`panic!("Block table is too small (completion)!")` (inputs_processor.rs:1669) once the window
crosses a block boundary, so the failure is data-dependent on splice length and block fill level.

`PagedAttentionScheduler::_preempt` clears `staged_speculative_tokens` and increments
`mistralrs_speculative_staged_drops_total` (paged_attention/scheduler.rs:1375-1380) and performs
no equivalent action for `pending_ff_tokens`. A preempted sequence returns through the prefill
path, which does not consume `pending_ff_tokens`, and the surviving splice is then replayed by
`sample_and_add_toks_inner` after a step that computed no KV for the splice - the same
token-count/KV-length divergence described under Defect 1.

### Defect 3: sampling penalties read a token history that omits the splice

`sample_sequence` snapshots the penalty context as `seq.get_toks().to_vec()` (sampling.rs:1448) and
passes the snapshot to `Sampler::sample`. At a fast-forward step the splice has been decoded into
the KV cache but not yet appended to `Sequence`, because `apply_pending_ff_tokens` runs after
`sample_sequence` returns (sampling.rs:866-876). Repetition, frequency, presence and DRY penalties
therefore ignore the K tokens decoded in the same window, and `seq.generated_len()` read by
`activate_required_tool_call_grammar` (sampling.rs:1442) undercounts by K.

Ordering the replay before sampling removes the divergence and removes wasted work: a splice that
finishes the sequence (max_seq_len, a stop string, or an EOS token inside the splice) currently
still pays a full `sample_sequence` call whose result is then discarded by the `continue` at
sampling.rs:876.

Two constraints on the reorder:

- `Sequence::take_pending_ff_tokens` must still run before `sample_sequence`, because
  `sample_sequence` writes the *next* splice onto the same field (sampling.rs:1577).
- `sample_sequence` must not run for a sequence that the replay finished, because `sample_sequence`
  advances the llguidance matcher with `Matcher::consume_token` (sampling.rs:1566) and stages a new
  splice. Sampling futures are zipped positionally against logits rows
  (sampling.rs:841-861), so skipping a sequence requires carrying the logits row index rather than
  relying on position.

The `try_sample_batch_cuda` branch (sampling.rs:837) needs no change: `can_sample_batch_cuda`
returns `false` when any sequence carries a `SequenceRecognizer` other than `SequenceRecognizer::None`,
so no sequence reaching that branch can hold a splice.

### Defect 4: `Matcher::consume_ff_tokens` errors are discarded

`Matcher::consume_ff_tokens` (llguidance-1.4.0 matcher.rs:146-152) calls `consume_tokens` and
discards the `Result` with `let _ =`, returning the token vector even when consumption put the
matcher into `MatcherState::Error`. `sample_sequence` stages the returned vector without
re-checking matcher state (sampling.rs:1574-1578), so a faulted matcher still produces a committed
splice.

### Defect 5: fast-forward tokens carry a differently shaped `Logprobs`

`apply_pending_ff_tokens` constructs `Logprobs { token, logprob: 0.0, bytes: None, top_logprobs: None }`
(sampling.rs:738-743). `logprob: 0.0` encodes probability 1 under the grammar constraint. `bytes:
None` is harmless because `finish_or_add_toks_to_seq` recomputes completion bytes from the token
through `tok_trie().decode_ext` before calling `Sequence::add_token`. `top_logprobs: None` on a
request that set `return_logprobs` produces a response element shaped differently from every
sampled token in the same completion.

### Code verified correct, not to be changed by a fix

A fix pass driven by the defect list above can churn on the following, all of which were checked
during this review and hold:

- The GDN relaxation at gdn/backend.rs:836 is sound. `causal_conv1d_full` reads `&cache.conv_state`
  as left context and writes `cache.conv_state = new_conv_state` on the CUDA, Metal and CPU paths
  (gdn/backend.rs:970-1011), so a widened `RecurrentBatchKind::Decode` window continues the
  convolution state rather than restarting it.
- CUDA decode graph replay already excludes widened windows.
  `CudaGraphDecodeStep::one_token_continuation` returns `Ok(None)` when `q_len != 1` or
  `rows.query_len != 1` or `rows.decode_window != 1` (pipeline/cuda_graph.rs:600-602), so a
  fast-forward step falls back to the eager path with no additional guard.
- Retokenization at the splice boundary is handled inside llguidance and needs no engine-side
  backtracking. `TokenParser::ff_tokens` (llguidance-1.4.0 tokenparser.rs:677-726) prepends the
  bytes of the last emitted token to the forced bytes, retokenizes, falls back to retokenizing
  without the prefix when the result does not start with the existing token
  (llguidance-1.4.0 tokenparser.rs:700-712), and calls `tokenize_and_chop`
  (llguidance-1.4.0 tokenparser.rs:200) to drop trailing bytes a longer token could still extend.
- The `context_lens` narrowing to `(query_len - 1, 1)` (inputs_processor.rs:1630-1634) is correct
  and uniform across the batch, so `LogitsSelection::from_context_lens` resolves it to
  `LogitsSelection::Decode` (pipeline/mod.rs:1092, 1124-1132) rather than the ragged-span bail.
- `full_query_lens` (inputs_processor.rs:1588, 1619) is correctly kept separate from the narrowed
  `context_lens` for the `paged_single_token_decode` test (inputs_processor.rs:1710-1711) and for
  the paged block table `query_len` (inputs_processor.rs:1727-1729).
- `Which.GGUF` reaches the normal pipeline, so the `ff_bench.py` numbers do exercise the mechanism.
  `GGUFLoader::load_native_normal` (gguf.rs:566) delegates to `NormalLoaderBuilder` (gguf.rs:699)
  for any architecture with a native adapter, and `build_normal_pipeline` supplies
  `supports_grammar_fast_forward` from the env flag (normal.rs:292). The
  `supports_grammar_fast_forward: false` at gguf.rs:1415 applies only to the legacy `GGUFPipeline`
  fallback.

### Remediation constraints

A correct fix satisfies all of the following. Constraints are stated as invariants rather than as
a prescribed diff.

1. For every sequence and every step, the number of tokens appended to `Sequence` during the step
   equals the number of positions the forward pass computed KV for during the step.
2. Whenever a staged splice is not fed into the decode window, the splice is removed from
   `Sequence::pending_ff_tokens` and the sequence's `SequenceRecognizer::Llguidance` matcher is
   rolled back by the splice length via `Matcher::rollback` (llguidance-1.4.0 matcher.rs:93).
3. `PagedAttentionScheduler` reserves `active_pending_ff_tokens().len()` additional slots for any
   sequence whose splice will be fed on the step being scheduled.
4. `PagedAttentionScheduler::_preempt` leaves no sequence in `SequenceState::Waiting` holding a
   non-empty `Sequence::pending_ff_tokens`, and satisfies constraint 2 for any splice it removes.
5. The penalty context passed to `Sampler::sample` (sampling.rs:1448) includes every token decoded
   in the window whose logits are being sampled.
6. A sequence finished by splice replay does not reach `sample_sequence`.
7. Grouping completion batches by splice width is not used to satisfy constraint 1. Splice lengths
   are data-dependent per sequence, so grouping on splice width serializes multi-sequence batches
   down to one sequence per step.

### Validation gap

`ff_bench.py` (branch `ff-demo-artifacts`) sends one request at a time, on CPU, with no
PagedAttention, at `temperature=0.0`, under a regex that forces 100% of the completion. Defects 1
and 2 require concurrency or PagedAttention to manifest and Defect 3 requires sampling freedom, so
no defect above falls inside the measured configuration.

The byte-identical-output check that `plans/declarative-forging-flask.md` names as "the correctness
gate" cannot fail under the measured configuration: a regex forcing an exact passage under greedy
sampling pins both the flag-on and flag-off runs to the same string by construction, independent of
whether fast-forward engaged.

A check that discriminates: fixed seed, `temperature > 0`, a partially forcing grammar such as a
JSON schema, asserting token-for-token equality between `MISTRALRS_GRAMMAR_FAST_FORWARD=1` and
`MISTRALRS_GRAMMAR_FAST_FORWARD` unset. Defect 3 fails that check. Adding two concurrent requests
carrying different grammars exposes Defect 1, and enabling PagedAttention exposes Defect 2.

---

## Current State

- `Sequence::pending_ff_tokens: Vec<u32>` holds a grammar-forced splice between decode steps, with
  accessors `active_pending_ff_tokens`, `set_pending_ff_tokens`, `take_pending_ff_tokens`
  (sequence.rs:788, 1248-1258).
- `sample_sequence` stages a splice by calling `Matcher::consume_ff_tokens` after
  `Matcher::consume_token` commits the sampled token, gated on
  `GeneralMetadata::supports_grammar_fast_forward` and `!llg.is_stopped()` (sampling.rs:1573-1579).
- `make_completion_chunk` appends a staged splice to the decode window and widens `query_len` from
  1 to `1 + splice_len` (inputs_processor.rs:1597-1601, 1608-1619).
- `make_completion_chunk` narrows logit selection to `(query_len - 1, 1)` for a fast-forward window,
  resolving to `LogitsSelection::Decode` (inputs_processor.rs:1629-1634, pipeline/mod.rs:1092, 1124-1132).
- `make_completion_chunk` tracks `full_query_lens` separately from the narrowed `context_lens` for
  flash-attention metadata and paged block table width (inputs_processor.rs:1588, 1619, 1710-1711,
  1727-1729).
- `apply_pending_ff_tokens` replays each splice token through `finish_or_add_toks_to_seq`, halting
  on `!seq.is_running()` and returning whether the sequence finished (sampling.rs:727-747).
- `sample_and_add_toks_inner` removes a splice before sampling and replays the splice after
  sampling, skipping the sampled token when replay finished the sequence
  (sampling.rs:831-836, 866-880).
- `causal_conv1d` dispatches `RecurrentBatchKind::Decode` windows wider than one token to
  `causal_conv1d_full`, which continues convolution state from `cache.conv_state`
  (gdn/backend.rs:836, 970-1011).
- `CudaGraphDecodeStep::one_token_continuation` returns `Ok(None)` for any window wider than one
  token, routing fast-forward steps to the eager path (pipeline/cuda_graph.rs:600-602).
- `perf_flags::grammar_fast_forward_enabled` reads `MISTRALRS_GRAMMAR_FAST_FORWARD` once through a
  `OnceLock` and defaults to `false` (perf_flags.rs:5, 9, 35-37).
- `build_normal_pipeline` is the only `GeneralMetadata` construction site supplying
  `supports_grammar_fast_forward` from the env flag; the other six sites hardcode `false`
  (normal.rs:292, diffusion.rs:252, embedding.rs:706, ggml.rs:400, gguf.rs:1415,
  multimodal.rs:1531, speech.rs:330).
- `GGUFLoader::load_native_normal` delegates to `NormalLoaderBuilder`, so a GGUF model with a
  native adapter runs under `build_normal_pipeline` and reads the env flag (gguf.rs:566, 699).
- `TokenParser::ff_tokens` performs splice-boundary retokenization inside llguidance without
  engine-side backtracking, chopping trailing bytes a longer token could extend
  (llguidance-1.4.0 tokenparser.rs:677-726, 200).
- `cargo` build scripts fail to execute in the review container with `No such file or
  directory (os error 2)`, so `cargo check -p mistralrs-core` and `cargo test -p
  mistralrs-core` produce no build status for `d217eb32b` from the review session.

## Stubbed

- `make_completion_chunk` returns `anyhow::bail!("sequence has both a staged speculative proposal
  and a pending grammar fast-forward splice; these mechanisms are mutually exclusive")`, converting
  a scheduling state into a request failure raised during input construction
  (inputs_processor.rs:1603-1607).
- `pending_ff_batch_width` returns `None` for `StagedBatchState::Mixed` and
  `StagedBatchState::None`, suppressing splice feeding for the step while leaving every staged
  splice in `Sequence::pending_ff_tokens` (inputs_processor.rs:1529-1538).

## Missing

- No caller clears `Sequence::pending_ff_tokens` when `make_completion_chunk` declines to feed the
  splice - `sample_and_add_toks_inner` calls `Sequence::take_pending_ff_tokens` and
  `apply_pending_ff_tokens` without testing whether the splice reached the decode window, so a
  `StagedBatchState::Mixed` batch appends `splice_len + 1` tokens to a sequence whose KV cache grew
  by 1 position (inputs_processor.rs:1573, sampling.rs:835, 872-876).
- No caller rolls the llguidance matcher back when a splice is discarded, and
  `Matcher::consume_ff_tokens` already advanced the matcher past the splice at stage time
  (sampling.rs:1574, llguidance-1.4.0 matcher.rs:146, 93).
- `PagedAttentionScheduler::completion_token_cost` adds only `active_staged_speculative_len` to
  `num_uncomputed_tokens`, so a step feeding a splice of length K writes `1 + K` slot mappings
  against an allocation sized for 1 (paged_attention/scheduler.rs:540-544,
  inputs_processor.rs:1663-1686).
- The `allocate_slots` call site computes `num_tokens` from `seq.len()` plus
  `staged_speculative` or plus 1, with no `active_pending_ff_tokens` term
  (paged_attention/scheduler.rs:1186-1190).
- `PagedAttentionScheduler::_preempt` clears `staged_speculative_tokens` and performs no equivalent
  action for `pending_ff_tokens`, so a preempted sequence returns through the prefill path holding
  a splice that the prefill path does not consume (paged_attention/scheduler.rs:1373-1380).
- `Sampler::sample` receives a penalty context snapshotted before splice replay, omitting the K
  splice tokens decoded in the same window (sampling.rs:1448, 866-876).
- `sample_sequence` stages the splice returned by `Matcher::consume_ff_tokens` without re-testing
  `Matcher::is_stopped`, and `Matcher::consume_ff_tokens` discards the `Result` of its internal
  `consume_tokens` call with `let _ =` (sampling.rs:1573-1578, llguidance-1.4.0 matcher.rs:146-152).
- `apply_pending_ff_tokens` sets `top_logprobs: None` on every replayed token, producing a response
  element shaped differently from a sampled token when a request sets `return_logprobs`
  (sampling.rs:738-743).
- No test covers `pending_ff_batch_width`, matching the shape of
  `mixed_staged_widths_disable_batched_verification_input` which covers
  `staged_batch_state_from_widths` (inputs_processor.rs:1529, speculative/staging.rs:54).
- No test asserts that a `StagedBatchState::Mixed` batch leaves `Sequence` token count and KV cache
  length equal.

## Divergence

- normal.rs:289-291 states "Mechanism validated correct (byte-identical output vs. unpatched
  decode)"; the validating run is `ff_bench.py` on a single sequential request, on CPU, without
  PagedAttention, at `temperature=0.0`, under a regex forcing 100% of the completion
  (branch `ff-demo-artifacts`, `ff_bench.py`, `RESULTS.md`).
- `plans/declarative-forging-flask.md` names the byte-identical-output check as "the correctness
  gate"; a regex forcing an exact passage under greedy sampling produces the same string with the
  flag set and unset independent of whether fast-forward engaged (branch `ff-demo-artifacts`,
  `ff_bench.py`).
- pipeline/mod.rs:1315-1318 states that pipelines hardcoding `supports_grammar_fast_forward: false`
  "build their own inputs and never consume that field"; `GGUFLoader::load_native_normal`
  delegates to `NormalLoaderBuilder`, so a GGUF model with a native adapter runs under
  `build_normal_pipeline` and the `false` at gguf.rs:1415 governs only the legacy `GGUFPipeline`
  fallback (gguf.rs:566, 699, 1415, normal.rs:292).
- `RESULTS.md` reports "~6.1-6.2x faster" for the flag-on configuration; the measured grammar
  forces 100% of the completion, and `RESULTS.md` records the separate sumac measurement of ~3% for
  a tool-call grammar forcing only scaffold tokens (branch `ff-demo-artifacts`, `RESULTS.md`).
