# 2026-09-09: Grammar fast-forward remaining work

Scope: everything still outstanding on `grammar-fast-forward` (through `49848e501`, on top of
v0.9.3 `d5ae0f18f`) for the `MISTRALRS_GRAMMAR_FAST_FORWARD` mechanism. Self-contained; no prior
entry is needed to act on the tasks below.

---

## Long form

### The mechanism, in one pass

`Sequence::pending_ff_tokens` (sequence.rs:788) carries a grammar-forced token splice across one
decode step boundary. `sample_sequence` stages a splice by calling `Matcher::consume_ff_tokens`
after `Matcher::consume_token` commits the sampled token (sampling.rs:1628-1633), gated on
`GeneralMetadata::supports_grammar_fast_forward` (pipeline/mod.rs:1319), which
`build_normal_pipeline` alone populates from `MISTRALRS_GRAMMAR_FAST_FORWARD`
(normal.rs:296, perf_flags.rs:35-37) and the six other `GeneralMetadata` construction sites
hardcode to `false`.

On the next step `resolve_pending_ff_batch` (inputs_processor.rs:1547) decides which splices the
batch can carry, `make_completion_chunk` appends the surviving splices to the decode window
(inputs_processor.rs:1619-1623) widening `query_len` from 1 to `1 + splice_len`, and
`sample_and_add_toks_inner` replays each splice into its `Sequence` before sampling
(sampling.rs:844-855).

`Matcher::consume_ff_tokens` (llguidance-1.4.0 matcher.rs:146) consumes as well as computes, so a
staged splice has already advanced the matcher. Any code path that drops a splice therefore also
rolls the matcher back, which is what `Sequence::discard_pending_ff_tokens` (sequence.rs:1265)
exists for.

### Task 1: populate `bytes` on replayed tokens

`apply_pending_ff_tokens` sets `Logprobs::bytes` to `None` and the nested `TopLogprob::bytes` to
`None` (sampling.rs:745, 750). `finish_or_add_toks_to_seq` copies `Logprobs::bytes` into
`ResponseLogprob::bytes` through `logprob.bytes.clone().map(|b| b.into_bytes())`
(sampling.rs:525), so a completion answering a request with `return_logprobs` set carries a null
`bytes` on every grammar-forced token and a populated `bytes` on every sampled token.

`finish_or_add_toks_to_seq` already decodes the same token one line later through
`tok_env.tok_trie().decode_ext(&[logprobs.token], include_special)` (sampling.rs:205-207) to build
the `completion_bytes` it hands `Sequence::add_token`, so the string a replayed token needs is
derivable at the point `apply_pending_ff_tokens` runs, from `this.get_metadata().tok_env()`.

A sampled token reaches `Logprobs::bytes` as `Some(String)` from `Sampler::sample`, so matching
that shape for a replayed token removes the divergence.

### Task 2: decide the failure mode when a splice rollback fails

`Sequence::discard_pending_ff_tokens` logs `tracing::warn!` and returns when
`Matcher::rollback` fails (sequence.rs:1271-1276). A sequence continuing past a failed rollback
holds a matcher advanced past tokens the sequence never emitted, and every subsequent
`compute_mask` on that sequence answers for the wrong grammar position, so the completion
continues and its output is wrong.

`Matcher::rollback` reaches `TokenParser::rollback` (llguidance-1.4.0 tokenparser.rs:381-418),
which returns `Err` on exactly two conditions: `n_tokens > self.llm_tokens.len()`, and
`check_initialized` failing because the parser sits in an error state. Neither is reachable for a
splice staged by `sample_sequence`: `Matcher::consume_ff_tokens` pushed the splice onto
`llm_tokens` before `set_pending_ff_tokens` recorded it, and `sample_sequence` refuses to stage a
splice when `llg.is_error()` holds (sampling.rs:1631). Nothing between staging and discarding
consumes a token on that matcher, and `activate_required_tool_call_grammar` replaces a recognizer
only when `matches!(seq.recognizer, SequenceRecognizer::None)` holds (sampling.rs:64, 70-71).

The repository style is to trust internal guarantees rather than to handle unreachable states, so
the branch either goes away or stops being lenient. Continuing with a corrupted matcher produces
wrong output behind a log line, so putting the sequence into an error state costs one line and
removes the silent-wrong-output outcome.

### Task 3: two tests

The existing tests colocate in `mod tests` at the bottom of the file holding the code, name
themselves after the invariant they hold (`padded_decode_rows_alias_row_zero_without_kv_writes`,
`mixed_staged_widths_disable_batched_verification_input`), and build sequences through a local
fixture rather than a shared harness. Two tests cover the two invariants that the
single-sequence-CPU benchmark cannot reach.

Test A, in `mistralrs-core/src/pipeline/inputs_processor.rs`: a batch whose sequences hold splices
of differing lengths leaves every `Sequence::active_pending_ff_tokens` empty after
`resolve_pending_ff_batch`, and a batch whose sequences hold splices of equal length leaves every
splice in place. `resolve_pending_ff_batch` takes `&mut [&mut Sequence]` and needs no pipeline,
device or tokenizer. The `mod tests` block in inputs_processor.rs builds no `Sequence` today, so
the test needs a local fixture; `DefaultScheduler`'s `test_sequence`
(scheduler/default_scheduler.rs:385) and `PagedAttentionScheduler`'s `test_sequence`
(paged_attention/scheduler.rs:1807) are the two shapes already in the repository to follow.

Test B, in `mistralrs-core/src/paged_attention/scheduler.rs`: a running sequence holding a splice
of length K is allocated `seq.len() + K` slots.
`completion_batches_bootstrap_new_rows_without_dropping_staged_rows`
(paged_attention/scheduler.rs:4074) is the same test shape over `set_staged_speculative`, and
`test_scheduler` plus `test_sequence` are already in that `mod tests` block.

Neither test asserts on matcher state. No test in the repository constructs an
`llguidance::Matcher`, and doing so needs a tokenizer and a `ParserFactory`, so the rollback half
of `Sequence::discard_pending_ff_tokens` stays uncovered rather than pulling a new fixture shape
into the crate.

### Not a task: fast-forward yields nothing on a multi-sequence batch

`resolve_pending_ff_batch` discards every splice in the batch unless
`staged_batch_state_from_widths` returns `StagedBatchState::Homogeneous`, which needs every
sequence in the batch to hold a splice of identical length (speculative/staging.rs:17-38). Splice
lengths follow each sequence's grammar position, so two sequences agree only when their grammars
and positions agree, and `StagedBatchState::Mixed` is the ordinary outcome for a batch of two or
more grammar-constrained sequences.

The step still pays `Matcher::consume_ff_tokens` per sequence, which forces bytes and tokenizes
them, plus `Matcher::rollback` per sequence, which restores parser bytes and clears the parser
caches. `mistralrs_grammar_ff_splice_drops_total` (sequence.rs:1278) counts each discard.

Output stays correct under every batch composition, so no correctness work follows from the above.
The available responses are a scope statement on the flag, which reads today as an unqualified
"opt in via MISTRALRS_GRAMMAR_FAST_FORWARD=1 once you've measured it's a win for your own grammar
shape and hardware" (normal.rs:289-295), and ragged-width support, which needs the decode window
padded to the widest splice with the pad positions kept out of the KV cache and out of
`slot_mapping` (inputs_processor.rs:1685-1711), against an `input_width` check that rejects rows of
differing `query_len` (inputs_processor.rs:1664-1670). `LogitsSelection::from_context_lens`
already accepts a per-row `start` at a uniform `len` (pipeline/mod.rs:1135-1149), so logit
selection needs no change under ragged widths.

`mistralrs_grammar_ff_splice_drops_total` under a real multi-sequence grammar load answers whether
ragged-width support is worth its size.

### Verified, and not to be changed

- `causal_conv1d` dispatching a `RecurrentBatchKind::Decode` window wider than one token to
  `causal_conv1d_full` (gdn/backend.rs:836) continues convolution state rather than restarting it:
  `causal_conv1d_full` reads `&cache.conv_state` and writes `cache.conv_state = new_conv_state` on
  the CUDA, Metal and CPU paths (gdn/backend.rs:970-1011).
- `CudaGraphDecodeStep::one_token_continuation` returns `Ok(None)` when `q_len != 1` or
  `rows.query_len != 1` or `rows.decode_window != 1` (pipeline/cuda_graph.rs:600-602), so a
  fast-forward step takes the eager path with no further guard.
- Splice-boundary retokenization happens inside llguidance and needs no engine-side backtracking:
  `TokenParser::ff_tokens` (llguidance-1.4.0 tokenparser.rs:677-726) prepends the bytes of the last
  emitted token to the forced bytes, retokenizes, retokenizes again without the prefix when the
  result does not start with the existing token, and calls `tokenize_and_chop`
  (llguidance-1.4.0 tokenparser.rs:200) to drop trailing bytes a longer token could extend.
- `make_completion_chunk` narrowing logit selection to `(query_len - 1, 1)` for a fast-forward
  window (inputs_processor.rs:1651-1656) is uniform across the batch and resolves to
  `LogitsSelection::Decode` (pipeline/mod.rs:1125-1132).
- `full_query_lens` (inputs_processor.rs:1610, 1641) stays separate from the narrowed
  `context_lens` for the `paged_single_token_decode` test (inputs_processor.rs:1732-1733) and for
  the paged block table `query_len` (inputs_processor.rs:1749-1751).
- `try_sample_batch_cuda` takes `&[&mut Sequence]` (sampling.rs:1335) and
  `can_sample_batch_cuda` returns `false` for any sequence whose `recognizer` is not
  `SequenceRecognizer::None`, so no sequence reaching the CUDA batched sampler holds a splice.
- `should_sample_step` returns `true` for every `is_prompt == false` step (pipeline/mod.rs:1700-1706),
  so a completion step that feeds a splice always reaches `sample_and_add_toks_inner` and replays
  it.
- The one `process_inputs` call that passes a subset of the batch passes `active_input_seqs` for
  prompt chunking (pipeline/mod.rs:2464-2470), where `is_prompt` holds and
  `resolve_pending_ff_batch` correctly does not run.
- `Which.GGUF` reaches `build_normal_pipeline`: `GGUFLoader::load_native_normal` (gguf.rs:566)
  delegates to `NormalLoaderBuilder` (gguf.rs:699) for any architecture with a native adapter, so
  the `supports_grammar_fast_forward: false` at gguf.rs:1415 governs only the legacy `GGUFPipeline`
  fallback.

### Build status

No `cargo` invocation completes in the review container: build scripts fail to execute with
`No such file or directory (os error 2)`. Build and test status for `49848e501` comes from the
implementing session, not from the reviewing session.

---

## Current State

- `Sequence::pending_ff_tokens` holds a grammar-forced splice between decode steps, with
  `active_pending_ff_tokens`, `set_pending_ff_tokens`, `take_pending_ff_tokens` and
  `discard_pending_ff_tokens` (sequence.rs:788, 1248-1279).
- `Sequence::discard_pending_ff_tokens` drops a splice, calls `Matcher::rollback(splice.len())`,
  and increments `mistralrs_grammar_ff_splice_drops_total` (sequence.rs:1265-1279).
- `resolve_pending_ff_batch` runs once per completion step from
  `TextInputsProcessor::process_inputs` and discards every splice in the batch unless
  `staged_batch_state_from_widths` returns `StagedBatchState::Homogeneous`
  (inputs_processor.rs:1547-1558, 2669).
- `PagedAttentionScheduler::completion_token_cost` and the `allocate_slots` call site both add
  `active_pending_ff_tokens().len()`, and `PagedAttentionScheduler::_preempt` calls
  `Sequence::discard_pending_ff_tokens` (paged_attention/scheduler.rs:543, 1188-1192, 1388).
- `sample_and_add_toks_inner` replays each sequence's splice before sampling, records completions
  in `finished_mask`, and samples only the sequences the replay leaves running
  (sampling.rs:844-863, 904-908).
- `sample_sequence` stages a splice only when `!llg.is_error()` holds after
  `Matcher::consume_ff_tokens` returns (sampling.rs:1628-1638).
- `apply_pending_ff_tokens` builds a single-entry `TopLogprob` vector for a replayed token when
  `seq.return_logprobs()` holds (sampling.rs:735-752).
- `MISTRALRS_GRAMMAR_FAST_FORWARD` defaults to `false` through a `OnceLock` in
  `perf_flags::grammar_fast_forward_enabled` (perf_flags.rs:5, 9, 35-37).

## Stubbed

- `Sequence::discard_pending_ff_tokens` logs `tracing::warn!` and returns when `Matcher::rollback`
  fails, leaving the sequence running with a matcher advanced past tokens the sequence never
  emitted (sequence.rs:1271-1276).
- `make_completion_chunk` returns `anyhow::bail!("sequence has both a staged speculative proposal
  and a pending grammar fast-forward splice; these mechanisms are mutually exclusive")`, converting
  a scheduling state into a request failure raised during input construction
  (inputs_processor.rs:1625-1629).

## Missing

- `apply_pending_ff_tokens` sets `Logprobs::bytes` and `TopLogprob::bytes` to `None`
  (sampling.rs:745, 750), and `finish_or_add_toks_to_seq` copies `Logprobs::bytes` into
  `ResponseLogprob::bytes` (sampling.rs:525), so a completion answering a `return_logprobs` request
  carries a null `bytes` on grammar-forced tokens and a populated `bytes` on sampled tokens.
- No test covers `resolve_pending_ff_batch`; `mod tests` in inputs_processor.rs constructs no
  `Sequence` (inputs_processor.rs:1547, 2887).
- No test covers slot reservation for a sequence holding a splice, alongside
  `completion_batches_bootstrap_new_rows_without_dropping_staged_rows` which covers the same
  reservation for `set_staged_speculative` (paged_attention/scheduler.rs:4074).
- No test constructs an `llguidance::Matcher`, so the `Matcher::rollback` half of
  `Sequence::discard_pending_ff_tokens` has no coverage (sequence.rs:1265).
- Ragged-width fast-forward has no code: `make_completion_chunk` rejects rows of differing
  `query_len` (inputs_processor.rs:1664-1670), so a batch mixing splice lengths carries no splice
  at all.

## Divergence

- normal.rs:289-295 documents `MISTRALRS_GRAMMAR_FAST_FORWARD=1` as an unqualified opt-in to be
  measured per grammar shape and hardware; `resolve_pending_ff_batch` discards every splice on a
  batch that is not `StagedBatchState::Homogeneous`, so a multi-sequence grammar batch reaches the
  same token count and the same output as with the flag unset (inputs_processor.rs:1547-1558).

---

## Resolved

Commit `3b37de65e` on `grammar-fast-forward` closes Tasks 1, 2 and 3. The "Not a task" section and
the `Divergence` section below it still hold.

- Task 1 closes through `apply_pending_ff_tokens` decoding each replayed token with
  `tok_env.tok_trie().decode_ext(&[token], include_special)` and setting both `Logprobs::bytes` and
  `TopLogprob::bytes` to the decoded string (sampling.rs:752, 757, 762), matching what
  `finish_or_add_toks_to_seq` derives for a sampled token.
- Task 2 closes through `Sequence::discard_pending_ff_tokens` calling
  `set_state(SequenceState::Error)` when `Matcher::rollback` returns `Err` (sequence.rs:1284), with
  the unreachability argument recorded in the branch rather than the branch removed.
- Task 3 closes through three tests rather than two:
  `resolve_pending_ff_batch_discards_mismatched_splice_widths` and
  `resolve_pending_ff_batch_keeps_equal_width_splices` over a local `ff_test_sequence` fixture
  (inputs_processor.rs:3235, 3294, 3307), and
  `completion_batches_reserve_slots_for_pending_fast_forward_splices` asserting on
  `get_block_ids` rather than on the reservation arithmetic
  (paged_attention/scheduler.rs:4120).
- The `Matcher::rollback` half of `Sequence::discard_pending_ff_tokens` stays uncovered, as stated
  under Task 3.
- The flag's doc comment (normal.rs:289) states that a multi-sequence grammar-constrained batch
  ordinarily feeds no splice, which closes the `Divergence` entry below.
- No `cargo` invocation completes in the review container, so build and test status for
  `3b37de65e` comes from the implementing session.
