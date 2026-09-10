# 06-E -- static trace of the widened-decode-window / KV-row accounting for the B/N=1 nested FF bug (2026-09-10, ninth session)

Status: **deep static trace only, by explicit user choice this session (see Sec 0). No source edits,
no rebuild, no new server requests. Traced the entire Rust-level path from splice staging through the
widened decode window, PagedAttention context-length/slot-mapping construction, logits-row selection,
`num_computed_tokens` bookkeeping, CUDA-graph-decode-tail eligibility, down to the Rust wrapper around
the CUDA `paged_attention` kernel call. Found no confirmed accounting divergence anywhere in this
chain. This narrows, but does not close, the investigation -- read Sec 10-11 before treating this as
final.**

## 0. Environment constraint and the choice made this session

This container has no GPU, no `cargo`/`rustc`, and no access to the host server's process, logs, or
stdout -- confirmed again this session (`nvidia-smi`: not found; `which cargo rustc`: empty). A live
server is reachable read-only at `host.docker.internal:1234` (`/health`, `/metrics`, `/v1/models` all
`200`), but there is no log-access endpoint and no way from this container to trigger a host rebuild
or restart.

The task as given asked for source instrumentation, a rebuild, and a live N=1 OFF/ON comparison. Since
this container cannot do that autonomously, the user was asked how to proceed
(`AskUserQuestion`, this session) and chose: **"Go as deep as possible via static analysis only"** --
no `/workspace` source edits, continue tracing the exact code path by reading, and write this report
documenting what static analysis alone can and cannot establish, explicitly flagging live
instrumentation as a deferred next step rather than performing it.

Consequently, sections 2-6 below (which the task's deliverable template asks for as "traces") are
**derived from reading the code that would execute for the established reproduction**, not from a new
live run. The empirical facts they build on (exact splice width, metric deltas, output bytes) are
those already captured and committed in `06-b-n1-repro.md` and `06-c-root-cause.md` -- no new HTTP
requests were made this session.

## 1. Executive conclusion

Traced, by direct source reading against `/workspace` (`grammar-fast-forward` @ `cab8cd3ae`,
untouched), every Rust-level mechanism between "FF splice staged" and "first real token sampled after
it": widened-window construction (`inputs_processor.rs`), lm_head-narrowing logits selection
(`pipeline/mod.rs`'s `LogitsSelection`), `num_computed_tokens` advancement (`engine/mod.rs`),
CUDA-graph decode-tail/lookahead eligibility (`sampling.rs`'s `cuda_token_sampling_plan`), and the
Rust-side PagedAttention dispatch down to the CUDA kernel's own row/context-length contract
(`paged_attention/layers/paged_attention.rs`, `mistralrs-paged-attn`'s CUDA backend wrapper). **No
confirmed divergence was found anywhere in this chain.** Every invariant the task asked about (window
width, position IDs, which logits row gets selected, KV-cache slot assignment, `num_computed_tokens`
bookkeeping, whether the CUDA fast-path could silently mis-advance) checked out consistent on paper,
including one piece of new, concrete evidence this session did not have before: **grammar-constrained
sequences are structurally excluded from every CUDA-graph fast/lookahead decode path** in this
codebase (`cuda_token_sampling_plan`, `sampling.rs:1032-1039`, requires `SequenceRecognizer::None`),
so the `account_cuda_decode_rows` hard-coded `advance_num_computed_tokens(1)` that `06-c` flagged as
an untraced loose thread **cannot run** for our repro's sequence at all -- that specific candidate is
now ruled out, not just untested.

This pushes the investigation toward Success Criterion B ("model/KV state appears equivalent") **as
far as static Rust-source reading can establish**, with two explicit, named boundaries this session
did not cross: the raw CUDA/C++ kernel body underneath the `paged_attention()` FFI call, and any
actual runtime values (no live instrumentation was run, per the scoping decision in Sec 0). The `\r`
degeneration remains **unexplained** by anything found this session -- per the task's explicit
instruction not to overfit to the symptom, this report does not attempt to explain `\r` specifically;
it reports what was and was not found in the accounting layer.

## 2. Exact reproduction referenced (not re-run this session)

Unchanged from `06-b-n1-repro.md`/`06-c-root-cause.md`: Qwen3.5-4B-GGUF, CUDA, N=1, temperature 0,
`plans/ff-round-two/fixtures/nested.schema.json`, seed 42, `max_tokens=64`, fresh server restart per
leg. FF-OFF: 5/5 identical, `finish_reason=stop`, 47 tokens, valid JSON. FF-ON: 5/5 identical,
`finish_reason=length`, 64 tokens, degenerates to repeated `\r` immediately after `"run":`. Exactly one
splice staged and fed, width 1 token (`mistralrs_grammar_ff_splices_staged_total` delta 1,
`tokens_fed_total` delta 1, `splice_drops_total`/`tokens_dropped_total` delta 0). No new requests were
sent this session; these facts are read from the already-committed reports and raw JSON, not re-derived.

## 3. Instrumentation added

**None.** Per Sec 0, the user chose static analysis only. No `tracing::debug!`/`eprintln!` or any
other source change was made to `/workspace` at any point this session. `/workspace` remained on
`grammar-fast-forward` @ `cab8cd3ae`, clean, throughout (verified at session start and can be
re-verified: `git status --porcelain=v1` returns nothing).

## 4. FF-OFF: traced code path (no widening)

For FF-OFF, `supports_fast_forward` is false at `sampling.rs:1634`, so `llg.consume_ff_tokens()` is
never called and `seq.pending_ff_tokens` stays empty for the whole generation. Every decode step
therefore takes the *ordinary* width-1 path through `inputs_processor.rs:1589-1631`:
`use_pending_ff` is false (`pending_ff_batch_width` sees no active splice anywhere in the batch), so
`pending_ff = &[]`, `narrow_for_ff = false`, `context_lens.push((0, query_len))` with `query_len == 1`
(just the single backlog token, the one most recently sampled). `LogitsSelection::from_context_lens`
(`pipeline/mod.rs:1093-1162`) sees `context_lens == [(0, seq_len)]` for every sequence and returns
`Self::All` (`pipeline/mod.rs:1123-1125`), so `select()` (`pipeline/mod.rs:1224`) is a no-op clone --
the model's one computed position's logits pass straight through to `extract_logits`/`self.output`.
This is the simple, well-trodden ordinary-decode path; nothing here is specific to this investigation.

## 5. FF-ON: traced code path for the widened window (derived from source, not re-run)

For the one decode step immediately following the splice (established by `06-c`: staged right after
`"run":`, replayed on the next step), based on the metric evidence (splice width 1, no speculative
decoding active for this request) the traced code does the following:

- **Window construction** (`inputs_processor.rs:1589-1631`): `use_pending_ff` is true for this one
  step (`pending_ff_batch_width` sees a uniform width-1 splice across the N=1 batch). `pending_ff =
  seq.active_pending_ff_tokens()` (the 1 staged token, still present -- `apply_pending_ff_tokens`'s
  `take_pending_ff_tokens()` runs *later*, in `sample_and_add_toks_inner`, after this step's forward
  pass has already run; see Sec 9 for why this ordering is safe). `ctxt` (the model input for this
  sequence) is assembled as `[backlog_token] then [ff_token]` -- `ctxt[start_pos..]` (the one
  not-yet-KV-computed real token from the previous step) followed by `ctxt.extend(pending_ff...)`
  (line 1608-1611) -- i.e., **chronological order**, backlog before splice, not reversed.
  `query_len = host_width = 2`. `narrow_for_ff = true` (line 1625, since `pending_ff` is non-empty and
  no speculative decoding is active), so `context_lens.push((query_len - 1, 1)) == (1, 1)` (line
  1626-1627) -- "only the position after both new tokens needs a sampled logit."
- **Position IDs**: `effective_context_len = start_pos + query_len` (line 1620, pushed to
  `position_ids`), later expanded by `decode_positions_tensor` (`pipeline/mod.rs:1054-1071`) into the
  two consecutive RoPE positions `[start_pos, start_pos+1]` for the two window rows -- backlog token
  gets the earlier position, ff token the later one, matching input order.
- **PagedAttention slot mapping and context lengths** (`inputs_processor.rs:1654-1701`): both new
  rows get real KV-cache slots (`block_start..block_end`, width `query_len == 2`, line 1659-1682), and
  **per-row, monotonically increasing** context lengths are built for both (`for row in 0..query_len {
  full_context_len = start_pos + row + 1; ... }`, line 1684-1700) -- row 0 (backlog) gets
  `start_pos+1`, row 1 (ff token) gets `start_pos+2`. Both rows' slots and context lengths are
  distinct and correctly ordered; nothing collapses them into one.
- **Attention dispatch** (`models/quantized_llama.rs:59-125`, `forward_attn`): the whole
  `seq_len == 2` q/k/v goes into a single `paged_attn.forward(...)` call (line 95-108) -- no
  case-split between "the two new tokens attending to each other" and "attending to the cache" at
  this layer.
- **PagedAttention's own dispatch** (`paged_attention/layers/paged_attention.rs:2031-2138`,
  `forward_impl`): tries `try_prefix_gather_prefill`, then `try_regular_prompt` (gated on prompt/mask
  conditions that don't apply mid-generation), then falls through to `run_decode` (line 2137) --
  confirmed by reading `try_regular_prompt`'s guard (`custom_decode` / `AttentionMask::None` checks,
  lines 1592-1601) that a mid-generation widened window is not a "prompt," so it reaches `run_decode`.
- **`run_decode`** (`paged_attention/layers/paged_attention.rs:1648-1732`): explicitly handles
  `ctx.dims.seq_len > 1` (line 1656) by **flattening** `[batch, heads, seq_len, head_size]` into
  `[batch*seq_len, heads, head_size]` -- i.e., it treats the 2 window rows as 2 independent decode
  rows in a widened "virtual batch," each addressed by its own `context_lens`/`block_tables` entry
  (not by any notion of "these two rows belong to the same sequence"). Critically, **the KV-cache
  write happens before this dispatch** (`write_kv_cache(...)`, lines 1676-1688, runs before the
  `match DecodePlan::choose(...)` that picks the actual kernel, line 1704) -- so by the time row 1 (the
  ff token, `context_len = start_pos+2`) is read back out of the cache for attention, row 0's
  (backlog token's) key/value have *already* been physically written to their slot. A row-based decode
  kernel that gathers "cache contents up to this row's own `context_len`" therefore sees the correct,
  causally-ordered history for both rows, purely because of write-before-read ordering, not because
  the kernel has any special multi-token-per-sequence logic.
- **The CUDA kernel wrapper's own contract** (`mistralrs-paged-attn/src/cuda/backend/paged_attention.rs:250,290`):
  reads `num_seqs` directly off the query tensor's first (already-flattened) dimension and requires
  `context_lens` to have exactly one entry per that same dimension -- i.e., the kernel's own notion of
  "sequence" is really "decode row," fully generic over how many rows one logical `Sequence` supplies,
  as long as the Rust side (confirmed above) supplies correct, monotonically increasing per-row
  context lengths and correctly pre-written cache slots. This is not FF-specific plumbing: the exact
  same `ctxt.extend(...)`/per-row `context_lens` construction in `inputs_processor.rs:1608-1700` is
  shared with staged speculative-decoding proposals (`staged_speculative`, extended into `ctxt` right
  before `pending_ff`, same function) -- speculative decoding routinely uses widths > 1 through this
  identical mechanism, which is evidence (not proof) that the underlying multi-row-decode contract is
  an established, already-exercised pattern in this codebase, not a code path unique and novel to FF.
- **Logits selection**: `extract_logits` (`pipeline/mod.rs:2980-2986`) is applied to the *hidden
  states* `x` (post final-norm, pre-lm_head -- `models/quantized_llama.rs:373-375`), not to
  already-projected logits, so "narrow for FF" is a genuine compute-skipping optimization, not a
  post-hoc re-selection. `LogitsSelection::from_context_lens` with `context_lens == [(1, 1)]` and
  `batch == 1` resolves to `Self::Decode { start: 1, len: 1 }` (`pipeline/mod.rs:1130-1133`), and
  `select()` does `logits.narrow(1, 1, 1)` (`pipeline/mod.rs:1230`) -- **the last of the two window
  rows**, i.e., the hidden state computed after attending through both the backlog token and the ff
  token. This is exactly the position needed to predict the token that should follow the ff splice.
- **`num_computed_tokens` advancement**: two guarded blocks exist in `engine/mod.rs`
  (2000-2008 inside the CUDA-step-completion arm, 2109-2117 unconditionally after it), both gated
  `if seq.num_computed_tokens() == before { seq.advance_num_computed_tokens(scheduled) }` against the
  *same* `before`/`scheduled` snapshot -- an idempotent "advance exactly once" pattern, not a
  double-advance. But see Sec 8: this session did not confirm which of the two arms actually executes
  for this backend/request shape, only that whichever does, `scheduled` should equal the widened
  `query_len` (2), not 1 -- not independently re-derived this session, inherited from `06-c`'s citation
  of the same lines.
- **CUDA-graph decode-tail/lookahead exclusion** (new this session, see Sec 6): confirmed this
  specific sequence can never enter either CUDA fast-path, so the *hard-coded* `advance_num_computed_tokens(1)`
  in `account_cuda_decode_rows` (`engine/mod.rs:921-926`) is not reachable for it.

## 6. New this session: grammar-constrained sequences cannot enter the CUDA-graph decode-tail/lookahead paths

`06-c` (Sec 7) flagged `engine/mod.rs`'s `account_cuda_decode_rows`/`continue_cuda_decode_batch` as an
untraced loose thread specifically because `account_cuda_decode_rows` (`engine/mod.rs:921-926`)
unconditionally does `advance_num_computed_tokens(1)` per row with **no reference to `pending_ff_tokens`
or window width at all** -- exactly the shape of bug the task's hypothesis list names ("model/KV
position not advanced [enough]"). This session traced the eligibility gates for both places this fast
path can be entered:

- The resident CUDA decode-tail loop (`engine/mod.rs:979-1104`, `continue_cuda_decode_batch`) is only
  continued/launched when `sampling::can_submit_cuda_token_batch_seqs(seqs)` is true
  (`pipeline/execution.rs:352`, `engine/mod.rs` call sites), which loops every sequence through
  `cuda_token_sampling_plan(seq)` (`sampling.rs:1173-1186`) and returns `false` for the whole batch the
  moment any sequence fails it.
- The one-shot "launch one token ahead" attempt inside the ordinary `submit_step` path
  (`pipeline/mod.rs:2374-2383`, `cuda_decode_lookahead`) requires the *same*
  `sampling::can_submit_cuda_token_batch_seqs(input_seqs)` (line 2379) in addition to
  `can_launch_one_token_lookahead`.
- `cuda_token_sampling_plan` (`sampling.rs:1032-1039`) returns `None` -- excluding the whole batch --
  if `!matches!(&seq.recognizer, SequenceRecognizer::None)`. Our sequence's recognizer is
  `SequenceRecognizer::Llguidance(_)` for the entire grammar-constrained generation (established by
  `06-c`), so this condition is true and `cuda_token_sampling_plan` returns `None` for it on every
  single decode step, start to finish.

**Consequence:** for this repro (and for any grammar-constrained request at all, FF-ON or FF-OFF),
neither CUDA fast path is ever entered. `account_cuda_decode_rows`'s hard-coded `+1` advance --
06-c's leading untraced suspicion -- is structurally unreachable here. This rules the CUDA-graph
decode-tail candidate **out**, not merely "untested." The sequence always goes through the generic
`forward_inputs`/`extract_logits`/`sample_causal_gen` path traced in Sec 5, on every step, in both
FF-OFF and FF-ON.

## 7. Side-by-side: what the first real sample after the splice should see (per source, not observed live)

```text
OFF (ordinary path, no splice ever occurs for this sequence):
  every step: query_len=1, context_lens=(0,1) -> LogitsSelection::All, no narrowing
  model/KV position advances by exactly 1 per step, in lockstep with seq.tokens

ON (this one step, per Sec 5's trace):
  window: [backlog_token, ff_token], query_len=2
  RoPE positions: [P, P+1] (P = start_pos, i.e. the backlog token's position)
  PagedAttention context_lens (per row): [P+1, P+2] -- both rows get real, distinct KV slots
  KV cache write: BOTH rows written before either is read back for attention
  logits selection: narrow(dim=1, start=1, len=1) -- the hidden state after both new tokens
  -> first real sample should be conditioned on a hidden state that attended through
     [..existing cache.., backlog_token, ff_token], i.e. logically the same context OFF's
     model would have if it had reached this exact token sequence by two ordinary single-token
     steps instead of one widened one.
```

Per this trace, **no divergence was found** between what OFF's model would see at the equivalent
logical position and what ON's widened-window construction supplies to the model for the first real
sample. This is the Sec 1 conclusion, restated concretely: the `?` in the task's comparison template
resolves, per static reading, to "the same logical position as OFF, reached by a different physical
mechanism (one widened forward pass instead of two ordinary ones)" -- not to a different or wrong
position.

## 8. First concrete divergence: none found

No divergence was found in the traced Rust-level accounting. This is stated plainly, per the task's
explicit instruction not to claim a root cause the evidence doesn't establish, and per Success
Criterion B's instruction to keep tracing rather than declare victory prematurely. Two honest
boundaries where this session's tracing stopped, not because anything there was ruled out, but because
going further requires tools this session did not use:

1. **The raw CUDA/C++ kernel body.** This session read the Rust *wrapper* around the CUDA
   `paged_attention()` call (`mistralrs-paged-attn/src/cuda/backend/paged_attention.rs:250-390,453+`)
   and confirmed its *declared* row/context-length contract is generic enough to support the widened
   window correctly, given correct inputs (which Sec 5 confirms the Rust side supplies). It did **not**
   read the actual `.cu`/PTX kernel source that contract wraps -- that source may not even be in this
   repository (candidate for an external/vendored dependency; not checked this session per the
   instruction to avoid unnecessary broad work). A kernel-internal bug specific to `num_seqs > 1` rows
   sharing one logical sequence's causal history within a single launch (as opposed to `num_seqs`
   independent real sequences, which is the kernel's much more common use case) cannot be ruled out by
   this session's evidence.
2. **No runtime values were observed.** Every claim in Sec 4-7 is "this is what the code, read
   statically, computes/does for this input shape" -- not "this is what was measured to happen." Any
   defect that depends on the actual numeric values used at runtime (an off-by-one that only manifests
   for a specific `start_pos`/block-boundary combination, a race between the async pipeline lock and a
   cache write, a dtype/precision issue) is invisible to this method by construction. This is exactly
   the gap live instrumentation (Sec 0's declined option) would close.

## 9. One ordering question this session resolved explicitly

The task's hypothesis list includes "interaction between `pending_ff_tokens` replay and the subsequent
real sample." This session confirmed the exact ordering within one decode step: `seq.pending_ff_tokens`
is read (non-destructively, `active_pending_ff_tokens()`) by `inputs_processor.rs` to build the
widened window **before** the forward pass runs, and only **taken** (destructively,
`take_pending_ff_tokens()`, `sequence.rs:1254-1256`) afterward, inside `sample_and_add_toks_inner`
(`sampling.rs:857`), which runs after `forward_inputs` has already produced the widened logits
(`pipeline/mod.rs:2074-2075` -> `2183`, or the PagedAttention arm's equivalent at `2600-2760`). This
ordering is safe and non-racy for a single request: nothing else can mutate `seq.pending_ff_tokens`
between the read and the take within one step (no evidence of concurrent access to the same
`Sequence` found in this trace), and the *replay* itself (`apply_pending_ff_tokens`, already fully
traced by `06-c` Sec 3) only appends to `seq.tokens`/bookkeeping -- it does not touch the matcher or
request a second forward pass. This was previously stated as "inherited, not re-verified" by `06-c`;
this session re-derived it directly from the current source and found it consistent.

## 10. Confidence level

- **High confidence:** grammar-constrained sequences never enter either CUDA-graph fast/lookahead
  decode path (`sampling.rs:1032-1039`, `1173-1186`, `1195-1206`; `pipeline/mod.rs:2374-2383`;
  `pipeline/execution.rs:352`). This is a definitive source-level fact, not an inference, and it rules
  out `06-c`'s leading untraced candidate (`account_cuda_decode_rows`'s hard-coded `+1`).
- **Medium-high confidence:** the widened-window construction, PagedAttention context-length/slot
  assignment, logits-row selection, and position-ID assignment are internally consistent and correct
  for the established single-splice, width-1 case, based on reading every function in the chain from
  `inputs_processor.rs` through the CUDA kernel wrapper's declared contract. This is "no divergence
  found by careful reading," not "proven correct by execution" -- the distinction the task's Success
  Criterion B explicitly asks to preserve.
- **Low confidence / explicitly unresolved:** whether the raw CUDA kernel body correctly implements
  the multi-row-per-logical-sequence causal-attention semantics its Rust wrapper's contract implies,
  and whether runtime execution actually matches this session's static reading for the exact
  `start_pos`/block/slot values this repro produces. Both require either live instrumentation or
  reading kernel source not covered this session.
- The `\r` degeneration itself remains **unexplained**. Per the task's explicit instruction, this
  report does not manufacture an explanation for it from the (currently non-divergent) accounting
  layer -- if the accounting layer is genuinely equivalent, the explanation for the observed symptom
  must live either in the CUDA kernel internals (Sec 8, item 1) or somewhere not yet traced by any
  of `06-c`/`06-d`/this report.

## 11. Smallest safe next step

Two independent options, neither performed this session, in order of how directly they'd resolve the
remaining uncertainty:

1. **Live instrumentation** (the path this session's `AskUserQuestion` offered and the user declined
   for now): the smallest useful version, per this session's trace, would log exactly three scalars at
   the point `sample_sequence` receives its `logits` argument for the first real sample after a
   splice -- `seq.num_computed_tokens()`, the RoPE position the model actually used for that row (if
   accessible without deep kernel changes), and the sampled token id -- compared against a control
   step where an equivalent forced token arrives via FF-OFF's ordinary masked path. This would directly
   test Sec 7's "should be equivalent" claim against real execution, closing the Sec 8/item 2 gap.
2. **Read the CUDA kernel source directly** underneath `mistralrs-paged-attn/src/cuda/backend/paged_attention.rs`'s
   `paged_attention()` call (likely a `.cu`/PTX file bundled in or built by that crate, not yet located
   or opened this session) to check whether its per-row attention computation genuinely treats
   `context_lens[i]` as authoritative per-row history (as the Rust wrapper's contract implies) or
   contains any implicit single-token-decode assumption not visible from the Rust side. This is still
   static reading (no build/run needed) and was not done this session purely due to time/scope, not
   because it was judged unnecessary -- it is the most direct way to close Sec 8/item 1 without a live
   run.

Neither should be attempted speculatively; per the task's discipline section, no source change or new
sweep should follow from this report without deciding, with the user, which of these two (or a live
run) is worth the cost next.

## What evidence would be required before implementing a fix

Unchanged in spirit from `06-d`'s Sec 10: either (1) a live-instrumentation observation (option 1
above) showing the first real post-splice sample actually reads from a KV/logits row inconsistent with
Sec 7's traced expectation, with the exact source statement responsible identified, or (2) a
kernel-level read (option 2 above) surfacing a genuine multi-row-per-sequence assumption violation.
Absent either, the evidence gathered so far (this report plus `06-c`/`06-d`) does not support any
specific code change -- it has ruled out several plausible loci (CUDA-graph decode-tail advance,
llguidance token-content divergence, window/position/logits-row construction) without yet identifying
where the actual defect lives.

## Files added

- `plans/ff-round-two/reports/06-e-kv-row-root-cause.md` (this report). No other files changed. No
  Rust source touched at any point this session.

## Git/worktree state

- `/workspace`: verified at session start to be a genuinely separate worktree from
  `/tmp/ff-artifacts-wt` (distinct device/inode, `git worktree list` shows both), branch
  `grammar-fast-forward`, HEAD `cab8cd3ae`, clean throughout -- confirmed again at the end of this
  session (`git status --porcelain=v1` empty).
- The previous artifact worktree at `/tmp/ff-artifacts-wt` had been pruned (per the user's note at
  session start); its administrative metadata under `/workspace/.git/worktrees/` was gone, leaving an
  orphaned, non-git plain directory at that path. All prior work was safe on the `ff-demo-artifacts`
  branch itself (`b63fecfe6` was already the branch tip). The stale directory was removed and a fresh
  `git worktree add /tmp/ff-artifacts-wt ff-demo-artifacts` was run and independently verified
  (distinct `stat` device/inode from `/workspace`, distinct branch, correct HEAD).
- This report was written and committed from that freshly recreated worktree, branch
  `ff-demo-artifacts`.
- No server requests were made this session beyond read-only `/health`, `/metrics`, `/v1/models`
  probes to confirm the live server's reachability (no completion/generation requests were sent).

Not proceeding to Plan 07.
