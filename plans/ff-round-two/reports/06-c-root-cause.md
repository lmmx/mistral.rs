# 06-C -- source-level root-cause investigation of the B/N=1 nested-schema FF failure (2026-09-10, seventh session)

Status: **narrowed to a specific code region and a specific structural trigger; the exact defective
value/state could not be pinned to a single line without either external-crate source access or an
instrumented rebuild, both out of scope for this investigation. Read section 7 before treating
anything here as final.**

No Rust source modified. No rebuild. No new server requests were made in this session -- this is a
pure source-reading trace against `/workspace` (`grammar-fast-forward` @ `cab8cd3ae`, untouched) plus
re-inspection of the already-committed evidence in `06-b-n1-repro.md`, its raw JSON, and this
project's own prior source-review journal entries (`ff-demo-artifacts:docs/journal/`).

## 1. Executive conclusion

The failure is real, reproducible, and localized to **the single decode step where a fast-forward
splice is staged and replayed immediately before the grammar must choose to open a nested JSON
object** (`"run": {`). Of the four fixture schemas, `nested.schema.json` is the *only* one with an
object-valued property, and it is the only fixture implicated in any anomaly across all 34
requests run so far (24 in the Plan 06 sweep, 10 in the focused repro). All five defects previously
catalogued against this mechanism (`docs/journal/2026-09-09-fast-forward-splice-review.md`) were
resolved by commit `49848e501` (confirmed ancestor of `cab8cd3ae`), and none of them match this
failure's signature: this run staged and fed its one splice cleanly (`splice_drops_total` and
`tokens_dropped_total` deltas were both zero across every ON repeat), so the previously-fixed
*discard-without-rollback* defects (1, 2, 4) are structurally not in play here -- nothing was
discarded. This appears to be a **new, previously-untested failure mode**: no artifact on this
branch's history (the CPU `ff_bench.py` benchmark, the splice-review's code-reading audit, or this
plan's own 24-run sweep before this follow-up) exercised a splice immediately preceding a
grammar-forced entry into a deeper JSON nesting level, on a live model, with free choice resuming
inside that nested level.

I traced the mistralrs-side bookkeeping for the successful (non-discarded) splice path -- window
widening (`completion_token_cost`), KV-cache-length accounting (`num_computed_tokens` advancement),
and replay (`apply_pending_ff_tokens`/`finish_or_add_toks_to_seq`) -- and found each individually
consistent with the invariants the splice-review already validated for this path. I could not,
within this investigation's constraints, rule in or rule out the one remaining candidate: whether
`llguidance::Matcher::consume_ff_tokens`'s own splice computation is correct at a nesting-boundary
transition, because the `llguidance` crate (workspace-pinned to `1.2.0`, referenced as `1.4.0` in
last year's review) is not vendored in this checkout and this container has no network access to
fetch its source.

## 2. Exact failing prefix / splice involved

- Fixture: `plans/ff-round-two/fixtures/nested.schema.json`. First required property `run` is
  **object-valued** (`{"release_channel_name": enum[3], "observed_error_count": integer}`), second
  required property `note` is a free string. This is the only one of the four fixtures
  (`narrow`/`partial`/`wide`/`nested`) with a nested object -- the other three are flat
  (`property: {type: string|integer}` only, confirmed by reading all four files).
- Prompt: `"Report the status of the last deployment. Respond with only the requested JSON object."`,
  `temperature=0.0`, `seed=42`, `max_tokens=64`, N=1 (no concurrent peer, no batch-shape interaction
  possible).
- FF OFF output (5/5 identical): `{"run": {"release_channel_name": "stable", "observed_error_count":
  0}, "note": "The last deployment was successful."}`, `finish_reason=stop`, 47 completion tokens.
- FF ON output (5/5 identical): `{\n  "run":  \r  \r  \r  \r  \r  \r  \r  \r  \r  \r  \r  \r  \r  \r
  \r  \r  \r  \r  \r ` (raw, includes the `\n` and double space actually emitted), `finish_reason=
  length`, 64 completion tokens (budget exhausted, never recovers).
- FF ON metric deltas, every one of the 5 repeats: `mistralrs_grammar_ff_attempts_total{outcome=
  "staged"}` = 1, `{outcome="empty_splice"}` = 62, `splices_staged_total` = 1, `tokens_fed_total` =
  1, `splice_drops_total` and `tokens_dropped_total` = 0 (absent from the delta, i.e. exactly zero).
  One splice, width 1 token, staged and fed exactly once, at some point between the second output
  token (`"run":`, itself produced by ordinary per-token constrained sampling, since 62 of the 63
  useful attempts are `empty_splice`) and the point where generation degenerates. Every other decode
  step in the ON generation went through the *ordinary* per-token grammar-masked path
  (`empty_splice`, meaning llguidance found no additional forceable continuation beyond the token
  just sampled -- not a defect, the expected outcome at a genuine choice point per
  `06-ff-observability-audit.md` Sec 2.2).

The corruption is not merely "different output" -- the literal opening brace of the nested object,
`{`, **never appears** in the ON output at all. Where OFF produces `"run": {` (colon, one space,
brace), ON produces `"run":` followed by two spaces and then an unbroken run of `\r`. The nested
object is never opened; generation loops in what looks like a whitespace-accepting grammar state
that never advances to requiring `{`.

## 3. Source-level execution trace

All line numbers are against `/workspace` (`grammar-fast-forward` @ `cab8cd3ae`), read directly,
not from memory of the prior review (whose line numbers were against an earlier commit and have
since shifted).

**Splice construction and staging** -- `mistralrs-core/src/pipeline/sampling.rs`, inside the
function containing the per-token constrained-sampling block (the `SequenceRecognizer::Llguidance`
arm, `sampling.rs:1623-1668`):
- `sampling.rs:1629`: `llg.consume_token(second_logprobs_response.token)` commits the token the
  model/grammar-mask actually just chose (the ordinary, per-step advance).
- `sampling.rs:1633-1635`: if the grammar is still active and FF is supported, `llg.consume_ff_tokens
  ()` is called. Per this project's own prior source review
  (`docs/journal/2026-09-09-fast-forward-splice-review.md:86`, citing `llguidance-1.4.0
  matcher.rs:146`), this call both **computes and consumes** the forced continuation -- the matcher's
  internal parser state is already advanced past the returned tokens by the time this call returns.
  I could not independently re-verify this against source in this session (see Sec 7), so this fact
  is inherited from the prior review, not re-derived here.
- `sampling.rs:1636-1638`: `matcher_error = llg.is_error()` is checked before committing.
- `sampling.rs:1642-1645`: only on `FfAttempt::Staged` is `seq.set_pending_ff_tokens(splice)` called
  (`sequence.rs:1250-1252`, a plain field write, no side effects).

**Window widening / KV accounting for the next step** (not fully re-traced line-by-line this
session; inherited from the prior review, spot-checked below): `PagedAttentionScheduler::
completion_token_cost` (`paged_attention/scheduler.rs:540-545`) adds `active_pending_ff_tokens().
len()` to the token budget for the sequence's next scheduled step, and the general (non-CUDA-tail)
decode-completion path advances `num_computed_tokens` by the full `scheduled` width, guarded against
double-advance (`engine/mod.rs:2000-2008`, `2109-2117`: `if seq.num_computed_tokens() == before {
seq.advance_num_computed_tokens(scheduled) }`). This matches invariant 1 from the prior review's
"Remediation constraints" section ("the number of tokens appended to `Sequence` during the step
equals the number of positions the forward pass computed KV for during the step") for the
successful path, and I found nothing in this general path that special-cases nesting depth, JSON
structure, or anything grammar-shape-dependent -- it operates purely on token counts.

**Splice replay** -- `sampling.rs:834-866` (`sample_and_add_toks_inner`):
- `sampling.rs:857`: `let ff_toks = seq.take_pending_ff_tokens();` -- takes the width-1 splice.
- `sampling.rs:862`: `apply_pending_ff_tokens(this, prefix_cacher, seq, ff_toks, eos_tok).await?`
- `apply_pending_ff_tokens` (`sampling.rs:731-769`): for the one splice token, decodes it via
  `tok_env.tok_trie().decode_ext(&[token], include_special)` (`sampling.rs:752`), builds a
  synthetic `Logprobs` with `logprob: 0.0` (probability 1, forced), and calls
  `finish_or_add_toks_to_seq(this, prefix_cacher, seq, logprobs, eos_tok, true)` (`sampling.rs:766`).
- `finish_or_add_toks_to_seq` (`sampling.rs:188-...`): calls `seq.add_token(...)` (pushes the token
  into `seq.tokens`, extending the logical prefix `seq.get_toks()`), checks stop strings and tool-call
  state, and -- confirmed by reading the whole function body -- **never touches `seq.recognizer` /
  the llguidance matcher** except in one specific branch: "Mid-stream grammar activation for tool
  calls" (`sampling.rs:240-248`), gated on `matches!(seq.recognizer, SequenceRecognizer::None)`. For
  our sequence, `seq.recognizer` is `SequenceRecognizer::Llguidance(_)`, not `None`, so this branch
  does not fire, and the matcher is not touched at all during replay. This is expected and, per the
  prior review's citation of the external crate's contract (`consume_ff_tokens` already consumes),
  *should* be correct: the matcher was already advanced when the splice was staged, so replay is
  purely `Sequence`-side bookkeeping (token list, stop strings, prefix cache) and is not supposed to
  touch the matcher again.

**Resumed real sampling**: after the replay loop, `sample_and_add_toks_inner` proceeds to the
remaining running sequences and the *next* decode step's forward pass produces the logits from which
the grammar mask is computed at `sampling.rs:1561-1571` (`llg.compute_mask_or_eos()`,
`mask.is_allowed(...)`, building the `-inf` bias vector `acc`), which is added to the raw logits
(`sampling.rs:1588`) before the real token is sampled. This is the point where, if the matcher's
internal position is not actually where the token stream is, the computed mask would be wrong for
the *actual* generated prefix -- but whether this happened here is exactly what I could not confirm
(see Sec 5-7).

## 4. FF path vs non-FF path comparison

| step | FF OFF | FF ON |
|---|---|---|
| `"run":` (key + colon) | ordinary per-token sampling, grammar-masked each step | same -- 62 `empty_splice` attempts recorded, i.e. ordinary per-token path for everything up to and around this point |
| entry into the nested object (`{`) | model emits `{` directly as the next masked token | **one** `consume_ff_tokens()` call returns a non-empty (width-1) splice here; it is staged, then replayed via `apply_pending_ff_tokens`/`finish_or_add_toks_to_seq` (`seq.tokens` extended by 1, matcher not touched during replay since it was already advanced at stage time) |
| immediately after | model continues into `"release_channel_name":` etc., real per-token masked sampling | model never produces `{`; output shows two spaces then an unbroken run of `\r`, `finish_reason=length` |

The two paths are identical in every input (prompt, schema, seed, sampler config) up to this one
point. The only mechanical difference between them at this point is: OFF never calls
`consume_ff_tokens`/stages/replays anything (the flag gates the whole block off, `sampling.rs:1634`
`if supports_fast_forward && grammar_active`); ON does, exactly once, right at the object-nesting
boundary.

## 5. The first concrete divergence in state or invariant

I can state the *location* of the first concrete divergence precisely (the one `consume_ff_tokens`
call at `sampling.rs:1635`, and its replay at `sampling.rs:862`, both occurring at the transition
into the nested `run` object) but **not** the exact value that is wrong, for a specific reason: the
divergence is not visible in mistralrs's own bookkeeping (token counts, KV-length accounting, and
`Sequence` state all appear internally consistent for this path, per Sec 3), which leaves two
candidate loci for the actual defect, neither of which I could fully verify in this session:

1. **The splice's token content is itself wrong or incomplete for a nesting transition.** If
   `Matcher::consume_ff_tokens()` returns a token that decodes to something other than (or less than)
   what the nested object genuinely requires -- e.g., a whitespace token that the grammar's
   canonicalized JSON-formatting rules treat as forced but that does not actually correspond to a
   safe, resumable parser position for a *nested* object the way it would at the top level -- then
   mistralrs would be faithfully replaying a bad instruction from the external crate. This is
   consistent with the observed output (no `{` ever appears; the two literal spaces after `"run":`
   in the ON output, versus one in OFF, are circumstantial but suggestive of an extra forced
   whitespace token).
2. **The matcher's parser-stack position after the splice does not actually match the stated
   contract.** If, specifically for a transition into a deeper nesting level, `consume_ff_tokens()`
   does *not* fully advance the matcher's internal state the way the top-level case does (a
   plausible class of bug in any recursive-descent-style grammar matcher: forgetting to push a stack
   frame for the new nesting level while still reporting the byte/token as consumed), the mask
   computed at `sampling.rs:1561` on the next real step would be evaluated against the *pre-nesting*
   grammar position -- which, for a JSON-schema-derived grammar, very plausibly still permits
   whitespace-class bytes (many JSON-schema-to-grammar compilations allow `\s*` between structural
   tokens), which would explain both "no `{` ever required" and "why `\r`, a whitespace byte, loops
   indefinitely instead of erroring or producing unrelated garbage."

Both loci point at the interaction between mistralrs's single call to `consume_ff_tokens()` and
whatever that call does internally for a **nested** grammar transition specifically -- not at
anything in mistralrs's own scheduler, KV-accounting, or replay bookkeeping, which I read in full
for the successful path and found to already satisfy the invariants the prior defect review
established as necessary (Sec 194-213 of `06-b-n1-repro.md`'s companion journal entry). I was not
able to determine which of the two (or something else inside the external crate not covered by
either) is the actual defect without reading `llguidance`'s own source for `consume_ff_tokens` and
`compute_mask_or_eos` at a nesting boundary, which this environment cannot do (no vendored copy, no
network).

## 6. Why this explains the repeated `\r` output

If either candidate in Sec 5 holds, the model's *own* forward pass sees a token stream that
correctly includes the spliced token (KV cache and position accounting are confirmed consistent per
Sec 3), so the model itself is not corrupted -- but the **grammar mask** applied on top of its logits
at every subsequent step would be wrong (too permissive), because it is derived from a matcher state
that either received the wrong forced token or did not fully register the nesting transition. A
too-permissive mask that still happens to allow whitespace-class bytes at every position (plausible
for a JSON grammar, which commonly allows optional whitespace almost everywhere) would let the
model's own preference -- which, if the real context looks unusual/confusing to it because a
structurally-odd token was forced in, may itself have an elevated preference for whitespace-like
tokens -- repeatedly win the argmax at temperature 0, forever, since neither the mask nor the model's
own state ever forces or biases toward the actual next required structural token (`{`). This is
consistent with, but not proof of, the observed infinite `\r` loop; I have not independently
verified that a JSON-schema-derived llguidance grammar's whitespace-class token set would actually
tokenize/decode to `\r` specifically in this model's vocabulary, which would be one further step to
firmly closing this explanation.

## 7. Confidence level and remaining uncertainty

- **High confidence:** the failure is real, reproducible, and specific to this fixture's nested
  structure -- established empirically in `06-b-n1-repro.md` (5/5 reproducibility both directions)
  and corroborated here by the fixture-structure comparison (nested vs. the three flat fixtures) and
  by the metric evidence (exactly one splice, cleanly staged and fed, no discard).
- **High confidence:** this is not any of the five previously-catalogued and -fixed defects
  (`docs/journal/2026-09-09-fast-forward-splice-review.md`) -- none of those concern the successful,
  non-discarded path, and this run discarded nothing.
- **Medium confidence:** the divergence originates at or immediately after the single
  `consume_ff_tokens()` call that stages the splice crossing into the `run` object's nesting level
  (`sampling.rs:1635`), given it is the only point where FF ON and FF OFF's execution paths actually
  differ before the observed corruption appears.
- **Low confidence / unresolved:** which of the two candidate loci in Sec 5 (bad splice content vs.
  incomplete matcher-state advancement at a nesting boundary) is the actual defect, and whether the
  defect is in mistralrs's usage of the `llguidance` API or inside `llguidance` itself. This is the
  narrowest remaining unknown. Resolving it needs one of: (a) `llguidance` crate source (network
  access to fetch v1.2.0, or a vendored copy) to read `Matcher::consume_ff_tokens` and
  `compute_mask_or_eos` directly; (b) an instrumented debug build that logs the splice's decoded
  token(s) and the matcher's `is_stopped`/mask state immediately before and after the splice, which
  this investigation was explicitly told not to build; or (c) a minimal, source-preserving live
  experiment that varies only the fixture (e.g. a synthetic schema with a nested object as the
  *second* rather than *first* required property, or with a non-enum flat sibling before the nested
  one) to see whether the trigger is "nesting" specifically or something more specific to being the
  *first* forced structural element of the generation -- not run in this session, since it would be
  a new experiment beyond the single flagged cell this investigation was scoped to.
- I did **not** find any CR/LF-specific special-casing anywhere in the traced mistralrs code
  (`finish_or_add_toks_to_seq`, `apply_pending_ff_tokens`, the splice-staging block) -- the `\r`
  appears to be an ordinary vocabulary token decoded and emitted like any other, not a sentinel or
  fallback value mistralrs inserts itself. I could not rule out CR/LF-specific behavior inside
  `llguidance`'s own whitespace handling, for the reason given above.
- I did **not** find evidence implicating CUDA or PagedAttention specifically (per the task's
  instruction not to assume CUDA without evidence): the token-count/KV-accounting invariants I could
  check are backend-agnostic in the code (`num_computed_tokens`, `completion_token_cost`), and the
  prior review separately verified that CUDA-graph decode replay excludes any window wider than one
  token (`pipeline/cuda_graph.rs:600-602` per that review, not re-read this session), so a
  fast-forward step already falls back to the eager path regardless of backend. Nothing in this
  investigation points at CUDA/paged-attention as the locus; it remains untested rather than ruled
  out at the "is the CUDA decode-tail lookahead path grammar-aware" question I raised and did not
  fully resolve while reading `engine/mod.rs`'s `account_cuda_decode_rows`/`continue_cuda_decode_batch`
  functions -- flagged here as a loose thread, not a finding: I could not confirm whether that path
  is unconditionally excluded for any grammar-constrained sequence, only that Plan 06's own dimension-1/2
  data (splice discard/loss rates) and this reproduction's metric deltas are both consistent with the
  ordinary (non-CUDA-tail) constrained-decode path having run throughout.

## 8. What a future fix would need to preserve

Not a proposed patch -- a list of invariants a fix must not break, derived from Sec 3-6 and from the
prior review's own "Remediation constraints" (which a nesting-boundary fix must not regress):

- The five already-fixed defects (discard-without-rollback, PagedAttention slot accounting for the
  splice width, penalty-context ordering, matcher-error discard, and top_logprobs shape) must stay
  fixed; nothing here suggests touching any of that code.
- Whatever the eventual fix, it must not disable fast-forward for flat (non-nested) grammars --
  every flat-schema run in this investigation (Plan 06's workload A across N=1..8, and `narrow`/
  `wide`/`partial` throughout workload B/C) produced no observed anomaly, so the defect is specific
  to some structural trigger, not to fast-forward as a mechanism.
- Any fix touching the `consume_ff_tokens()` call site or its replay must preserve the property that
  a splice, once staged, corresponds exactly to tokens the model's forward pass actually computed KV
  for in that window (invariant 1 from the prior review) -- this investigation found no evidence that
  invariant is currently broken for this failure, so a fix should not need to touch the
  window-widening/KV-accounting machinery at all, only whatever governs what `consume_ff_tokens()`
  is allowed to force and/or how its result is validated before being staged.
- A fix should be verifiable with exactly this reproduction (N=1, `nested.schema.json`, seed 42,
  5-repeat determinism check) before being declared correct, and should additionally be checked
  against a schema where the nested object is *not* the first required property, since this
  investigation did not vary that and cannot say whether "first forced element of the whole
  generation" vs. "nesting" is the operative trigger.

## Files added

- `plans/ff-round-two/reports/06-c-root-cause.md` (this report). No other files changed.

## Git/worktree state

- `/workspace`: branch `grammar-fast-forward`, HEAD `cab8cd3ae`, clean, untouched. No source read
  produced any edit; this was a read-only investigation.
- This report was written and committed from the separate worktree at `/tmp/ff-artifacts-wt`,
  branch `ff-demo-artifacts`.
- No server requests were made this session; no host server state was touched.

Not proceeding to Plan 07.
