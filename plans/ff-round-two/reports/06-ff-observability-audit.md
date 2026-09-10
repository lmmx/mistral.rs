# 06 — FF observability audit: what the code proves, and the instrumentation added (2026-09-10)

Class: **audit + fix**. This session read the control flow instead of running another experiment.
Plan 06's workload sweep was **not** started; no schema/model experiment was run; no scheduling or
fast-forward behaviour was changed.

Source branch tip audited: `4ba40ca9e` (`fix: disable grammar fast forward for AnyMoE`), read in a
second worktree. Two commits were landed on `grammar-fast-forward`; see "Files changed" below.

## 1. Current FF control-flow map

Everything below was read at `4ba40ca9e` and is asserted from source, not inferred from a run.

```
load time
  perf_flags::grammar_fast_forward_enabled()      OnceLock over MISTRALRS_GRAMMAR_FAST_FORWARD
     |                                            default false; accepts 1/true/TRUE/yes/on
     v
  GeneralMetadata.supports_grammar_fast_forward
     gguf.rs:1415  ggml.rs:400  normal.rs:289   = flag && !no_kv_cache && !is_xlora
     amoe.rs:61                                 = false (overrides the wrapped pipeline)
     multimodal.rs:1531 embedding.rs:706 speech.rs:330 diffusion.rs:252 = false
     |
     v
request time
  add_request.rs:606  build_sequence_recognizer(llg_factory, constraint)
     -> SequenceRecognizer::Llguidance(Box<llguidance::Matcher>)  for a constrained request
     -> SequenceRecognizer::None                                  otherwise
     |
     v
decode step: sample_and_add_toks_inner (sampling.rs:834)
  a. replay any splice staged last step: take_pending_ff_tokens -> apply_pending_ff_tokens
  b. try_sample_batch_cuda  -- can_sample_batch_cuda (sampling.rs:947) returns false for ANY
     sequence with a recognizer, so a constrained request never takes the batched CUDA path
     and always reaches sample_sequence.
  c. sample_sequence (sampling.rs:1490), tail block:
        match seq.recognizer
          None => nothing
          Llguidance(llg) =>
            ends_turn = sampled token is EOS && llg.is_accepting()
            if llg.is_stopped() || ends_turn        -> no FF possible this step
            else
              llg.consume_token(tok)?              -- `?`: a matcher error here fails the sequence
              if !supports_fast_forward             -> no FF possible, ever, for this pipeline
              else if llg.is_stopped()              -> the sampled token completed the grammar
              else
                splice = llg.consume_ff_tokens()
                if !splice.is_empty() && !llg.is_error() -> seq.set_pending_ff_tokens(splice)
                                                           + splices_staged_total
                else if llg.is_error()                   -> warn, nothing staged
                else                                     -> nothing staged, silently
     |
     v
next decode step: engine/mod.rs:1819
  resolve_pending_ff_batch (staging.rs:58) — non-prompt steps only. Unless every splice-carrying
  row in the batch agrees on width AND no splice-free row is mixed in with them
  (staged_batch_state_from_widths, staging.rs:54), EVERY splice in the batch is discarded and its
  matcher rolled back: splice_drops_total{reason="batch_shape"}.
  Surviving splices widen scheduled_token_counts -> tokens_fed_total (engine/mod.rs:1851).
  Other discard reasons already instrumented: "sequence_end" (default_scheduler.rs:240,340;
  paged scheduler:1246,1454; utils/mod.rs:101,255; sampling.rs:486,499,511),
  "preemption" (paged scheduler:1390), "realloc" (sequence.rs:1376).
```

Two facts worth pinning because prior sessions speculated about them:

- **The CUDA batched-sampling fast path is not a candidate explanation.** `can_sample_batch_cuda`
  (`sampling.rs:947-953`) returns `false` for any sequence whose `recognizer` is not
  `SequenceRecognizer::None`. A grammar-constrained request always reaches `sample_sequence` and
  therefore always reaches the `consume_ff_tokens` call site, given the gate.
- **The speculative-decoding pipeline deliberately never fast-forwards.** `driver.rs:288` and
  `verifier.rs:900`/`:965` pass `supports_fast_forward = false` literally.

## 2. Answers to the six questions

**1. What must hold for `splices_staged_total` to increment?** All of, per decode step:
`MISTRALRS_GRAMMAR_FAST_FORWARD` set truthy at first read; the pipeline is `normal`/`gguf`/`ggml`
(not AnyMoE, multimodal, embedding, speech, diffusion); `!no_kv_cache`; `!is_xlora`; the sampling
path is `sample_and_add_toks*`, not the speculative driver/verifier; the request carries a
constraint so `recognizer` is `Llguidance`; the matcher was not already stopped and the sampled
token did not end the turn; `consume_token` succeeded; the matcher is still not stopped after it;
`consume_ff_tokens()` returned a non-empty vector; and `is_error()` is false.

**2. What can make `consume_ff_tokens()` return empty?** Established previously and not re-derived
here (no llguidance source or cargo registry in the audited tree — the lockfile pins 1.4.0): there
are no additional forceable bytes at this grammar position, or grammar forcing is disabled for the
lexer spec, or the forced bytes do not tokenize canonically under the `TokEnv`. **Empty is a
legitimate, expected outcome and is not evidence of a defect.** For GGUF the env is the
HF-tokenizer-backed `ByteTokenizerEnv` built in `llg.rs:50-56`.

**3. What can prevent the sampling block being entered at all?** `recognizer == None`
(unconstrained request, or a tool-call grammar deactivated at `sampling.rs:1651-1659`); matcher
already stopped or turn ended; `consume_token` returning `Err` (propagates via `?`, failing the
sequence before FF is considered); the sequence finishing during the pre-sampling splice replay
(`sampling.rs:857-866`, which filters it out of `running_seqs` precisely so it cannot stage again);
and the speculative pipeline, which passes the flag as `false`.

**4. Could the metrics distinguish the six states? Before this session: no — four of six were
indistinguishable.**

| State | Before | After |
|---|---|---|
| FF disabled (flag/`no_kv_cache`/xlora/pipeline) | invisible | `ff_support_total{supported="false",reason=…}` at load; `ff_attempts_total{outcome="unsupported"}` per step |
| FF enabled, no forced bytes | invisible | `ff_attempts_total{outcome="empty_splice"}` |
| FF enabled, non-canonical tokenization | invisible, and **not separable** from the row above | still folded into `empty_splice` — llguidance returns the same empty vector; separating them needs a change inside llguidance, not here |
| matcher error | `warn` log only | `ff_attempts_total{outcome="matcher_error"}` + the existing warn |
| splice staged | `splices_staged_total` (+ `debug` log) | unchanged, plus `outcome="staged"` |
| staged and fed | `tokens_fed_total` | unchanged |

The one row that remains unseparated is called out honestly: "grammar forced nothing" and
"tokenization could not produce canonical FF tokens" are the same return value at our call site.
Distinguishing them requires llguidance to say which, and is not worth a fork.

**5. Is there an existing hook that answers these?** No.
`tracing::debug!("fast-forward splice computed")` (`sampling.rs:1636`) fires only on success;
`tracing::warn!` fires only on matcher error; the four `mistralrs_grammar_ff_*` counters all fire
only on the success or the discard path. Nothing recorded the resolved
`supports_grammar_fast_forward` anywhere. And because a `metrics`-crate counter is only registered
when first touched, an unreached call site and a never-incremented one render identically — as
nothing at all. `mistralrs-server-core/src/metrics.rs:150-176` installs one `PrometheusBuilder`
recorder with bucket configuration only and no allowlist, re-confirmed this session, so no filtering
explanation exists.

One further fact, not previously recorded, that matters for plan 05's harness: **only
`mistralrs-server-core` ever installs a metrics recorder.** `mistralrs-pyo3` does not
(`lib.rs:3223` calls `initialize_logging()` and nothing else). Every `mistralrs_grammar_ff_*`
counter is therefore a no-op in the in-process `equality`/`routing-log`/`arms` harness modes, by
construction — those modes can only ever see `tracing` output. That is why the load-time record
added below emits an `info` log line as well as a counter.

**6. Smallest change that makes the state machine observable.** Implemented, two commits:

- `mistralrs_grammar_ff_support_total{supported,reason}` — one increment per pipeline load, with
  `reason` in `{enabled, flag_disabled, no_kv_cache, xlora, pipeline_unsupported}`, plus an `info`
  line carrying the same two fields. The gate's three conjuncts moved into one
  `pipeline::ff_metrics::resolve_support(no_kv_cache, is_xlora)` that the three text pipelines call;
  AnyMoE records `pipeline_unsupported` because it overrides a pipeline that already recorded its
  own answer.
- `mistralrs_grammar_ff_attempts_total{outcome}` — one increment per **grammar-constrained** decode
  step, `outcome` in `{unsupported, grammar_stopped, empty_splice, matcher_error, staged}`.
  Unconstrained sequences are not counted, so the counter's total is exactly the number of
  constrained decode steps: the denominator that has been missing.
- The load-time record also pre-registers all five `outcome` series with `.increment(0)`, so a
  process where the staging path is never reached exports explicit zeros instead of nothing.
  Verified against the pinned `metrics-exporter-prometheus` 0.18.3 in a standalone probe:

  ```
  # TYPE mistralrs_grammar_ff_support_total counter
  mistralrs_grammar_ff_support_total{supported="true",reason="enabled"} 1

  # TYPE mistralrs_grammar_ff_attempts_total counter
  mistralrs_grammar_ff_attempts_total{outcome="grammar_stopped"} 0
  mistralrs_grammar_ff_attempts_total{outcome="empty_splice"} 3
  mistralrs_grammar_ff_attempts_total{outcome="staged"} 0
  mistralrs_grammar_ff_attempts_total{outcome="unsupported"} 0
  mistralrs_grammar_ff_attempts_total{outcome="matcher_error"} 0
  ```

Rejected as larger than necessary: a per-step log line (that is per-token logging), a histogram of
splice widths (dimension 4's job, and plan 06 already reserves that change), a debug framework, and
any redesign of the four existing counters, which are kept exactly as they are.

### Why this taxonomy and not the one the brief suggested

The brief floated `disabled / not_eligible / empty_splice / matcher_error / staged`. Derived from
the control flow, `not_eligible` is not one state: the two ways to be ineligible have completely
different fixes. `unsupported` means the process is misconfigured and no request will ever
fast-forward; `grammar_stopped` means the pipeline is healthy and this particular step had a
finished grammar. Folding them would reproduce, one level up, exactly the ambiguity this change
exists to remove. `disabled` is dropped as a per-step outcome because it is a load-time property,
better answered once by `ff_support_total` than repeated per token; `unsupported` covers its runtime
shadow.

`classify_attempt` checks `supports_fast_forward` **first**, before grammar state, so a run with the
capability off reports `unsupported` on every constrained step rather than a mixture that depends on
what the grammars happened to be doing.

## 3. Why it is behaviour-preserving

- `resolve_support` returns `flag && !no_kv_cache && !is_xlora`, the identical boolean the three
  pipelines computed inline; the diff at each site is the expression's spelling. AnyMoE still stores
  a literal `false`.
- At the sampling call site the staging condition is unchanged. It was
  `!splice.is_empty() && !llg.is_error()`; it is now `matches!(outcome, FfAttempt::Staged)` where
  `classify_attempt(true, true, matcher_error, splice.len())` yields `Staged` on exactly that
  conjunction. The warn branch is unchanged. `llg.is_error()` is hoisted into a local, which changes
  nothing: short-circuit evaluation already called it on every path through the old `if`/`else if`.
- Nothing new is written to `Sequence`, no matcher call is added or removed, no scheduling input
  changes, and the counter increments cannot fail or panic.
- Cost is one `metrics` counter increment per constrained decode step and one enum comparison; the
  load-time work is six increments and one log line per pipeline.

## 4. Verification performed

A Rust toolchain was installed in this container (none was present), and:

- `cargo check -p mistralrs-core --lib --all-targets` — clean, no warnings, at each of the two
  commits.
- `cargo test -p mistralrs-core --lib` — **1382 passed, 0 failed, 5 ignored** at the final commit,
  including the pre-existing FF tests in `speculative::staging`, `sequence` and the schedulers.
- 7 new tests in `pipeline::ff_metrics::tests`, all CPU-only, no model, no tokenizer, no GPU:
  `unsupported` dominates every runtime state; a stopped grammar is not an empty splice;
  `matcher_error` wins over splice length (both at length 0 and 7, matching the call site's
  precedence); an empty splice with a healthy matcher is its own outcome; a non-empty splice from a
  healthy matcher is `staged`; all five outcomes are reachable from `classify_attempt`; and both
  label sets are bounded, distinct, and have exactly one "supported" member.
- `rustfmt` clean on both changed files. Two pre-existing formatting deviations elsewhere in the
  tree (`amoe.rs:844`, `speculative/staging.rs:56`, both present at `4ba40ca9e` before this session)
  were left alone.

**Not verified, and explicitly unverified: everything requiring a GPU.** This container has no
`/dev/nvidia*` and no CUDA runtime. Nothing here demonstrates that fast-forward works, that a splice
is ever staged on a real model, or what the outcome distribution looks like in practice. The claim
made is narrower and is proved by the source: *whatever* happens at the call site, the next run will
say which of five things it was.

## 5. How the next CUDA smoke run should be read

Rebuild the server from the new tip with `MISTRALRS_GRAMMAR_FAST_FORWARD=1`, issue the same
constrained requests, and scrape `/metrics`. Read it in this order:

1. **`mistralrs_grammar_ff_support_total` absent entirely** → the build predates these commits, or
   the recorder was installed after the pipeline loaded. Note that `mistralrs serve` installs it
   first thing (`mistralrs-cli/src/commands/serve.rs:46-48`), but the embedded
   `mistralrs_server_router_builder.rs:284` path may not; check the startup `info` line
   ("grammar fast-forward support resolved for this pipeline") to be sure.
2. **`{supported="false"}`** → FF is off in this process, and `reason` says which conjunct did it.
   Fix that; nothing downstream is meaningful. `reason="flag_disabled"` with the env var set means
   the value was not one of the accepted spellings, or the `OnceLock` was read before the var was
   set.
3. **`{supported="true",reason="enabled"}` and all five `ff_attempts_total` series at 0** → no
   constrained request reached the sampling tail block at all. Suspect the request shape (the HTTP
   route needs `{"type":"json_schema","value":{…}}`, per `openai.rs`), not fast-forward.
4. **`outcome="unsupported"` non-zero while `supported="true"` was recorded** → more than one
   pipeline is loaded and the serving one is not the FF-capable one.
5. **`grammar_stopped` dominant** → generations are ending before the grammar does much; lengthen
   them or use a more forcing schema. This is `05-harness.md`'s discriminator 2, now measurable.
6. **`empty_splice` dominant, `staged` zero** → the path is live and llguidance is forcing nothing
   here. This is the case that has always been indistinguishable from "broken", and it is **not**
   evidence of a defect. It is the point at which discriminator 1 (a non-GGUF checkpoint via
   `Which.Plain`, testing tokenizer canonicality) becomes worth running, and the point at which a
   `warn`-level llguidance question, not a mistral.rs one, is the next step.
7. **`staged` non-zero** → the precondition is discharged. Plan 06's dimensions 1, 2 and 4 have a
   denominator and its sweep may proceed.
8. **`matcher_error` non-zero** → a real defect; the accompanying `warn` line carries the message.

For the in-process Python harness modes there are no counters at all (see §2.5). There, the only
signal is the startup `info` line, which now exists; read it with `MISTRALRS_DEBUG=1` or
`RUST_LOG=mistralrs=info`.

## 6. Plan 06 amendment

**Yes, plan 06 should be amended**, and with a stronger gate than the one proposed. A single
"prove the staging path is reached" gate conflates three failures with three different owners.
`06-batch-shape-measurement.md` has been amended with a three-part precondition:

- **A — capability.** `mistralrs_grammar_ff_support_total{supported="true",reason="enabled"} >= 1`
  in the serving process. Fails ⇒ configuration/build problem.
- **B — reachability and classification.** `mistralrs_grammar_ff_attempts_total` present with a
  non-zero total, and its per-outcome distribution recorded in the report whatever it says. Fails ⇒
  request-shape or harness problem, not an FF problem.
- **C — measurability.** `mistralrs_grammar_ff_splices_staged_total > 0`, unchanged from
  `05-harness.md`. Fails ⇒ dimensions 1, 2 and 4 are reported as *not yet measurable*, with B's
  distribution attached as the reason. Dimension 5 (tokens/s and forward passes, flag on vs off)
  stays measurable throughout and does not wait on any of these.

B is the part worth insisting on: it is the check that turns "we saw nothing" into a named state,
and it is the only one of the three that can be satisfied while still reporting a negative result
usefully.

## 7. What the next session should execute

1. Build the CUDA server from `grammar-fast-forward` tip `9c413eed4`.
2. Start it with `MISTRALRS_GRAMMAR_FAST_FORWARD=1` and `-v`; capture the startup log and grep for
   `grammar fast-forward support resolved`.
3. `curl` one constrained request, then scrape `/metrics` with
   `--metric-prefix mistralrs_grammar_ff_`, and walk §5's ladder. Record the outcome distribution
   verbatim in a report even if — especially if — `staged` is zero.
4. Only if gate C passes, start plan 06's sweep. Otherwise write up which rung of §5 the run landed
   on, and run discriminator 1 or 2 as that rung directs.

Do not, on the strength of a zero `staged`, conclude that fast-forward is broken. §2.2 still holds:
an empty splice is a legitimate llguidance answer.

## 8. Files changed

On `grammar-fast-forward`:

| File | Change |
|---|---|
| `mistralrs-core/src/pipeline/ff_metrics.rs` | new — the two label sets, `resolve_support`, `classify_attempt`, the two record functions, 7 tests |
| `mistralrs-core/src/pipeline/mod.rs` | `pub(crate) mod ff_metrics;` |
| `mistralrs-core/src/pipeline/{gguf,ggml,normal}.rs` | inline gate → `ff_metrics::resolve_support(...)` |
| `mistralrs-core/src/pipeline/amoe.rs` | record `pipeline_unsupported` on the metadata override |
| `mistralrs-core/src/pipeline/sampling.rs` | classify and record each constrained decode step |

On `ff-demo-artifacts`: this report, and the precondition section of
`plans/ff-round-two/06-batch-shape-measurement.md`.

`ff_harness.py` was not modified. The analysis did not show the harness to be the blocker: it
reached the server, sent well-formed constrained requests and scraped `/metrics` correctly. The
blocker was that there was nothing to scrape.

## 9. Commit hashes

| Branch | Commit | Subject |
|---|---|---|
| `grammar-fast-forward` | `6c1edc2bb` | feat(core): record resolved grammar fast-forward support at pipeline load |
| `grammar-fast-forward` | `9c413eed4` | feat(core): classify every constrained decode step's fast-forward outcome |
| `ff-demo-artifacts` | see this commit | docs: FF observability audit and plan 06 precondition |
