# 06-F -- runtime finding: the B/N=1 nested-schema FF divergence starts one step after the splice, not at it (2026-09-10, tenth session)

Status: **runtime finding / narrowed hypothesis, not a root-cause conclusion.** This report records
the first live-instrumented observation in this investigation (06-b through 06-e were reproduction,
metrics, and static source tracing only, per each report's own scope notes). It **rules out** two of
06-e's open uncertainties empirically and **narrows** the defect's location to one specific
transition, but does not identify the defective line or mechanism.

A minimal `tracing::debug!` patch (four hunks, three files, no new state, described in-session; not
committed to this branch or `grammar-fast-forward`) was applied locally to a `grammar-fast-forward`
@ `cab8cd3ae` checkout on the host, built, and run twice (FF-OFF then FF-ON, fresh server each time)
against the exact reproduction from `06-b-n1-repro.md`/`06-c-root-cause.md`. This report documents
only the resulting log evidence and what it establishes. **No source code is modified or committed by
this report; the instrumentation patch itself lives only in the host's local working tree on
`grammar-fast-forward` and is out of scope for this commit.**

## 1. Reproduction (unchanged from 06-b/06-c/06-e)

- `unsloth/Qwen3.5-4B-GGUF`, quant 4, paged attention on, CUDA, `--max-batch-size 8 --max-seqs 8`.
- Request: prompt `"Report the status of the last deployment. Respond with only the requested JSON
  object."`, `temperature=0.0`, `seed=42`, `max_tokens=64`, `enable_thinking=false`,
  `plans/ff-round-two/fixtures/nested.schema.json` as `grammar={"type":"json_schema","value":...}`.
  One small unconstrained warmup (`max_tokens=16`, no grammar) sent first on each leg, matching prior
  sessions' methodology.
- Two fresh server restarts, one FF-OFF (`MISTRALRS_GRAMMAR_FAST_FORWARD` unset) and one FF-ON
  (`MISTRALRS_GRAMMAR_FAST_FORWARD=1`), `RUST_LOG=warn,mistralrs_core::pipeline=debug` on both, `-v`.
- Outputs matched every prior session exactly: FF-OFF `finish_reason=stop`, 47 completion tokens,
  valid JSON. FF-ON `finish_reason=length`, 64 completion tokens, degenerates after `"run": ` into a
  repeated whitespace-class token, never reaching `{`.

## 2. Instrumentation (for context; not part of this commit)

Four `tracing::debug!` sites, gated so they fire only on the code paths relevant to this
investigation (an FF splice being staged, a decode window actually widened for one, and the
`LogitsSelection::Decode` branch that a widened window's logits-row selection resolves to), plus one
unconditional line logging every real sampled token (`logical_position`, `sampled_token`) at the end
of `sample_sequence`. No new per-sequence or per-batch state was added; this session's request is
N=1, so no sequence-identifier field was needed to disambiguate concurrent sequences (noted as a gap
for any future concurrent-N repetition of this method).

## 3. Token-by-token comparison around the failure

`logical_position` is `seq.get_toks().len()` at the moment each token is sampled (1-indexed from the
first generated token of this request; both legs' own second request -- the JSON-schema one -- reset
this numbering from prompt length, so the absolute numbers below are internally comparable between
the two legs but are each request's own running count, not tied to a global token index).

| logical_position | FF-OFF `sampled_token` | FF-ON `sampled_token` |
|---|---|---|
| 31 | 328 | 328 |
| 32 | 5917 | *(not sampled -- replayed from the staged splice)* |
| 33 | 763 | **763** |
| 34 | 313 | **220** |
| 35 | 198 | 220 |
| 36 | 262 | 201 |
| 37 | 328 | 220 |
| 38+ | (continues correctly to a valid, schema-conforming close) | locked into a repeating `220, 220, 201` cycle through `max_tokens` |

Exact FF-ON log sequence around the event (verbatim, timestamps as recorded):
```
... sample_sequence result logical_position=30 sampled_token=220
ff_trace: splice staged splice_tokens=[5917]
ff_trace: sample_sequence result logical_position=31 sampled_token=328
ff_trace: widened decode window built query_len=2 effective_context_len=33 window_tokens=[328, 5917] logits_span=Some((1, 1))
ff_trace: LogitsSelection::Decode selected start=1 len=1 seq_len=2
ff_trace: sample_sequence result logical_position=33 sampled_token=763
ff_trace: sample_sequence result logical_position=34 sampled_token=220
ff_trace: sample_sequence result logical_position=35 sampled_token=220
ff_trace: sample_sequence result logical_position=36 sampled_token=201
...(repeats 220, 220, 201 to logical_position=91, where max_tokens cuts generation off)
```

FF-OFF's log has no `splice staged` / `widened decode window built` / `Decode selected` lines at all
(as expected -- the flag is off) and instead shows the ordinary per-token path at every position,
including independently sampling `5917` at position 32 via its own masked forward pass.

## 4. What this establishes

1. **The splice's forced token content is confirmed correct at runtime, not just by 06-d's source
   argument.** FF-OFF independently samples `5917` at logical position 32 through its own masked
   forward pass; FF-ON's splice forces the identical `5917` without a forward pass. This is now an
   observed fact, not an inference from shared code paths.
2. **The widened-window step itself -- window construction, `query_len`, `LogitsSelection::Decode`'s
   row selection, and the real sample taken from it -- is confirmed correct at runtime.** The one real
   sample FF-ON takes immediately after replaying the splice (`logical_position=33`, from a
   `query_len=2` window `[328, 5917]` narrowed via `LogitsSelection::Decode{start:1,len:1}` to the
   second row) produces `763` -- **the exact same token** FF-OFF produces at the same logical position
   through its own, unwidened, ordinary forward pass. This directly confirms 06-e's static conclusion
   ("no divergence found in the traced Rust-level accounting for this step") with a live value, not
   just by reading code.
3. **The first divergence is one step later, at `logical_position=34` -- the first ordinary
   (`query_len=1`, not widened) decode step after the FF event.** No further `widened decode window
   built` line appears after position 33; every subsequent step is an ordinary single-token decode,
   and it is exactly there that FF-ON's output permanently diverges from FF-OFF's (`220` vs `313`) and
   never recovers, settling into a fixed three-token cycle (`220, 220, 201`) for the rest of the
   generation.

## 5. Why this rules out two candidates and narrows to two others

- **Rules out: FF splice content as the defect.** Already weakened by 06-d's source argument; now
  directly disproved by observation (item 1 above) -- the forced token is provably, not just
  plausibly, identical to what the unconstrained masked path independently produces.
- **Rules out: the widened step's own logits-row selection / window construction as the immediate
  cause.** The real sample taken from the widened window (`logical_position=33`) is correct. Whatever
  is wrong does not corrupt this specific forward pass or this specific row selection.
- **Narrows to: something specific to the transition from a widened (`query_len=2`) decode step back
  to an ordinary (`query_len=1`) one.** Two candidates, both already named as open threads in prior
  reports, neither instrumented or checked this session:
  1. **`num_computed_tokens` bookkeeping at the transition.** `06-c` (Sec 3) and `06-e` (Sec 5) both
     cite two guarded advance sites in `engine/mod.rs` (`if seq.num_computed_tokens() == before {
     seq.advance_num_computed_tokens(scheduled) }`, at two call sites) as consistent on paper but not
     independently re-derived against a live `scheduled` value for this exact transition. An off-by-one
     here (advancing by 1 instead of the widened step's 2, or vice versa on the following step) would
     shift every subsequent step's KV read by exactly one position from that point on -- consistent
     with a divergence that starts at one specific step and never resolves.
  2. **CUDA decode-graph re-entry immediately after a widened step.** `06-e` (Sec 6) proved grammar-
     constrained sequences cannot *enter* the CUDA-graph decode-tail/lookahead path in general
     (`SequenceRecognizer::Llguidance` fails `cuda_token_sampling_plan`'s gate on every step), but did
     not check whether that gate is correctly re-evaluated specifically in the single step immediately
     following a widened one. The server's startup log for both legs shows 8 width-1 decode graphs
     captured (`Captured CUDA decode graph: batch bucket N (1 live rows), 1 query tokens`) and ready;
     if the very next step after a `query_len=2` step is misidentified as an ordinary width-1 decode
     eligible for one of these pre-captured graphs without accounting for the extra token already
     consumed, the graph would read/write KV at a stale slot -- again consistent with a permanent,
     non-recovering divergence starting at exactly one step.

Both candidates would produce the same observed signature (correct through the forced token and the
one real sample after it, permanently wrong from the very next step on); this report does not
distinguish between them, since doing so needs source tracing or further instrumentation not done
this session.

## 6. Explicit scope statement

This is a **runtime finding and a narrowed hypothesis, not a root-cause conclusion.** It establishes,
by direct observation for the first time in this investigation, exactly which step is provably
correct (the widened FF step itself, `logical_position=33`) and exactly which step is the first
provably wrong one (`logical_position=34`). It does not identify which of the two candidates in Sec 5
is responsible, does not read `engine/mod.rs`'s advance sites or the CUDA-graph re-entry gate against
this specific transition, and proposes no fix. Per instruction, no source code was modified as part of
producing this report, and this report is being committed alone -- the instrumentation patch that
produced this data is not part of this commit.

## 7. Next steps (not run this session)

Investigate the two Sec 5 candidates separately:
1. Read (or instrument further, if reading is inconclusive) `engine/mod.rs`'s `num_computed_tokens`
   advance sites for the exact `before`/`scheduled` values at the widened-to-ordinary transition for
   this reproduction.
2. Read (or instrument further) whichever code path decides CUDA-graph decode-tail/lookahead
   eligibility per-step, specifically for the one step immediately following a widened FF step, to
   confirm or refute that a captured width-1 graph is (correctly or incorrectly) engaged there.

Neither was started this session.

## Files added

- `plans/ff-round-two/reports/06-f-runtime-divergence-point.md` (this report). No other files
  changed; no source code touched or committed.

## Git/worktree state

- `/workspace` (the host's separate checkout used for the instrumented build and live runs, not this
  worktree): branch `grammar-fast-forward`, still carries the local, uncommitted instrumentation
  patch described in Sec 2. That patch is not part of this commit and was not touched by this report.
- This report was written and committed from a freshly recreated worktree at `/tmp/ff-artifacts-wt`
  (the prior worktree there had been pruned between sessions, consistent with `06-e`'s note that this
  happens), branch `ff-demo-artifacts`, independently verified distinct from `/workspace` (`git
  worktree list`, distinct device/inode).

Not proceeding to further investigation this session, per instruction.
