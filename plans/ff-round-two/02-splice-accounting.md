# 02 — Close the splice-accounting defect (New E)

**Class:** fix — an established defect. **Blocked by:** 01. **Prerequisite for:** 06.

## The defect

Three counters exist:

| Counter | Incremented at |
|---|---|
| `mistralrs_grammar_ff_splices_staged_total` | `pipeline/sampling.rs:1630`, inside `sample_sequence` |
| `mistralrs_grammar_ff_tokens_fed_total` | `engine/mod.rs:1851`, when a splice reaches a decode window |
| `mistralrs_grammar_ff_splice_drops_total{reason}` | `sequence.rs:1276`, inside `discard_pending_ff_tokens` |

Within one `sample_and_add_toks_inner`, a splice is staged inside `sample_sequence`, and the sampled
token is applied **after** that by `finish_or_add_toks_to_seq`. If that token finishes the sequence
by a route that does not discard the splice, the sequence is dropped holding a splice that is
neither fed nor dropped.

`observability.mdx` tells operators to watch
`rate(…splice_drops_total) / rate(…splices_staged_total)`. A denominator holding splices the
numerator can never count makes that ratio read low — and it reads low exactly on short
grammar-constrained completions, which is the tool-call shape the feature targets.

## The deliverable is an invariant, not a patch

> **`splices_staged_total == tokens_fed_total_events + splice_drops_total`**, for every splice, over
> the life of the process, modulo splices in flight at scrape time.

(`tokens_fed_total` counts *tokens*, not splices. Decide during Step 2 whether to add a
splice-granular fed counter or to state the invariant in token terms with a matching
`ff_tokens_dropped_total`; see Step 3. State whichever you choose in the docs.)

The call site to patch **follows from the audit**. Do not patch the obvious length-cap path and
declare victory — the point of Step 1 is to find every path that can strand a splice.

## Step 1 — terminal lifecycle audit (do this first, output a table)

For each site below, answer: *can a sequence reach here holding a non-empty
`pending_ff_tokens`, and if so what happens to it?*

Termination sites on `grammar-fast-forward`:

- `engine/mod.rs:948` (`SequenceState::Error`), `engine/mod.rs:1540` (`GeneratedImage`)
- `paged_attention/scheduler.rs:1242`, `:1449`, `:2202` (`Canceled`), `:1985`, `:2143` (`Eos`),
  `:1990` (`Error`)
- `pipeline/response.rs:70`, `:102` (image/speech), `:129`, `:151` (`Length(0)`)
- `pipeline/sampling.rs:225` (`Eos`), `:481` (`Canceled`), `:494`, `:504` (computed `reason`)
- `scheduler/default_scheduler.rs:240`, `:337` (`Canceled`)

And for each `StopReason` variant: `Eos`, `Length`, `ModelLength`, `StopTok`, `StopString`,
`Canceled`, `ToolCalls`, `GeneratedImage`, `GeneratedSpeech`.

Two paths are already handled and should appear in the table as such: preemption
(`paged_attention/scheduler.rs:1386`, reason `"preemption"`) and reallocation (`sequence.rs:1373`,
reason `"realloc"`).

One nuance the audit must resolve rather than assume: staging is suppressed at
`pipeline/sampling.rs:1616-1619` only when `ends_turn` holds, and `ends_turn` requires **both** that
the token is an EOS token **and** that `llg.is_accepting()`. An EOS token sampled while the matcher
is not accepting therefore does **not** suppress staging, and may still reach a `StopReason::Eos`
termination. Establish whether that combination is reachable; do not assert either way from the
`ends_turn` name.

`StopReason::ToolCalls` (`pipeline/sampling.rs:346`, `:599`) deserves the same explicit treatment —
it is the terminal reason most likely to coincide with a staged splice.

Write the table to `plans/ff-round-two/reports/02-lifecycle-audit.md` **before** editing code.

## Step 2 — implement what the audit found missing

- Add a `discard_pending_ff_tokens(reason)` call, or an equivalent drain, on every stranding path
  the audit identifies.
- Prefer a single new reason constant (suggested: `"sequence_end"`) unless the audit shows
  materially different classes, in which case use one reason per class and document each. Reason
  strings are a metric label; keep the cardinality small and fixed.
- Do **not** change the staging conditions at `sampling.rs:1616-1631`. Suppressing staging earlier
  is a different (and larger) change; if the audit suggests it, record it as a proposal and hand
  back.
- Do **not** rename or re-purpose the three existing counters.

## Step 3 — make the invariant checkable

Counters registered through the `metrics` facade are awkward to assert in a unit test. Do both of:

1. **State assertions.** For each stranding path found, an inline `#[test]` that stages a splice
   (`set_pending_ff_tokens`), drives the sequence through that termination, and asserts
   `active_pending_ff_tokens().is_empty()` afterwards. Place tests inline per the crate norm —
   `mistralrs-core` has no `tests/` directory.
2. **A token-conservation counter.** If the invariant is stated in token terms, increment a
   `mistralrs_grammar_ff_tokens_dropped_total` by `splice.len()` inside `discard_pending_ff_tokens`
   alongside the existing splice counter. This is independently needed by plan 06, which must
   measure *forced tokens lost*, not just splices lost. Adding it here is the cheaper place.

## Step 4 — documentation

In `docs/src/content/docs/` (`observability.mdx`):

- state the conservation invariant explicitly;
- list every `reason` label value with the lifecycle event it denotes;
- keep the drop-rate PromQL, and add the sentence that makes it honest now that the denominator
  balances;
- document `mistralrs_grammar_ff_tokens_dropped_total` if Step 3.2 adds it.

This closes the `observability.mdx` entry in the second-round entry's Divergence list. Plan 08 does
not repeat it.

## Exit criteria

Audit table published; every stranding path either discards or is argued unreachable in the table;
tests added and passing; `observability.mdx` updated; the whole thing is one commit (or one commit
per stranding class), with the audit report referenced in the message.
