# 08 — Test-coverage gaps and documentation divergences

**Class:** fix. **Blocked by:** 01 for the tests. The documentation half needs no toolchain and may
proceed even if 01 hands back blocked, labelled as unverified-by-build.

Independent of plans 02–07 except where ownership is noted below. Can run in parallel.

## Part A — test coverage

`mistralrs-core` has **no `tests/` directory**; the norm is inline `#[test]` modules. Follow it. The
branch added three tests and none to `sampling.rs`, `inputs_processor.rs` or `sequence.rs`, which
hold 11, 17 and 42 existing ones respectively — so there is an established local style at each site
to copy.

Uncovered today:

1. **`apply_pending_ff_tokens`** (`pipeline/sampling.rs:724`). The splice replay path. Cover: a
   splice replays into the sequence token by token; the token-trie requirement
   (`sampling.rs:738` errors without one) is exercised; the return value that signals the sequence
   finished mid-splice is exercised.
2. **The splice branch of `make_completion_chunk`** (`pipeline/inputs_processor.rs:1597-1632`),
   including the `(query_len - 1, 1)` logit narrowing.
3. **The `full_query_lens` substitutions** (`pipeline/inputs_processor.rs:1706-1725`).
4. **Both `anyhow::bail!` guards** — `inputs_processor.rs:1564-1571` (unresolved splice reaching
   `make_completion_chunk`) and `:2440-2451` (any splice reaching `make_completion_prefill_chunk`).
   Assert the error, not just the absence of a panic.
5. **The `Matcher::rollback` half of `discard_pending_ff_tokens`** (`sequence.rs:1261-1279`),
   including the failure path that sets `SequenceState::Error`.

Plan 02 adds the terminal-lifecycle tests; do not duplicate them here.

**Flag-on/off token equality is not a unit test.** `perf_flags` reads the env var through a
`OnceLock` at load (`perf_flags.rs:33-35`), so both settings cannot exist in one process. That check
is owned by plan 05's `equality` mode and its driver. Note this in a comment where a reader would
otherwise expect such a test, so the next reviewer does not file it as a gap again.

## Part B — documentation divergences

Five were recorded. Two are owned elsewhere; do not do them twice.

| # | Divergence | Owner |
|---|---|---|
| 1 | `structured-output.mdx` lists X-LoRA, multimodal and `--no-kv-cache` as the exclusions; AnyMoE inherits the feature through `AnyMoePipeline::get_metadata` (`pipeline/amoe.rs:247-249`) and changes expert-selection granularity (`amoe/mod.rs:263-267`) | **plan 03** |
| 2 | `observability.mdx` drop-rate PromQL has a denominator the numerator cannot match | **plan 02** |
| 3 | `structured-output.mdx` presents the whole-batch discard as a property of splice lengths alone | here |
| 4 | `throughput-tuning.mdx:163` calls the `perf_flags.rs` switches "only for debugging and benchmarking comparisons" | here |
| 5 | `environment-variables.md:64` places `MISTRALRS_GRAMMAR_FAST_FORWARD` under "Server and UI" while the other two `perf_flags.rs` entries sit under "CUDA acceleration" (`:70`, `:73`) | here |

Corrections for the three owned here:

**3.** The sentence reads: "Span length tracks each request's own position in its own grammar, so two
concurrent grammar-constrained requests ordinarily agree on nothing, and the batch falls back to the
flag-off baseline for that step." That is true but incomplete. The discard is also a property of
`completion_batch_indices` composing the batch without reading splice lengths
(`paged_attention/scheduler.rs:547-579`), and of `discard_pending_ff_tokens` having no
partial-rollback caller (`sequence.rs:1261-1263`). Rewrite to attribute all three, **without**
promising any of plan 07's options — this is a description of current behaviour, not a roadmap.

**4.** `MISTRALRS_GRAMMAR_FAST_FORWARD` is the only entry in `perf_flags.rs` that defaults to **off**
and the only one that turns a behaviour **on**; the other two are debugging switches for behaviour
that is on by default. Reword the sentence so it stops describing all three as the same kind of
thing.

**5.** Pick one home for the variable and make the cross-reference agree with it. "Server and UI" is
wrong for a core inference switch and "CUDA acceleration" is wrong for a flag with no CUDA
dependency; if neither table fits, add a row to the most appropriate existing table rather than
inventing a new section, and update the `/reference/environment-variables/#…` anchor that
`structured-output.mdx` links to. Whichever is chosen, the link and the table must match — the
current defect is the mismatch, not the choice.

## Also record, do not fix

The second-round entry's Missing list includes: `MISTRALRS_GRAMMAR_FAST_FORWARD` has no counterpart
in `mistralrs-cli/src/args/mod.rs`, `mistralrs-pyo3/mistralrs.pyi`,
`mistralrs-server-core/src/mistralrs_for_server_builder.rs` or the `mistralrs/src/` builder surface,
all of which the closest merged precedent `bea02b2c4` (MTP speculative decoding, 146 files) extended
for its decode-shape change. **That is plan 09's subject and is owned by another document. Do not
add CLI, pyo3 or builder surface here.**

## Exit criteria

Five test gaps covered and passing; divergences 3, 4 and 5 corrected; 1 and 2 verified as closed by
their owning plans (or noted as still open if those plans have not run).
