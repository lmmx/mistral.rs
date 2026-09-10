# 05 — The experimental harness

**Class:** supporting. **Blocked by:** 01. **Needed by:** 03 Part B, 04 Deliverable 2, 06.

## What exists and why it cannot answer any of the open questions

`ff_bench.py` on `ff-demo-artifacts` (78 lines) builds a `Runner` over `Which.GGUF`
(`unsloth/Qwen3.5-4B-GGUF`), sends one chat completion with
`grammar_type="regex"`, `grammar=re.escape(PASSAGE)` at `temperature=0.0`, and times repeats. Three
limits:

1. **`Which.GGUF` only.** None of the models plans 03 and 04 need is available here as GGUF.
2. **A fully forcing regex at `temperature=0.0` pins both runs to the same string by construction.**
   It can never detect a divergence, which is exactly what plans 03 and 04 must detect.
3. **No concurrency.** Plan 06 needs several in-flight requests carrying *different* grammars.

`ff_bench.py` and `RESULTS.md` are the reproducibility record for the 6.1–6.2x demo measurement.
**Do not modify either.** Add a new `ff_harness.py` beside them.

## The constraint that shapes everything

`perf_flags::grammar_fast_forward_enabled` reads `MISTRALRS_GRAMMAR_FAST_FORWARD` through a
`OnceLock` (`perf_flags.rs:33-35`) and each pipeline reads it once at load, so the flag is a
**process-wide load-time constant**. Every flag-on/flag-off comparison must therefore be **two
processes**, not two `Runner`s and not two requests. Build the harness so a single invocation runs
one flag setting and writes a JSON report, and a driver script runs it twice and diffs the reports.
Anything that tries to toggle the flag inside one process is measuring nothing.

## Modes to build

### `equality` — for plans 03 and 04

- Loader: `Which.Plain` (model id, arch, dtype from CLI args), so any HF checkpoint works.
- One request, fixed prompt, fixed `max_tokens`, `temperature=0.0`, fixed seed, `enable_thinking=False`.
- Grammar: a **partially** forcing JSON schema, supplied as a fixture file, not a fully forcing
  regex. Fixture requirement: it must contain forced spans long enough for splices to be staged
  (object braces, quoted key names, `":"` separators) **and** free spans where the model chooses
  (string values, an enum with several members, a number). Commit at least one such fixture under
  `plans/ff-round-two/fixtures/`.
- Output: JSON with the full token id list, the decoded text, the finish reason, per-step timings,
  and the flag value read from the environment.
- The driver compares two reports and asserts token-id-for-token-id equality, reporting the first
  divergent index and both continuations when they differ.

### `routing-log` — for plan 03 Part B

`equality` plus collection of the per-forward `topk(1)` expert index logged at `amoe/mod.rs:266`
(via `RUST_LOG` capture or a structured sink), keyed by layer and batch row, emitted into the same
JSON report so the driver can diff routing as well as tokens.

### `arms` — for plan 04 Deliverable 2

`equality` plus collection of the nine per-site arm messages, aggregated to a count per site per arm
in the JSON report.

### `concurrency` — for plan 06

- Submit **N concurrent** requests, each with its own grammar drawn from a fixture set, plus a
  configurable fraction of unconstrained requests.
- Requires a build where `paged_attn_supported()` is true — it is a compile-time `const fn`
  returning `false` on a CPU build (`utils/mod.rs:297-305`) — so **CUDA or Metal only**.
- Scrape the Prometheus counters before and after the run and record the deltas, not the absolute
  values.
- Parameters: `N`, schema-set (identical / differing), unconstrained fraction, duration or request
  count, seed.

## Report layout

All modes write to `plans/ff-round-two/reports/` as
`<plan>-<mode>-<flag>-<timestamp>.json`, plus a human-readable `.md` summary written by the driver.
Reports are committed to `ff-demo-artifacts`; harness code lives beside `ff_bench.py` on the same
branch.

## Non-goals

- No new inference features, no changes to `mistralrs-core` from this plan.
- Do not fold the timing loop into the equality modes — timing under a partially forcing grammar is
  not comparable to `RESULTS.md` and inviting that comparison is how a misleading number gets born.
