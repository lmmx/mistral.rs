# 09 — Runtime / API surface parity (dependency note, not a design)

**Class:** verification. **Owned elsewhere.**

## Ownership

The second-round entry lists the missing CLI / Python / server-builder / Rust-builder surface for
`MISTRALRS_GRAMMAR_FAST_FORWARD` under Missing. That surface is **already specified** in a separate
document: `docs/journal/2026-09-08-fast-forward-runtime-toggle.md` and its accompanying
`.patch`, which live in the **sumac** repository, not in this one — a coding agent working in this
checkout cannot follow that reference.

So: **do not design a toggle here.** If you invent an interface in this repo and that plan lands its
own, the two will conflict. This plan restates the contract that the runtime-toggle design must
satisfy, so that whoever implements it can be checked against the code as it actually stands.

## What the code does today (verify each before relying on it)

- `perf_flags::grammar_fast_forward_enabled` (`perf_flags.rs:33-35`) reads
  `MISTRALRS_GRAMMAR_FAST_FORWARD` through a `OnceLock` defaulting to `false`.
- Each pipeline reads it **once at load**: `pipeline/normal.rs:289`, `pipeline/gguf.rs:1415`,
  `pipeline/ggml.rs:400`. The flag is therefore a **process-wide load-time constant** with no
  per-request, per-model or runtime override.
- At those three sites the value is `perf_flags::grammar_fast_forward_enabled() && !no_kv_cache &&
  !is_xlora`. Note the binding differences recorded in the development plan: `no_kv_cache` is a
  struct field at `gguf.rs:1415` and `ggml.rs:400` and a local only in `build_normal_pipeline`;
  `is_xlora` is a local at all three (`gguf.rs:1319`, `ggml.rs:310`, `normal.rs:170`).
- `GeneralMetadata::supports_grammar_fast_forward` (`pipeline/mod.rs:1316`) is set at all seven
  construction sites, and is `false` at `multimodal.rs:1531`, `speech.rs:330`, `diffusion.rs:252`
  and `embedding.rs:706`.

## Contract any runtime toggle must satisfy

1. **Scope must be stated.** Per-request, per-model, or per-process — pick one and say so. A
   per-request toggle is the largest change, because the value is currently read once at load and
   consumed through `GeneralMetadata`, which is handed out as an `Arc` on every step.
2. **The conjuncts survive.** `!no_kv_cache` and `!is_xlora` must remain necessary conditions
   however the toggle is expressed. A toggle that can turn the feature on for an X-LoRA or
   `--no-kv-cache` pipeline is a regression of round one's Task 3.
3. **AnyMoE stays excluded**, per plan 03 Part A. A new toggle must not route around the
   `AnyMoePipeline::get_metadata` override.
4. **The four pipelines that hold it `false` stay `false`** unless their input processors learn to
   consume the field (that is D2, deferred and unstarted).
5. **Surfaces to cover**, per the `bea02b2c4` precedent: `mistralrs-cli/src/args/mod.rs`,
   `mistralrs-pyo3/mistralrs.pyi`, `mistralrs-server-core/src/mistralrs_for_server_builder.rs`,
   the `mistralrs/src/` builder surface, plus docs and examples.
6. **The env var must keep working**, or its removal must be a deliberate, documented break — it is
   what `ff_bench.py`, `RESULTS.md` and every experiment in plans 03–06 drive the feature with.

## Deliverable

No code. When the runtime-toggle implementation appears, check it against points 1–6 and record the
result in `plans/ff-round-two/reports/09-parity-check.md`. If it has not appeared by the time plans
01–08 are complete, record that the surface gap remains open and is owned externally.
