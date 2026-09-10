# 03 — Make AnyMoE safe, then gather evidence (New D)

**Class:** fix (Part A) + prove/disprove (Part B). **Blocked by:** 01 for Part A; 01 and 05 for
Part B. **Part B does not gate Part A.**

## The finding

`AnyMoePipeline::get_metadata` (`pipeline/amoe.rs:247-249`) returns the wrapped pipeline's
`GeneralMetadata` unchanged, so an AnyMoE model over a normal, GGUF or GGML pipeline reports
`supports_grammar_fast_forward: true` whenever `MISTRALRS_GRAMMAR_FAST_FORWARD` is set. No entry on
the branch records a decision to include AnyMoE.

`MoeMlp::forward` (`amoe/mod.rs:258-284`) takes `[b, s, h]`, computes the gate, reduces it with
`gate.mean(1)` — the mean **across the sequence dimension** — and picks one expert per batch row
with `topk(1)` for the whole window. At `s == 1` each generated token picks its own expert. At
`s == 1 + K` one expert serves all `1 + K` positions, chosen from a mean gate over a window mixing
the sampled token with K grammar-forced ones.

So the *granularity of expert selection provably changes with the flag*. Whether the selected
expert also changes on any given checkpoint is unknown.

## Why Part A does not wait for Part B

Fast-forward is defended throughout the branch and its documentation as an optimisation that cannot
change output. On AnyMoE it demonstrably changes the computation. Correctness must not rest on an
experiment showing that one 0.6B checkpoint happens not to diverge.

The engineering question is not "does this model diverge?" but **"can we honestly expose
fast-forward for AnyMoE while keeping the output-preservation contract?"** Until someone makes that
semantic decision deliberately, the answer is no.

## Part A — exclude AnyMoE from fast-forward (do this now)

`AnyMoePipeline::get_metadata` must not advertise `supports_grammar_fast_forward: true`.

Implementation notes, in preference order:

1. `get_metadata` returns `Arc<GeneralMetadata>` and is called on every step, so **do not clone and
   mutate per call**. Build the overridden `Arc<GeneralMetadata>` once — at `AnyMoePipeline`
   construction, or lazily in a `OnceLock`/`RwLock` cache keyed on the target's metadata pointer —
   and return the cached `Arc`.
2. If the wrapped metadata is rebuilt during AnyMoE's gate-training lifecycle (check
   `pipeline/amoe.rs` for anywhere the target pipeline is replaced or re-loaded), the cache must be
   invalidated there. Verify this before choosing the construction-time variant.
3. Whatever the mechanism, **only** `supports_grammar_fast_forward` may be altered. Every other
   field passes through untouched.

Test: an inline `#[test]` constructing (or stubbing) an `AnyMoePipeline` over metadata with
`supports_grammar_fast_forward: true`, asserting `get_metadata().supports_grammar_fast_forward` is
`false`. If the type is too heavy to construct in a unit test, factor the override into a small
free function (`fn anymoe_metadata_override(inner: &GeneralMetadata) -> GeneralMetadata`) and test
that instead — the test must cover the override rule, not the plumbing.

Documentation: `structured-output.mdx` currently says fast-forward "is not available under X-LoRA or
for multimodal pipelines" and names `--no-kv-cache`. Add AnyMoE to that list, with one sentence
saying why (per-window rather than per-token expert selection). This closes the first
`structured-output.mdx` entry in the second-round entry's Divergence list; plan 08 does not repeat
it.

## Part B — the divergence experiment (evidence for a future re-enable)

Cheapest of the three open questions and needs no GPU: `AnyMoeLoader` warns and disables
PagedAttention (`pipeline/amoe.rs:66-70`), so this path never wanted a paged build.

Corrections to the folklore, already established by the research and not to be re-litigated:
AnyMoE is **not** tied to Mistral-7B. `create_anymoe_layers` is implemented by fourteen
architectures including `models/qwen3.rs`, `models/qwen2.rs` and `models/smollm3.rs`, and
`AnyMoeConfig::hidden_size` (`amoe/mod.rs:144`) feeds `linear(config.hidden_size, n_experts, vb)`
(`amoe/mod.rs:207`), so it is the **base model's own `hidden_size` from its `config.json`**, not the
4096 the shipped TOML hardcodes.

Setup:

- Base and expert: **Qwen3-0.6B for both** (~2.5 GB on disk for the pair).
- Gate training data: `examples/amoe.json` from this repo, ten rows. Cut `layers` to `[0, 1, 2]` and
  `epochs` to `25` to keep the fitting pass to minutes.
- `hidden_size`: read it out of the base model's `config.json`; do not use the TOML default.
- Grammar: a **partially** forcing JSON schema, not `ff_bench.py`'s fully-forcing
  `re.escape(PASSAGE)` regex. A fully forcing regex at `temperature=0.0` pins both runs to the same
  string by construction and can therefore never fail.
- Sampling: `temperature=0.0`, fixed seed, fixed prompt, fixed `max_tokens`, `enable_thinking=False`.
- Two **processes**: one with `MISTRALRS_GRAMMAR_FAST_FORWARD` unset, one with it set.
  `perf_flags::grammar_fast_forward_enabled` reads through a `OnceLock` at load
  (`perf_flags.rs:33-35`), so a single process cannot test both settings.
- Part A gates AnyMoE off, so the experiment needs the gate temporarily lifted. Run Part B from a
  scratch commit that reverts Part A, or put Part A's override behind a test-only escape hatch that
  is not reachable from a normal build. **Do not weaken Part A in the shipped code to make the
  experiment convenient.**

Instrumentation that makes it diagnostic rather than pass/fail: log the `topk(1)` index chosen at
`amoe/mod.rs:266` per forward, per layer, per batch row.

Read the outcome off this table:

| Tokens | Expert indices | Conclusion |
|---|---|---|
| identical | identical | The only outcome that would support re-enabling. Report it; **do not re-enable** — hand back for a semantic decision. |
| identical | differ | Routing diverges; equality here is luck of one checkpoint. Stays disabled. |
| differ | (either) | Confirmed output divergence. Stays disabled; record it in `structured-output.mdx`. |

## Deliverable

- Part A: code + test + docs, one commit.
- Part B: `plans/ff-round-two/reports/03-anymoe-divergence.md` — setup, exact commands, the two
  token streams, the per-forward expert-index logs, and the table row reached. Numbers and
  transcript; no re-enable decision.
