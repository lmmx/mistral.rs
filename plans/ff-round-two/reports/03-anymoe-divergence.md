# 03 -- AnyMoE grammar fast-forward divergence report

Status: Part A (code fix) implemented and committed. Part B (empirical two-process run) **not executed** in this session -- see
"Why Part B was not run" below. This document is therefore evidence gathered from the source
tree only, plus the exact setup a future run should use.

Note on location: this report belongs at `plans/ff-round-two/reports/03-anymoe-divergence.md` on
the `ff-demo-artifacts` branch per the plan's own layout, but the working session is restricted to
the `grammar-fast-forward` branch and told not to switch branches or touch the `plans/` tree (which
does not exist as a file in this branch's working copy -- it lives only on `ff-demo-artifacts`).
This file is therefore left in the scratchpad; move it into the reports directory on
`ff-demo-artifacts` by hand, or ask a session with access to that branch to do so.

## What is established by reading the code (no execution required)

`MoeMlp::forward` (`mistralrs-core/src/amoe/mod.rs:257-285`):

```rust
fn forward(&self, xs: &Tensor) -> Result<Tensor> {
    // ^ [b, s, h]
    let gate = self.gate.forward_t(xs, self.training)?;
    // ^ [b, s, n_e]  (softmax over experts, per position)
    let gate = gate.mean(1)?;
    // ^ [b, n_e]      (mean over the *sequence* dimension)
    let TopKOutput { values: _, indices } = gate.topk(1)?;
    ...
    let indices = indices.reshape((b, 1, 1, 1))?.expand((b, 1, s, h))?;
    let gathered_outputs = stacked_outputs.contiguous()?.gather(&indices.contiguous()?, 1)?;
    gathered_outputs.squeeze(1)
}
```

Three facts follow directly from this code, independent of any specific checkpoint:

1. **One expert index is chosen per forward call, per batch row, for the whole window.** The
   `topk(1)` index is computed once from the window-mean gate probability and then broadcast
   (`.expand((b, 1, s, h))`) across every position `s` in that window via `gather`. There is no
   per-position expert selection inside a single forward call.
2. **At `s == 1` (ordinary decode, flag off) this degenerates to per-token selection.** The mean
   over a length-1 sequence dimension is a no-op, so each decode step's expert choice is driven
   only by that step's own token.
3. **At `s == 1 + K` (a widened grammar fast-forward decode window) the same call must serve `K+1`
   positions with one expert**, chosen from the softmax-probability mean of `K+1` positions: the
   freshly sampled token plus `K` grammar-forced tokens replayed as already-known. Whether that
   window-mean argmax equals what each of the `K+1` positions would have chosen individually is a
   property of the gate weights and the specific token embeddings in the window -- it is not
   guaranteed either way by the architecture. The mechanism can select a different expert for the
   forced positions than they would have gotten under `s == 1` decode whenever the per-position
   argmax experts inside the window are not already unanimous. It is equally possible for a given
   checkpoint, on a given window, for all positions to agree, in which case output is unchanged
   for that window by luck rather than by design.

This settles the mechanism-level question the plan poses ("can grammar FF alter expert
selection/output") in the affirmative, by construction, independent of the checkpoint used. It does
**not** settle whether any specific checkpoint (e.g. Qwen3-0.6B/Qwen3-0.6B under the plan's Part B
setup) actually diverges on real prompts -- that is an empirical question requiring the run below.

## MTP / speculative-decoding interaction

`mistralrs-core/src/speculative/driver.rs:266` and `verifier.rs:822` both call
`target.get_metadata()` (or `pipeline.get_metadata()`) to read pipeline capabilities, the same
`GeneralMetadata::supports_grammar_fast_forward` field gated in
`pipeline/sampling.rs:849` (`let supports_fast_forward = metadata.supports_grammar_fast_forward;`
-- the single call site that decides whether to stage a splice at all). There is no AnyMoe-specific
wiring inside `speculative/`; if an `AnyMoePipeline` is ever used as a speculative target, it goes
through the same `get_metadata()` call this fix now overrides, so the Part A change closes this
path too without any speculative-specific code. No further action is needed here; this is recorded
because the plan asked for it to be checked explicitly, not because a defect was found.

## Why Part B (the two-process empirical run) was not executed

Part B requires downloading two copies of Qwen3-0.6B, building `mistralrs-cli`/`mistralrs-server`
from source, training an AnyMoE gate, and running two full inference processes (flag unset, flag
set) with logging patched into `amoe/mod.rs:266`. This session has network access but **no Rust
toolchain is installed** (`cargo`/`rustc` not found anywhere on `$PATH` or in common install
locations), and the operating instructions for this task explicitly say not to spend time
provisioning Rust. Building the crate is a hard prerequisite for running any of Part B, so it could
not be done here.

Everything else Part B specifies (model choice, gate-training config, prompt/sampling
discipline, instrumentation point) is recorded below exactly as the plan requires, ready to run
once a Rust toolchain is available.

### Setup, exactly as specified by plan 03

- Base and expert: Qwen3-0.6B for both.
- Gate training data: `examples/amoe.json` (ten rows), `layers = [0, 1, 2]`, `epochs = 25`.
- `hidden_size`: read from the downloaded base model's `config.json` (`Qwen3Config.hidden_size`),
  not the TOML default.
- Grammar: a partially forcing JSON schema (not a fully forcing `re.escape(...)` regex, which
  cannot diverge by construction at `temperature=0.0`).
- Sampling: `temperature=0.0`, fixed seed, fixed prompt, fixed `max_tokens`,
  `enable_thinking=False`.
- Two separate OS processes: one with `MISTRALRS_GRAMMAR_FAST_FORWARD` unset, one with it set to
  `1` -- `perf_flags::grammar_fast_forward_enabled` reads through a `OnceLock` at load
  (`mistralrs-core/src/perf_flags.rs:33-35`), so a single process cannot exercise both settings.
- Part A gates AnyMoE off at the source, so this run needs Part A's override temporarily bypassed
  -- e.g. a scratch commit that reverts the `anymoe_metadata_override` call in
  `AnyMoePipeline::new`, or a `#[cfg(test)]`/env-gated escape hatch not reachable from a normal
  build. **Do not weaken the Part A gate in shipped code to make this experiment convenient**, per
  the plan.
- Instrumentation: log the `topk(1)` index computed at `amoe/mod.rs:266` (now shifted a few lines
  by nothing Part A touched in `amoe/mod.rs` -- Part A only edited `pipeline/amoe.rs`), per
  forward, per layer, per batch row, to a file each process can be diffed against.

### Suggested commands (untested, for the next run)

```bash
# once a Rust toolchain is available:
cargo build --release --features cuda   # or the appropriate feature set for the host

huggingface-cli download Qwen/Qwen3-0.6B --local-dir /tmp/qwen3-0.6b-base
huggingface-cli download Qwen/Qwen3-0.6B --local-dir /tmp/qwen3-0.6b-expert

# read hidden_size out of /tmp/qwen3-0.6b-base/config.json and put it in the AnyMoE TOML,
# with layers = [0, 1, 2], epochs = 25, model_ids = ["/tmp/qwen3-0.6b-expert"]

MISTRALRS_GRAMMAR_FAST_FORWARD=0 ./target/release/mistralrs-cli run \
  --format anymoe -m <anymoe-config>.toml \
  --prompt-file prompt.txt --json-schema partial.schema.json \
  --temperature 0.0 --seed 42 --max-tokens 256 > /tmp/run_ff_off.log

MISTRALRS_GRAMMAR_FAST_FORWARD=1 ./target/release/mistralrs-cli run \
  --format anymoe -m <anymoe-config>.toml \
  --prompt-file prompt.txt --json-schema partial.schema.json \
  --temperature 0.0 --seed 42 --max-tokens 256 > /tmp/run_ff_on.log

diff <(grep '^TOKEN' /tmp/run_ff_off.log) <(grep '^TOKEN' /tmp/run_ff_on.log)
diff <(grep '^EXPERT_IDX' /tmp/run_ff_off.log) <(grep '^EXPERT_IDX' /tmp/run_ff_on.log)
```

## The four-row outcome table

| Tokens | Expert indices | Conclusion |
|---|---|---|
| identical | identical | The only outcome that would support re-enabling. Report it; **do not re-enable** -- hand back for a semantic decision. |
| identical | differ | Routing diverges; equality here is luck of one checkpoint. Stays disabled. |
| differ | (either) | Confirmed output divergence. Stays disabled; record it in `structured-output.mdx`. |
| **not run** | **not run** | **This session's status.** No Rust toolchain available to build and execute the two-process comparison; see "Why Part B was not executed" above. The code-level argument above already shows the mechanism *can* land in the "differ" rows for a generic checkpoint, but which row a specific Qwen3-0.6B run actually lands in is unconfirmed. |

## What is and is not concluded

- **Confirmed (code-level, not checkpoint-dependent):** grammar fast-forward widens the window fed
  into `MoeMlp::forward`, which computes and applies exactly one expert index per window rather
  than one per token. This is a genuine computational difference, not a latency-only change.
- **Not confirmed:** whether that computational difference actually changes sampled tokens or
  logged expert indices for any real checkpoint and prompt. No inference was run in this session.
- **No re-enable decision is implied by any of the above**, per the plan's instruction. Part A's
  gate stays in place regardless of what a future empirical run finds.
