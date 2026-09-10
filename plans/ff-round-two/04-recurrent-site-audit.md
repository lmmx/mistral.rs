# 04 — Audit the nine `SpeculativeDecode` sites (New A)

**Class:** prove/disprove. **Blocked by:** 01; the run half also by 05.

## Explicit non-goal

**Do not modify any of the nine branches in this plan.** Not one. The speculative arm looks like the
faster and more appropriate arm; hybrid recurrent state management is exactly where an apparently
innocuous branch change introduces silent numerical or state corruption. The deliverable here is
evidence and a table. An implementation proposal comes after, separately, and only for sites the
table clears.

**W3 stands** (`docs/journal/2026-09-09-fast-forward-development-plan.md`): do not add a
`RecurrentBatchKind::FastForward` variant. It would not make any of these nine fire anyway — they
test for `SpeculativeDecode` by name.

## The finding

Round one's W3 audited the twelve `== RecurrentBatchKind::Decode` comparisons and asked whether a
wider window breaks any. The complementary set was never audited. Nine sites test for
`RecurrentBatchKind::SpeculativeDecode` specifically, each to *enable* a path that only makes sense
when the decode window is wider than one token. A fast-forward window reaches all nine labelled
`Decode` (`recurrent_batch_kind_for_input`, `pipeline/mod.rs:733-744`, carries no fast-forward
term), so all nine take their other branch.

In each case the other branch is the conservative one — flush deferred state, snapshot the
checkpoints, apply pending transitions eagerly. So the expected failure mode is **cost, not
corruption**: a hybrid-recurrent model on CUDA pays the general path on every splice step while the
feature exists to save forward passes. "Expected" is doing work in that sentence; that is the claim
under test.

## Deliverable 1 — the audit table (reading only, no build required)

One row per site. Publish to `plans/ff-round-two/reports/04-site-audit.md`.

| Site | Arm taken with FF | What the `SpeculativeDecode` arm does | What the taken arm does | State contract the spec arm assumes | Safe to share with an FF window? | Evidence |
|---|---|---|---|---|---|---|

The nine sites:

1. `gdn/layer.rs:532` — CUDA speculative-checkpoint path in `forward_speculative_checkpoints`
2. `vision_models/qwen3_5/text.rs:850` — `should_stash_gdn_replay`
3. `vision_models/qwen3_5/text.rs:2321-2324` — `speculative_gdn`, gating `transition_gdn`
4. `pipeline/normal.rs:2210` — `transitions_supported` in `snapshot_hybrid_recurrent_checkpoints`
5. `pipeline/multimodal.rs:1874`
6. `pipeline/multimodal.rs:2233`
7. `vision_models/mod.rs:184`
8. `vision_models/mod.rs:243`
9. `pipeline/cuda_graph.rs:2379`

Round one's F3 is instance 3 (`vision_models/qwen3_5/text.rs:2340-2346` gates `deferred_gdn` on
`query_len == 1`, so every splice step on a CUDA Qwen3.5 build falls to
`flush_deferred_recurrent_state`). It is **owned by B1** in the development plan; this table records
it and does not restart its design.

The "state contract" column is the load-bearing one. For each speculative arm, write down what it
assumes about the window it is given — in particular whether it assumes the extra tokens are
*proposals that may be rejected* (speculative semantics) rather than *tokens already committed to
the sequence* (fast-forward semantics). That distinction, not the width, is what decides sharability,
and it is why "the window is wider than one" is not sufficient justification to make a site fire.

Any row whose "safe to share" answer is not a confident yes stays `unknown`. Unknown is an
acceptable and expected outcome for most rows.

## Deliverable 2 — control-flow confirmation (CPU, cheap)

New A is a claim about control flow, so confirm it as one.

- Add a `tracing::debug!` in **each** of the nine arms — both the `SpeculativeDecode` arm and the
  arm actually taken — recording a stable site identifier and which arm ran. Use the existing
  `tracing` machinery; **do not add a new env var** and do not add counters.
- Run one CPU inference per recurrent family with the flag set and a forcing grammar, and collect
  which arm each site reports.
- This alone confirms or refutes New A. It says nothing about cost or numerics.

Model per family, smallest first:

| Family | Code exercised | Checkpoint | Approx. size | CPU |
|---|---|---|---|---|
| Short convolution | `models/lfm2.rs` | `LiquidAI/LFM2.5-230M` | 0.46 GB | yes |
| Gated delta net | `gdn/backend.rs`, `gdn/layer.rs`, `vision_models/qwen3_5/text.rs` | `Qwen/Qwen3.5-0.8B` | ~1.5 GB | yes, tight |
| Mamba SSM | `models/granite.rs` | Granite 4 **hybrid** line, 350M member | <1 GB | yes |

Three constraints on that table:

- **Verify the Granite checkpoint before downloading weights.** `models/granite.rs` builds a
  `MambaLayer` only for layers whose `layer_types` entry is `GraniteLayerType::Mamba`
  (`granite.rs:55`, `:94`). The dense Granite 4 variants carry **no** such entry and exercise no
  Mamba code. Fetch `config.json` alone first and check `layer_types`; the hybrid line carries `-h-`
  in the repo name. The exact repo ids are unverified in the research — if no hybrid member under
  ~2 GB exists, record that and mark the Mamba row blocked rather than substituting a dense model.
- Qwen3.5-0.8B replaces the 4B the demo used and reaches the same GDN code.
  `Qwen/Qwen3-Next-80B-A3B-Instruct` is the only released Qwen3-Next size (~163 GB) and shares
  `gdn/` with Qwen3.5, so Qwen3.5-0.8B covers `models/qwen3_next.rs`'s recurrent behaviour without
  it.
- LFM2 is the family W3's argument turns on (`models/lfm2.rs:1283-1284`), so LFM2.5-230M is also the
  model that would catch a regression if anyone revisits the enum-variant decision.

Whether the instrumentation commit is kept or reverted after the run is a judgement call: keep it if
the messages read as useful production debug lines, revert it if they read as scaffolding. Say which
you did.

## Deliverable 3 — the CUDA half, specified and marked blocked

`gdn/layer.rs:532` and the Qwen3.5 `transition_gdn` / `deferred_gdn` paths are CUDA-gated, so
"is the taken branch numerically right, and what does it cost?" needs a CUDA build. Write the
experiment spec now so it is ready when a machine is:

- flag-off vs flag-on token equality on a hybrid-recurrent model under a partially forcing grammar,
  greedy, fixed seed, two processes (the `OnceLock` constraint from plan 05);
- tokens/s and forward-pass count for both, on a completion with a high forced fraction;
- per-site arm counts from Deliverable 2's instrumentation, to confirm the CPU result holds on CUDA.

Mark it **BLOCKED — needs CUDA**, and cross-reference B1 and R1 in the development plan, which carry
the same blocker. Do not attempt to emulate it.

## Exit criteria

Audit table published with every row answered or explicitly `unknown`; the nine-site arm log
collected on at least the families whose checkpoints were obtainable; the CUDA spec written and
marked blocked. **No branch modified.**
