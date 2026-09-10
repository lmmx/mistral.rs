# 06-M -- GGUF metadata confirms the GDN deferred-decode fast path is active for this model (2026-09-10, seventeenth session)

Status: **one fact, verified.** No source modified, no build, no instrumentation, no further
investigation, per explicit instruction.

`06-l` (Sec 4) identified a single unresolved precondition for `06-k`'s GDN deferred-state candidate:
whether `unsloth/Qwen3.5-4B-GGUF`'s actual `linear_key_head_dim`/`linear_value_head_dim` equal `128`,
matching `GDN_DECODE_K_DIM`/`GDN_DECODE_V_DIM` (`cuda/gdn.rs:18-19`) -- the values
`deferred_decode_supported` (`gdn/layer.rs:271-282`) requires before `forward_deferred_decode` can
ever run.

## GGUF metadata (read directly from the model file on the host, via `gguf-dump --no-tensors`)

```
qwen35.ssm.state_size      = 128
qwen35.ssm.time_step_rank  = 32
qwen35.ssm.inner_size      = 4096
```

Per the exact mapping in `mistralrs-core/src/gguf/normal_config.rs:1844-1894` (`build_qwen35`):
- `linear_key_head_dim = ssm.state_size = 128`
- `linear_value_head_dim = ssm.inner_size / ssm.time_step_rank = 4096 / 32 = 128`

## Result

**Both equal 128, matching `GDN_DECODE_K_DIM`/`GDN_DECODE_V_DIM` exactly.** This is the specific
model/dimension configuration the deferred-decode fast path was built for -- not a near-miss or a
coincidence. Combined with `06-k`/`06-l`'s source trace, this confirms the deferred-decode fast path is
active throughout ordinary decode for this exact model, and the invariant-violation mechanism those
reports describe is directly applicable to this reproduction, not merely a candidate contingent on an
unverified precondition.

This report intentionally goes no further than recording this fact, per instruction.

## Files added

- `plans/ff-round-two/reports/06-m-gdn-deferred-decode-confirmed-active.md` (this report). No other
  files changed; no source code touched, built, or benchmarked.

## Git/worktree state

- `/workspace`: branch `grammar-fast-forward`, HEAD `cab8cd3ae`. Carries the accumulated local,
  uncommitted `tracing::debug!` instrumentation from `06-f`/`06-i`'s sessions; this report neither adds
  to nor removes it, and it is not part of this commit.
- This report was written and committed from `/tmp/ff-artifacts-wt`, branch `ff-demo-artifacts`,
  verified independent of `/workspace`.

Not proceeding further this session, per instruction.
