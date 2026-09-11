# 06-S -- conv_state equivalence proven by worked example; a real but likely-insufficient
fmaf/accumulation-order gap found in the width4-tiled *output* path (2026-09-11, twenty-third session)

Status: **investigation finding, not a fix.** Per explicit instruction, no source was modified, no
instrumentation was added, and no live reproduction was run. This continues `06-r`'s remaining
hypothesis -- whether `causal_conv1d_full` (used by the widened grammar-FF step, via the branch's
relaxed dispatch) and `causal_conv1d_update` (used by ordinary decode, whether eager or
CUDA-graph-replayed) produce/consume genuinely equivalent `conv_state` at the widened -> ordinary
boundary -- with a worked numeric trace rather than structural pattern-matching.

## 1. What the two kernels are supposed to produce

Both maintain, per physical pool slot, a `[conv_dim, kernel_size]` circular window holding the most
recent `kernel_size` token values per channel in **oldest-to-newest order** (index `kernel_size-1` =
most recent), and produce `SiLU(depthwise_causal_conv(window))` as the per-token output. `conv_state`
itself carries no semantics beyond "last `kernel_size` raw token values" -- it is consumed only as raw
data, never interpreted as anything computed.

## 2. Are the state conventions actually equivalent? Yes -- proven by worked example

`06-r` established structurally that both families use the identical `gdn_state_row(slot_indices, b, 0,
1) = slot_indices[b]` addressing and the identical oldest-to-newest index convention. This session
walked the arithmetic with concrete values to remove any doubt about a hidden off-by-one:

Setup: `kernel_size=4`, prior state `[p0, p1, p2, p3]` (p3 = most recent token from the last decode
step before the widened window), widened window `seq_len=2` with new tokens `x0, x1`.

**State write** (`save_conv_state_kernel`, `gdn.cu:1501`): `pad = kernel_size - seq_len = 2`. For `i <
pad`: `cs[i] = prior[i + seq_len]` -> `cs[0]=prior[2]=p2`, `cs[1]=prior[3]=p3`. For `i >= pad`: `pos =
seq_len - kernel_size + i` -> `cs[2]=x[0]=x0`, `cs[3]=x[1]=x1`. Result: `[p2, p3, x0, x1]` -- exactly
the last 4 tokens ending at `x1`, correctly ordered.

**Output windows** (`causal_conv1d_full_kernel`, `gdn.cu:1391`): for output position `pos=0` (`x0`),
`src_pos` ranges `-3..0`, yielding window `[p1, p2, p3, x0]`. For `pos=1` (`x1`), `src_pos` ranges
`-2..1`, yielding window `[p2, p3, x0, x1]`.

**Sequential comparison**: a first `causal_conv1d_update` call on `x0` shifts `[p0,p1,p2,p3]` to
`[p1,p2,p3,x0]` and computes output from that exact window. A second call on `x1` shifts to
`[p2,p3,x0,x1]` and computes output from that. **Both windows, at both positions, and the final state,
are identical, term-for-term, to what `causal_conv1d_full` computes in one shot.** This holds for
`seq_len < kernel_size`, `seq_len == kernel_size`, and `seq_len > kernel_size` (the last case verified
separately: `pad` goes negative, the `if (i < pad)` branch is never taken, and the window is filled
entirely from `x`, which is also what two/more sequential update calls would converge to).

**Conclusion:** the state layout and indexing conventions are provably equivalent, not just
structurally similar. `save_conv_state_kernel`'s write path involves **no arithmetic at all** -- every
element is a direct copy of either a prior-state slot or a new-token slot, in both the generic and
width4-tiled dispatch (both call the same non-specialized `save_conv_state_kernel` for the state write;
only the *output* kernel differs by width). A pure-copy write cannot introduce a value-level bug beyond
an indexing error, and no indexing error was found. Barring a bug in the *inputs* to this copy (see Sec
4), **`conv_state` itself should be bit-for-bit identical** whether produced by `causal_conv1d_full` in
one call or by an equivalent sequence of `causal_conv1d_update` calls.

## 3. What was found instead: a real floating-point ordering gap in the width4-tiled *output* path

This is not the state-equivalence bug that was hypothesized, but it is a genuine, source-verified
discrepancy worth recording. `GDN_PACKED_CONV_WIDTH = 4` (`gdn.cu:41`), and Qwen3.5/Qwen3-Next's GDN
`linear_conv_kernel_dim` is `4` in every fixture and default in this codebase (`gguf/normal_config.rs
:3334,3393`, `qwen3_5_moe/text.rs:998`) -- so the width4-specialized kernels, not the generic ones, are
what this model actually exercises.

- `causal_conv1d_update_width4_kernel` -> `gdn_conv_width4_update` (`gdn.cu:1293-1312`): plain
  accumulation, `acc = 0; for i in 0..4: acc += values[i] * weights[i]` (left-to-right, separately
  rounded multiply then add, matching the generic kernel's style exactly).
- `causal_conv1d_full_width4_tiled_kernel` (`gdn.cu:1449-1499`), used by the widened step whenever
  `kernel_size == 4 && x_stride_c == 1` (`gdn.cu:1552-1553`, likely true for a freshly-computed
  contiguous `mixed_qkv` tensor, though the exact stride was not traced upstream this session): `acc =
  __fmaf_rn(x0, w0, __fmaf_rn(x1, w1, __fmaf_rn(x2, w2, x3 * w3)))` -- a **right-to-left**, **fused**
  multiply-add chain (one rounding per term instead of two, and a different association order than the
  plain sum).

`fmaf(a, b, c)` is not required to equal `c + a*b` at the bit level in IEEE 754 -- fewer roundings and a
different reduction order can change the last bit(s) of the result. This means the two kernels can
produce a **different `y` (the conv1d output feeding SiLU/gating downstream)** for the same logical
window, even though the window contents (and therefore `conv_state`) are identical. This is a real,
provable, structural asymmetry between the two paths, not FP noise from unrelated causes.

## 4. What remains unproven

- **Whether this output-level fmaf/ordering gap is large enough to explain the reported symptom.** The
  reproduction is described as *consistently* degenerate at the same logical position across fresh
  restarts at temperature 0 -- the signature of a structural/logical divergence, not of a value sitting
  near an argmax decision boundary that a few ULPs of rounding difference could flip. An epsilon-scale
  fmaf/rounding difference is real but is the least likely single explanation for a *consistent,
  deterministic, always-wrong* failure at exactly the same point every run; it has not been ruled out as
  a contributing factor, but should not be treated as the primary explanation without runtime evidence.
- **Whether `x_stride_c == 1` actually holds for the widened step's `mixed_qkv` tensor in the live
  repro**, i.e. whether the width4-tiled kernel (with the fmaf gap) is even the path taken, or whether
  it silently falls back to the generic kernel (which has no such gap, per Sec 3). Not traced upstream
  this session; would need either a source trace of how `mixed_qkv` is constructed for the widened
  window, or a runtime stride dump.
- **Whether the raw `mixed_qkv` (`x`) values fed into `causal_conv1d_full` during a real widened
  multi-token forward are bit-identical, upstream of conv1d, to what per-token sequential computation
  would produce** -- a batched-vs-sequential matmul floating-point question that is unverifiable from
  source and is not specific to conv1d. This class of divergence already exists at every ordinary
  prefill -> decode boundary in unmodified, working code, so by itself it is unlikely to be
  FF-specific, but it was not checked this session and conv_state's bit-identity claim in Sec 2 is
  conditional on it.
- **Whether the static equivalence proof in Sec 2 actually holds on the live GPU.** This is a paper
  proof from reading kernel source; it has not been checked against actual device memory.

## 5. Exact minimal runtime measurement needed next

The state-write path itself needs no further static work -- Sec 2's proof is as complete as source
reading allows. What is needed is a live comparison, and there is a genuinely minimal way to get half of
it for free:

1. **`MISTRALRS_CUDA_GRAPHS=0`** (`perf_flags.rs:3`, existing env toggle, no source change) disables
   `cuda_decode_graphs_enabled()`, so `try_cuda_decode_graph_forward` returns `None` unconditionally and
   *every* decode step -- including the first ordinary step right after a widened step -- falls back to
   eager `self.model.forward(...)`, i.e. `forward_embeds`. That step becomes observable to the **already
   -existing** `ff_trace_gdn_state_checksum` diagnostic (`text.rs`, uncommitted from `06-q`), which
   already gathers `conv_state` (not just `recurrent_state`) at both `forward_entry` and `forward_exit`
   of every `forward_embeds` call. **No new instrumentation is required for the "after widened step" and
   "start of next ordinary step" sides of the comparison** -- only this env var, on one repro run.
2. Upgrade the comparison granularity: `conv_sum`/`conv_sumsq` are permutation-invariant reductions --
   they cannot distinguish "correct 4 values in the correct order" from "the same 4 values in a
   different order." Given the state-write path is proven index-correct in Sec 2, this risk is low, but
   for a conclusive runtime check the diagnostic should dump the raw `[conv_dim, kernel_size]` values (or
   at minimum a per-position sum across `conv_dim`, i.e. one scalar per of the 4 window slots instead of
   one collapsed scalar) rather than a single sum/sumsq pair.
3. Run this once with grammar FF ON (`MISTRALRS_CUDA_GRAPHS=0`) and compare the logged `conv_state` at
   the boundary against the same position from an FF-OFF sequential-only run of the same prompt. This
   directly tests Sec 2's prediction against real values, including whatever upstream `mixed_qkv`
   differences Sec 4 flags as unverified.
4. **This run doubles as a bug-localization test independent of the conv_state comparison**: if the
   degenerate output *disappears* with `MISTRALRS_CUDA_GRAPHS=0` (all other reproduction conditions
   unchanged), that is strong evidence the CUDA-graph replay mechanism itself (not the conv1d state
   equivalence traced in this report) is implicated, since the only thing this toggle changes is
   removing the graph-replay path in favor of eager `forward_embeds`. If the degenerate output
   *persists* with graphs off, that rules out CUDA-graph replay entirely and localizes the bug to the
   eager `causal_conv1d_full`/`causal_conv1d_update` handoff (or something further upstream/downstream of
   conv1d) regardless of graph involvement -- in which case the width4-tiled fmaf gap from Sec 3 becomes
   a more serious candidate worth measuring directly (dump `y`, not just `conv_state`, at the same
   boundary).
5. If Sec 2's prediction holds (conv_state matches bit-for-bit) but the decoded token still diverges with
   graphs off, the investigation should move to `recurrent_state` and the delta-rule kernels
   (`gated_delta_rule_recurrence_kernel_vmajor_grouped`, flagged in `06-p` as audited for in-place-aliasing
   safety but not for this specific full-then-update cross-path handoff) rather than conv1d.

No runtime measurement was performed this session; this is the exact next step, not yet executed.

## 6. Repository state at end of session

- `/workspace`, branch `grammar-fast-forward`: dirty, one file changed
  (`mistralrs-core/src/vision_models/qwen3_5/text.rs`, the `06-q` checksum diagnostic), left uncommitted
  and unmodified this session. No source touched.
- This report was written and committed from `/tmp/ff-artifacts-wt`, branch `ff-demo-artifacts`.

## Files added

- `plans/ff-round-two/reports/06-s-conv1d-state-equivalence-proven-output-fmaf-gap-found.md` (this
  report). No other files changed on this branch this session.
