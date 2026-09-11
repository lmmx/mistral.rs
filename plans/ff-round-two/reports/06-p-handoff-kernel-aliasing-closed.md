# 06-P -- session handoff: kernel-aliasing hypothesis closed, recommended next step is runtime state comparison (2026-09-10/11, twentieth session)

Status: **handoff report, written at the user's explicit request to pause for the night.** Records
this session's kernel-level findings in full, the cumulative list of mechanisms this investigation
has now ruled out, the one fact that remains unexplained, and the user's own recommendation for the
next session (a runtime state comparison, not another static audit). No source modified this session
beyond what was already committed before this report was written (see Sec 6). No build, no run.

## 1. What this session did: read the actual multi-token recurrence kernel

Following `06-n`/`06-o` (flush fix implemented) and the runtime trace that falsified the flush
hypothesis (the flush demonstrably runs -- `pending_ff_width=Some(1)`, `flush_recurrent_speculative_
transitions` called, `flushed=true` -- and the bug is completely unchanged: position 33 still `763`,
position 34 still `220`, same `220/220/201` cycle), the user redirected the investigation to the
GDN kernels themselves: does the widened (`seq_len=2`) forward pass leave `recurrent_state`/`conv_state`
correctly represented for the very next ordinary (`seq_len=1`) forward pass to consume?

Two mechanisms were traced this session, both at the actual CUDA source level (not just the Rust
wrapper), both found **safe**:

### 1a. Conv1d state save kernel (`save_conv_state_kernel`, `cuda/gdn.cu:1502-1535`)

For the pooled case (Qwen3.5's paged GDN state), `causal_conv1d_cuda`'s multi-token path
(`cuda/gdn.rs:1727-1788`) aliases `conv_state_out` to the same pointer as `conv_state_in` -- true
in-place read/write, no separate output buffer. The kernel's own comment claims *"every read is ahead
of the write position."* This was verified, not just trusted, by working through the index arithmetic:
```c
int pad = kernel_size - seq_len;
for (int i = 0; i < kernel_size; i++) {
    if (i < pad) { cs[i] = prior[i + seq_len]; }   // read index i+seq_len
    else         { cs[i] = x[...]; }                // write index i
}
```
For `i < pad`, the read index `i + seq_len` always falls in `[pad, kernel_size)` -- exactly the range
written by the `else` branch at loop iterations `i' = i + seq_len > i`. Since the loop runs in strict
index order within one thread, every read at step `i` targets a slot only overwritten at a *later*
step `i'`. Holds for any `seq_len >= 1`. **Verified safe.**

### 1b. Recurrence kernel (`gated_delta_rule_recurrence_kernel_vmajor_grouped`, `cuda/gdn.cu:731-836`)

This is the kernel actually selected for our repro: `automatic_prefill_kernel`
(`cuda/gdn.rs:323-331`) picks `ValueMajorWarp{2,4,8}` (all three instantiate the same template) based
purely on `state_blocks` (batch x heads) vs. the GPU's SM count -- for our N=1 sequence, `state_blocks`
is essentially certain to be `<=` the SM count, giving `ValueMajor2`.

Traced against the user's four specific questions:

1. **Reads and writes the same state location during a multi-token sequence?** No. `state_bh` is read
   exactly once, at kernel entry, into per-thread registers (`float s[VALUES_PER_WARP][ROWS_PER_LANE]`,
   line ~777-785), and written exactly once, after the entire timestep loop completes (line ~828-835).
   Nothing touches `state_bh` in between.
2. **Assumes input/output state are distinct?** No -- the opposite: it is built to be alias-safe by
   construction, since there is only one read (before any write) and one write (after all reads).
3. **Processes timesteps strictly sequentially?** Yes -- an explicit `for (int t = 0; t < seq_len; t++)`
   loop (line 787), entirely register-resident; `s[value][row]` updates in-register at every timestep
   before moving to `t+1`.
4. **Any parallelism that makes aliased updates unsafe?** No. Across warps: each warp owns a disjoint
   slice of the V=128 dimension (`first_value = value_group * VALUES_PER_WARP`), so different warps
   never touch each other's state rows. Within a warp: the 32 lanes cooperate via `gdn_warp_sum`
   (`cuda/gdn.cu:537-543`), which uses `__shfl_down_sync`/`__shfl_sync` with a full `0xffffffff` mask --
   the standard synchronized warp-shuffle reduction, not an unsynchronized pattern.

**Verified safe**, same conclusion as the conv-state kernel.

## 2. Cumulative list of mechanisms this investigation has ruled out

As of this session, restated in the user's own words at the pause point:

- **conv state**: safe in-place multi-token update (Sec 1a, this session)
- **recurrent state**: safe in-place multi-token update (Sec 1b, this session)
- **GDN flush**: definitely executes before the widened step (`06-n`/`06-o`'s fix, confirmed live in
  the runtime trace pasted into this session: `pending_ff_width=Some(1)` -> `flush_recurrent_
  speculative_transitions` called -> `flushed=true`, all before the widened window is built)
- **KV slots**: widened write and next-step read agree (`06-i`, confirmed at runtime: index 32 writes
  to slot 128, the following step's independent lookup resolves the same index to slot 128)
- **logits selection**: selects the widened step's final row (`06-e`, `LogitsSelection::Decode`
  resolves to the last of the two window rows, matching the position needed to predict the token after
  the splice)
- **CUDA graphs**: not involved for this grammar-constrained sequence (`06-g`, both the resident-
  sampling and forward-pass graph-replay paths are gated on `SequenceRecognizer::None`, excluding any
  grammar-constrained sequence uniformly, on both FF-OFF and FF-ON, at every step)
- **llguidance**: the FF token is semantically forced the same way in ordinary decoding (`06-d`, the
  forcing mechanism and ordinary per-token masking share the identical `ff_tokens()` computation, so
  content is provably equivalent between FF-ON's splice and FF-OFF's masked sample at the same grammar
  position)

## 3. The fact that remains unexplained

**Position 33 (the widened step's own real sample) is correct, matching FF-OFF exactly. Position 34
(the very next, ordinary width-1 decode step) is the first wrong token, after which generation is
permanently locked into a `220/220/201` repeating cycle.** No mechanism traced so far -- across ten-plus
sessions spanning KV/PagedAttention accounting, block/slot allocation, the vendored attention kernel,
llguidance's forcing semantics, and now both pieces of GDN state -- explains this signature.

## 4. Recommended next step (not run this session): runtime state comparison

The user's explicit reasoning and recommendation, recorded here for continuity:

> That points much more strongly toward a runtime value comparison at the widened->ordinary boundary
> than another structural source audit... Compare the model state entering position 34 between FF-ON
> and FF-OFF. Ideally, don't dump giant tensors. Add a tiny diagnostic at the boundary for the affected
> sequence, for example: recurrent-state checksum / a handful of deterministic reductions, conv-state
> checksum, perhaps the relevant GDN layer's state -- immediately after the widened step, and
> immediately before the next ordinary step. Then run the exact N=1 reproduction ON/OFF.

Interpretation table for that comparison, as given:

| Observation | Meaning |
|---|---|
| State differs immediately after widened step | widened forward changes persistent state incorrectly |
| State matches after widened step, differs entering ordinary step | something between pipeline steps mutates/uses it |
| State matches entering ordinary step, output differs | problem is downstream of persistent GDN state |
| State differs only in one GDN layer | the defect is substantially localized |

Explicit guidance: **start with one or a few cheap scalar checksums over `recurrent_state`** (not a
tensor dump), and this is a **runtime comparison, not another blind kernel read** -- the user was
explicit that static kernel auditing should stop here pending this experiment's result.

Practical note for whoever picks this up: `06-f`'s style of instrumentation (a `tracing::debug!` logging
a cheap reduction, e.g. `recurrent_state.sum_all()?.to_scalar::<f32>()?` or similar, computed only for
the affected sequence's pooled row) is the natural mechanism, mirroring every prior instrumentation
round in this investigation -- gate it tightly (only for the sequence with an active FF splice /
`narrow_for_ff`) so it stays cheap and doesn't run on every ordinary step.

## 5. Explicit non-recommendation

The user was explicit: **do not keep reading more CUDA kernels blindly.** The two most concrete,
checkable structural hypotheses (conv-state aliasing, recurrent-state aliasing) have now both been
verified safe at the actual kernel source level, not just inferred from comments or Rust wrappers. Any
further static reading without a fresh runtime signal to aim it at would be unfocused. The next
session should run the runtime checksum comparison in Sec 4 before considering any further source
reading.

## 6. Repository state at handoff

- `/workspace`, branch `grammar-fast-forward`: clean working tree (no uncommitted changes) as of this
  report. Recent commits, most recent first:
  - `30c68b149` -- `chore(ff_trace): log engine pending ff width/use` (the unconditional
    `pending_ff_width`/`use_pending_ff` diagnostic logging that produced the runtime trace pasted into
    this session, proving the flush fires correctly)
  - `0b2773ea2` -- `chore(ff_trace): flush log lines` (reachability logging around the flush call site
    and inside Qwen3.5's flush implementation)
  - `1adb03ea0` -- `chore: tracing implementation to debug FF splicing and KV management` (the original
    `06-f`-era instrumentation: splice-staged, widened-window-built, predecessor-slot logs)
  - The flush-before-widened-step fix itself (`06-n`/`06-o`'s recommendation, applied in an earlier
    session) is present in the working tree/history as of these commits -- it is confirmed live and
    firing correctly by the runtime trace pasted into this session, and confirmed **not** to change the
    bug's behavior.
  - Below these: `cab8cd3ae` (the original branch tip this whole investigation started from) and its
    ancestors, untouched.
  - No GDN kernel source (`cuda/gdn.cu`, `cuda/gdn.rs`, `gdn/backend.rs`, `gdn/layer.rs`) was modified
    at any point in this investigation -- every finding in Sec 1 came from reading, not editing.
- This report was written and committed from `/tmp/ff-artifacts-wt`, branch `ff-demo-artifacts`,
  verified independent of `/workspace`.

## Files added

- `plans/ff-round-two/reports/06-p-handoff-kernel-aliasing-closed.md` (this report). No other files
  changed; no source code touched, built, or run this session.

Pausing here at the user's request. Not proceeding to the runtime checksum comparison until the next
session explicitly begins it.
