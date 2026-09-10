# 06-J -- static trace of the vendored PagedAttention CUDA kernel and its Rust FFI wrapper (2026-09-10, fourteenth session)

Status: **static source trace only, as instructed. No source modified, no build, no benchmarks.**
Reads the vendored CUDA kernel source `06-e`/`06-h` flagged as the last unread boundary, plus its
Rust FFI wrapper. Finds no per-row or cross-call special-casing anywhere in this code that could
differentially corrupt a widened-to-ordinary transition. This further narrows, without closing, the
investigation -- see Sec 5.

Builds directly on `06-i`'s confirmed result: the widened step writes the replayed splice token
(absolute index 32, `5917`) to physical slot 128, and the immediately following step independently
resolves the same index to slot 128 again. This report asks whether anything in the kernel or its
invocation path could still corrupt that transition despite the address match.

## 1. The KV write kernel: `reshape_and_cache` (`mistralrs-paged-attn/src/cuda/reshape_and_cache_kernel.cu:32-84`)

One CUDA thread block per token (`token_idx = blockIdx.x`), grid size `dim3 grid(num_tokens)`
(`backend/paged_attention.rs`'s `update_cache`, called with `num_tokens = query_len` -- `2` for the
widened step, `1` for the step after). Each block independently reads `slot_mapping[token_idx]`
(exactly the array `06-i` confirmed correct) and writes to `key_cache`/`value_cache` at the offset
that slot implies. There is no shared state, synchronization, or data dependency between blocks for
different `token_idx` values -- each writes disjoint memory determined solely by its own slot. Nothing
here treats "two tokens written in one launch because they came from one widened FF step" any
differently from "two tokens written in one launch because two unrelated sequences share a decode
batch." The kernel has no concept of which case it is in.

## 2. The attention read kernel: `paged_attention_v1_kernel`/`paged_attention_v2_kernel` (`mistralrs-paged-attn/src/cuda/pagedattention.cuh`)

`seq_idx = blockIdx.y` (line 134); `context_len = context_lens[seq_idx]` (line 138);
`block_table = block_tables + seq_idx * max_num_blocks_per_seq` (line 229, reused identically at line
382 for the value pass). Every subsequent index derived from these (block index, block offset, mask
against `context_len`) is scoped entirely to that one `seq_idx`'s own row. This matches `06-e`'s prior
citation (`run_decode` flattens `[batch, seq_len]` into independent rows before dispatch) but this
report reads the kernel itself, not just the Rust wrapper's contract for it: there is no code path in
either the V1 or V2 kernel that references another row's `context_len`, `block_table` entry, or
intermediate state. A kernel launch with `num_seqs=2` (the widened step) executes two fully
independent per-row attention computations; a launch with `num_seqs=1` (the step after) is not a
special case of the same kernel, it is the same generic code with `gridDim.y=1`. **The kernel has no
mechanism by which having just processed a width-2 batch could leave any row-specific residue for a
subsequent width-1 launch to pick up** -- there is no persistent state in this kernel at all; every
launch is a fresh, self-contained computation over its own inputs.

## 3. Which kernel actually runs for this repro: V1, not V2

`backend/paged_attention.rs:302-307`: `max_num_partitions = effective_max_context_len.div_ceil(512)`;
`use_v1 = max_num_partitions == 1 || num_seqs * num_heads > 512`. This repro's context lengths are
~30-90 tokens, far under the 512-token partition size, so `max_num_partitions == 1` and `use_v1` is
true for every single decode step in this repro (both flag settings, every position). This rules the
V2 "partitioned reduction" kernel path (`paged_attention_v2_kernel` +
`paged_attention_v2_reduce_kernel`, which use extra `tmp_out`/`exp_sums` cross-kernel workspace
tensors) out of consideration entirely for this specific bug -- it is simply never exercised here, so
any hypothetical size-dependent workspace-reuse issue in that path is moot for this reproduction.

## 4. Output buffer allocation and stream usage (Rust wrapper, `backend/paged_attention.rs`)

`out = unsafe { dev.alloc::<T>(elem_count) }` (line 310) allocates a **fresh** output tensor sized by
`out_shape.elem_count()` for *this* call's actual `num_seqs`/`num_heads`/`head_size` -- 2 rows' worth
for the widened step, 1 row's worth for the step after. This is standard Candle custom-op output
allocation, not a persistent or pre-sized buffer reused across calls with stale dimensions. Both the
V1 attention kernel and `reshape_and_cache` are launched via `dev.cuda_stream().cu_stream()`
(`paged_attention.rs:345, 674`), where `dev` is the CUDA device associated with the input tensors --
the same device object persists for the whole model's lifetime, so (absent some other part of the
pipeline opening a second stream on this device, which this report did not go looking for outside this
file) every kernel launch across the entire generation is enqueued on one stream, and CUDA's per-stream
FIFO ordering guarantees the widened step's `reshape_and_cache` write completes before the next step's
`paged_attention` read is even issued, let alone before it runs.

`k.device_ptr(k.stream())`/`q.device_ptr(q.stream())` etc. (lines 312-315, 656-658) use each tensor's
*own* associated stream to obtain its device pointer -- a Candle idiom for making sure a tensor's own
prior async work has landed before its pointer is read, not evidence of a second, independently-clocked
stream for kernel dispatch. Whether every tensor's own `.stream()` is always identical to `dev
.cuda_stream()` in this codebase is a fact about Candle's own CUDA device/stream management, which is a
dependency's internals, not this repo's source -- this report did not open Candle's source to verify
it, since that would be the "broad audit" the task explicitly excluded.

## 5. Conclusion

**No kernel-indexing defect found.** Both CUDA kernels involved (`reshape_and_cache`,
`paged_attention_v1_kernel`) are fully generic per-token/per-row, carry no cross-call state, and have
no code path that could behave differently depending on whether an adjacent row belongs to the same
widened FF step or an unrelated batch. Combined with `06-i`'s confirmed slot-address match, this rules
out "kernel indexing" as the likely cause, not merely leaves it untested.

**Stream/synchronization ordering is not obviously broken either**, as far as this file's code shows:
one consistent device-level stream is used for every kernel launch in this call chain, which gives
launch-order-equals-execution-order for free under ordinary CUDA semantics. This report cannot rule out
a subtler issue in Candle's own stream/allocator management underneath `dev.alloc`/`.device_ptr(...)`,
since that is a dependency's internals outside this repo's source and outside the "vendored kernel"
scope this session was asked to read.

This means **every code-level layer this investigation has been able to read -- Rust engine
bookkeeping (`06-c`/`06-e`/`06-g`), Rust scheduler/block-allocation arithmetic (`06-h`, confirmed at
runtime by `06-i`), and now the vendored CUDA kernel source itself -- shows no defect.** The
investigation has reached the limit of what static source reading, in this repository, can settle.

## 6. Distinguishing what's left: kernel indexing vs. synchronization vs. actual memory content

Per the task's framing, the two candidates to distinguish were kernel indexing and synchronization.
This report's finding is that **kernel indexing is ruled out** (Sec 2-3: fully generic, no per-call or
cross-row state to get wrong) and **synchronization shows no defect in this repo's code** (Sec 4: one
consistent stream, standard ordering guarantees) but cannot be fully closed without reading a
dependency's internals, which was out of scope.

There is a third possibility neither "indexing" nor "synchronization" quite names, which `06-i`'s own
Sec 4 scope note already flagged: **the write completing correctly at the right address but with wrong
or incomplete content** -- e.g. a numerical error specific to processing two new KV rows in one
`reshape_and_cache`/attention launch pair (as opposed to the far more common one-row case), which
would not show up as an indexing or ordering bug at all, only as wrong *values* at a provably correct
*address*. Nothing in this session's reading rules this in or out; it is simply not visible to static
analysis, since it requires an actual numeric value, not a code path, to be wrong.

## 7. Single smallest runtime observation left to distinguish this

Since address-level correctness is now confirmed (`06-i`) and no code-level indexing or gross
synchronization defect was found (this report), the one remaining runtime check that would actually
move this forward is a **content**, not address, check: read back the raw key/value cache content at
slot 128 (the confirmed-correct physical slot for `5917`, index 32) immediately after the widened
step's `reshape_and_cache` write, and again immediately before the following step's `paged_attention`
read consumes it. If the two reads differ, the defect is a genuine data-correctness bug in the widened
step's write (most likely candidate remaining, per Sec 6) or a real cross-stream race this report could
not see from the Rust wrapper alone. If they are identical, the investigation has exhausted every
mechanism reachable from `mistralrs-core`/`mistralrs-paged-attn`'s own source, and the next place to
look would be Candle's CUDA backend internals -- a dependency, and a materially different, larger scope
than this investigation has covered so far.

This was not run this session, per instruction to do static analysis only.

## Files added

- `plans/ff-round-two/reports/06-j-cuda-kernel-trace.md` (this report). No other files changed; no
  source code touched, built, or benchmarked.

## Git/worktree state

- `/workspace`: branch `grammar-fast-forward`, HEAD `cab8cd3ae`. Carries the accumulated local,
  uncommitted `tracing::debug!` instrumentation from `06-f` and `06-i`'s sessions; this report neither
  adds to nor removes it, and it is not part of this commit.
- This report was written and committed from `/tmp/ff-artifacts-wt`, branch `ff-demo-artifacts`,
  verified independent of `/workspace`.

Not proceeding further this session, per instruction.
