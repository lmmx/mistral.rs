# 06-I -- runtime confirmation: index 32's physical KV slot matches at write and read time (2026-09-10, thirteenth session)

Status: **runtime confirmation of `06-h`'s static argument.** No source modified beyond the two
`tracing::debug!` lines added this session (described below, not committed to `grammar-fast-forward`
or this branch -- same convention as `06-f`'s instrumentation). No benchmarks. One live FF-ON run,
same reproduction as every prior report in this series.

## 1. Instrumentation added (local, uncommitted, `grammar-fast-forward` working tree only)

Two `tracing::debug!` lines added to `pipeline/inputs_processor.rs`'s `make_completion_chunk`, inside
the existing `if let Some(paged_attn_metadata) = &mut paged_attn_metadata` block, after the per-row
`slot_mapping`/`full_context_len` loops:

```rust
if narrow_for_ff {
    tracing::debug!(
        block_start,
        write_slots = ?slot_mappings.last(),
        "ff_trace: widened-step KV write slots"
    );
}
if block_start > 0 {
    let check_pos = block_start - 1;
    if let Some(&block_number) = table.get(check_pos / paged_attn_metadata.block_size) {
        let check_slot = block_number * paged_attn_metadata.block_size
            + check_pos % paged_attn_metadata.block_size;
        tracing::debug!(check_pos, check_slot, "ff_trace: predecessor slot resolved");
    }
}
```

No new state or cross-step tracking: each step independently resolves the slot for the position
immediately before its own window (`block_start - 1`) using its own freshly-fetched block table
snapshot. The first block additionally logs the widened step's own write-time slots
(`narrow_for_ff`-gated, so it fires exactly once per splice).

## 2. Reproduction and result (unchanged reproduction, one FF-ON run only -- FF-OFF never widens, so
there is nothing to check there)

Same request as every report in this series: `unsloth/Qwen3.5-4B-GGUF`, `nested.schema.json`,
`temperature=0`, `seed=42`, `max_tokens=64`. `RUST_LOG=warn,mistralrs_core::pipeline=debug`,
`MISTRALRS_GRAMMAR_FAST_FORWARD=1`.

Exact log lines around the transition (verbatim):
```
ff_trace: splice staged splice_tokens=[5917]
ff_trace: sample_sequence result logical_position=31 sampled_token=328
ff_trace: widened decode window built query_len=2 effective_context_len=33 window_tokens=[328, 5917] logits_span=Some((1, 1))
ff_trace: widened-step KV write slots block_start=31 write_slots=Some([127, 128])
ff_trace: predecessor slot resolved check_pos=30 check_slot=126
ff_trace: LogitsSelection::Decode selected start=1 len=1 seq_len=2
ff_trace: sample_sequence result logical_position=33 sampled_token=763
ff_trace: predecessor slot resolved check_pos=32 check_slot=128
ff_trace: sample_sequence result logical_position=34 sampled_token=220
ff_trace: predecessor slot resolved check_pos=33 check_slot=129
```

`write_slots=Some([127, 128])` is the widened step's own `slot_mapping`, covering `block_pos` 31 and
32 in order: index 31 (`328`, the backlog token) writes to slot `127`, **index 32 (`5917`, the
replayed splice token) writes to slot `128`**. The very next window-build (confirmed by log ordering
and timestamps to belong to the iteration immediately following the widened step's own sample,
`logical_position=33` -> `763`, and immediately preceding the first wrong token's sample,
`logical_position=34` -> `220`) independently resolves `check_pos=32` -- the same absolute index --
against its own freshly-fetched block table, and gets `check_slot=128`.

**The two values match exactly: `128 == 128`.**

## 3. What this establishes

`06-h`'s static argument -- that `kv_cache_manager`'s block table is append-only, so the same absolute
index must resolve to the same physical block/slot whether looked up at write time (the widened step)
or at read time (the immediately following step) -- is now confirmed by direct runtime observation,
not just by reading the code. The Rust-level block-table indexing for the specific token whose KV the
widened step wrote (index 32, `5917`) is not the defect: it is written to slot 128 and later resolved
to slot 128, consistently.

This closes `06-h`'s Sec 5 "single smallest runtime value" question with a definitive answer: **they
match.** Per `06-h`'s own stated branching logic, this means the defect -- if it is in the KV
write/read path at all -- is not a Rust-level block-table indexing bug, and is either in the raw CUDA
kernel's actual memory access behavior for a multi-row single-launch decode step, or in
stream/synchronization ordering between the widened step's write and the following step's read, ahead
of the Rust-level accounting this investigation can observe.

## 4. Scope note

This confirms index 32's (the ff-token's) slot identity only. It does not check index 33's slot
(`763`, the token the widened step itself produces and the following step, iter C, writes for the
first time) against anything, since there is no earlier write to compare it to -- `763` is written for
the first time in the very step (`logical_position=34`'s iteration) that first goes wrong. Nor does it
observe anything about the *content* of what is physically stored at slot 128 or read back from it --
only that the same integer slot index is computed at both points. A defect in the actual write (e.g.
if the widened step's multi-row kernel launch physically wrote the wrong content to the correct slot,
or failed to complete the write before it was read) would produce this exact same log signature and
is not distinguished by this check.

## 5. Next step

Per the user's explicit direction: proceed to a static read of the vendored PagedAttention CUDA
kernel source (`mistralrs-paged-attn/src/cuda/pagedattention_v{1,2}_*.cu`,
`mistralrs-paged-attn/src/cuda/attention/*.cuh`), which prior reports in this series (`06-e` Sec 8,
`06-h` Sec 4) had flagged as unread and, in `06-e`'s case, uncertain whether it was even vendored
in-repo. It is -- confirmed present in this checkout this session. That read is reported separately.

## Files added

- `plans/ff-round-two/reports/06-i-slot-match-confirmed.md` (this report). No other files changed.

## Git/worktree state

- `/workspace`: branch `grammar-fast-forward`, HEAD `cab8cd3ae`. Carries this session's local,
  uncommitted `tracing::debug!` additions (Sec 1) on top of `06-f`'s prior local instrumentation;
  neither is part of this commit or this branch.
- This report was written and committed from `/tmp/ff-artifacts-wt`, branch `ff-demo-artifacts`,
  verified independent of `/workspace`.

Not proceeding to a fix. Continuing to the CUDA kernel read as a separate, subsequent step per
explicit instruction.
