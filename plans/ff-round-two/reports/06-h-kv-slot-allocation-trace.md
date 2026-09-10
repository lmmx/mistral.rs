# 06-H -- static trace of physical PagedAttention KV block/slot allocation across the widened-to-ordinary transition (2026-09-10, twelfth session)

Status: **static source trace only, as instructed. No source modified, no build, no benchmarks, no
live requests.** Traces the block-count allocation layer (`paged_attention/scheduler.rs`) and the
slot-index computation layer (`pipeline/inputs_processor.rs`) that `06-g` left unverified. Both check
out consistent by direct substitution of this repro's actual observed values. This exhausts the
Rust-level, source-derivable accounting; no defect was found at any layer this investigation has now
traced. One smallest runtime value is identified to take this further.

All absolute token indices below are 0-indexed into `seq.get_toks()`; they are one less than the
`logical_position` field `06-f`'s instrumentation logs (that field is `seq.get_toks().len()` measured
inside `sample_sequence`, i.e. the count *before* the sampled token is appended -- equivalently, the
0-indexed position the new token will occupy). Concretely, from `06-f`: token `328` is sampled with
`logical_position=31` (occupies index 31), the splice `5917` is replayed to occupy index 32, and the
real sample `763` is taken with `logical_position=33` (occupies index 33, the first token proven
correct). Index 34, occupied by whatever `logical_position=34` produces, is the first wrong token.

## 1. Which physical slot does index 33 (`763`) get written to?

Reproduces the exact call for the widened step (call it iter B, `06-f`'s `logical_position=31->33`
step). At the top of `paged_attention/scheduler.rs`'s per-step running-queue pass
(`schedule()`, lines 1185-1197), before this step's forward pass:

```rust
let num_tokens = if staged_speculative > 0 {
    seq_guard.len() + staged_speculative
} else if pending_ff > 0 {
    seq_guard.len() + pending_ff
} else if seq_guard.num_uncomputed_tokens() > 0 {
    seq_guard.len()
} else {
    seq_guard.len() + 1 // +1 for the new token to be generated
};
```

At this call, `seq_guard.len() = 32` (indices 0..31, i.e. through `328` at index 31 -- the splice
`5917` and the real sample `763` have not been appended yet), `pending_ff = 1` (the staged splice).
This takes the `pending_ff > 0` branch: `num_tokens = 32 + 1 = 33`. `allocate_slots(seq_id, 33, &[])`
(`kv_cache_manager.rs:280-362`) computes `num_required_blocks = 33.div_ceil(block_size)` and extends
`req.block_ids` (append-only, `kv_cache_manager.rs:302-308`) if the existing table is short of that.
This size **exactly covers indices 0..32** -- i.e., through the ff token at index 32, which is the
last index this step's forward pass needs a fresh slot for (`763` at index 33 is this step's *output*,
not something whose own KV this step writes -- that happens next step). No shortfall found: the
requested block count matches what the step's writes need.

`pipeline/inputs_processor.rs`'s `make_completion_chunk` (this container's local, uncommitted `06-f`
instrumentation patch shifts these line numbers by +9 from the untouched `cab8cd3ae` baseline; current
numbers cited) then builds this step's window: `start_pos = ctxt.len()(32) - decode_window(1) = 31`,
`ctxt = seq.get_toks()[31..]` (`[328]`) `.extend(pending_ff)` -> `[328, 5917]`, `query_len = 2`. The
slot loop (`inputs_processor.rs:1668-1689`, `for row in 0..query_len`) computes, for each absolute
position `block_start + row` (`31` and `32`):
```rust
let block_start = start_pos - seq.token_offset();   // 31 - 0 = 31
let block_end = block_start + query_len;             // 33
// per row: block_number = table[block_pos / block_size], block_offset = block_pos % block_size
// slot = block_number * block_size + block_offset
```
using `table = sequence_block_tables[seq_idx]`, a snapshot of `kv_mgr.get_block_ids(*seq.id())`
(`inputs_processor.rs:1574-1586`) taken once per this call -- i.e., reflecting the allocation this
same step just requested above. `seq.token_offset()` (`sequence.rs:1291-1293`) is a stored field, not
recomputed here; for this single-turn, no-prefix-reuse repro it is `0` throughout (not independently
re-verified this session, but nothing in this trace's call chain writes to it). Row 1 (`block_pos=32`)
resolves the physical slot `763`'s *predecessor* (`5917`) is written to -- this is the slot whose
correctness the next section depends on. `763` itself (index 33) is not written to any slot during
this step; it is produced as this step's sampled output and gets its own KV write next step.

## 2. Which slot does index 34's step (the first wrong token) subsequently read?

Iter C (producing `logical_position=34`) begins with a fresh `schedule()` pass. At this point
`seq_guard.len() = 34` (indices 0..33, now including `5917` at 32 and `763` at 33, both appended
during iter B's execution -- confirmed `add_token`, `sequence.rs:1630-1689`, never touches
`num_computed_tokens`, only `06-g` needed that fact; here it matters that it also never reorders
`seq.tokens`), `pending_ff = 0` (consumed by iter B's replay), `staged_speculative = 0|`, and
`seq_guard.num_uncomputed_tokens() = len(34) - num_computed_tokens(33) = 1 > 0` (per `06-g`'s
derivation that `num_computed_tokens` correctly advanced to `33` at the end of iter B). This takes the
**third** branch: `num_tokens = seq_guard.len() = 34`. `allocate_slots(seq_id, 34, &[])` requests
blocks covering indices 0..33 -- i.e., through `763` at index 33, the token this step's forward pass
needs a fresh slot to *write* into. Again, no shortfall: the requested size matches what this step's
one new write (index 33) needs.

`inputs_processor.rs` builds this step's window: `start_pos = ctxt.len()(34) - 1 = 33`, `ctxt =
seq.get_toks()[33..] = [763]`, `query_len = 1`. The slot loop resolves `block_pos = 33` (`763`'s own
new slot, to be written this step) against **the same `table`**, freshly re-fetched via
`kv_mgr.get_block_ids` for this call. Separately, this step's PagedAttention **context length** for
attention (`full_context_len = start_pos + row + 1 = 33 + 0 + 1 = 34`, `inputs_processor.rs:1694`)
tells the attention kernel to read back *all* cached keys/values for indices `0..33` inclusive when
computing this step's logits -- which includes index 32 (`5917`, written by iter B) and, depending on
the kernel's write-then-read ordering within its own launch (`06-e` Sec 5 already traced this specific
point: KV cache write happens before the attention read within one launch, `paged_attention.rs`'s
`run_decode`), index 33 (`763`, being written this same step).

Because `req.block_ids` is only ever **appended to**, never reordered or rewritten in place
(`kv_cache_manager.rs:288-308`; the only other mutators, `free` at line 368 and
`trim_request_to_num_tokens` at line 382, are not called anywhere on this sequence's live path between
iter B and iter C -- `free` only runs at sequence end, `trim_request_to_num_tokens` only for
speculative-decoding lookahead rollback, `active_staged_speculative_len() == 0` throughout this
repro), the block id that `table[32 / block_size]` resolves to in iter C is provably the **same**
array entry, at the same index, that iter B's own `table[32 / block_size]` resolved to when it wrote
`5917`'s KV. By construction, there is no code path in this chain that could make iter C's read of
index 32 land on a different physical block than iter B's write did.

## 3. Does the widened step change scheduler/block state differently from a normal width-1 step?

Yes, algorithmically -- it takes a different branch of the four-way `num_tokens` formula
(`pending_ff > 0` instead of the `num_uncomputed_tokens() > 0` or `+ 1` branches an ordinary step
takes) -- but substituting this repro's actual numbers into both branches shows each computes exactly
the token count its own step's writes require, with no shortfall or overshoot at either the widened
step or the step immediately following it. The three branches are not the same formula, but they are
not required to be: each is sized for what its own step is about to write, and `06-g`'s already-traced
`num_computed_tokens`/`num_uncomputed_tokens` state feeds correctly into whichever branch a given step
takes.

## 4. Conclusion

**No defect found.** This report traced the two remaining Rust-level layers `06-g` had not covered --
the scheduler's block-count sizing and the per-step slot-index computation -- against this repro's
actual concrete values (not just "the formula looks consistent," but the literal arithmetic: `32+1=33`
for iter B, `34` for iter C, and the append-only block-table argument for why index 32 resolves
identically in both steps). Combined with `06-e` (window/position/logits-row construction),
`06-f` (runtime confirmation that the widened step's own sample is correct), and `06-g` (CUDA
decode-graph exclusion, `num_computed_tokens` engine-side arithmetic), **every Rust-level accounting
mechanism this investigation can reach by reading source has now been traced for this exact
transition, and none show a divergence.**

This does not mean there is no bug in this vicinity -- it means the bug, if one exists in the KV
read/write path at all, is not visible as a token-count, block-count, or block-table-indexing error at
the Rust boundary. Two possibilities remain, neither reachable by further static reading of this
codebase's Rust sources:

1. **The raw CUDA/PTX kernel body** underneath `mistralrs-paged-attn`'s `paged_attention()` FFI call.
   `06-e` (Sec 8, item 1) already flagged this as unread by any session; this report's finding that
   every Rust-side formula and index is provably correct narrows the remaining suspicion onto this
   exact boundary -- specifically, whether the kernel's actual memory access for a **multi-row single
   launch** (iter B's widened step, two rows sharing one logical sequence's causal history within one
   kernel invocation) correctly writes both rows before any read that depends on them, as opposed to
   the far more common one-row-per-sequence case the kernel may have been primarily designed and
   tested against.
2. **A stream-ordering/completion race between kernel launches**, not visible in synchronous Rust
   source reading -- whether the GPU has actually finished writing index 32's (and 33's) KV entries
   before iter C's kernel launch reads them back, if writes and reads for consecutive steps are
   dispatched on a stream or queue whose ordering guarantees this trace's Rust-level reasoning assumes
   but cannot itself confirm.

## 5. Single smallest runtime value to instrument next

Given every Rust-level formula is now traced and self-consistent, the next step that would actually
discriminate between "the bug is below the Rust boundary" and "there is a Rust-level mistake this
trace still missed" is to observe, at runtime, the **one number this report predicts must match but
cannot itself execute**:

**The physical slot (or block id + block offset) assigned to absolute token index 32 (the ff token
`5917`), read once at the moment iter B writes it, and read again at the moment iter C's forward pass
resolves it for attention.** This report's Sec 2 argument is that these must be numerically identical
by construction (append-only block table, same index, same `block_size`). If a live run shows they
are *not* identical, the defect is a genuine Rust-level block-table indexing bug this trace missed
despite the argument above, and is worth a second static pass focused specifically on whatever
mutates `req.block_ids` between the two reads. If they *are* identical, the defect is conclusively
below the Rust boundary (candidates 1 or 2 above), and the investigation's next move is reading the
actual kernel source or a targeted CUDA-side check, not further Rust instrumentation.

This was not run this session, per instruction to do static analysis only.

## Files added

- `plans/ff-round-two/reports/06-h-kv-slot-allocation-trace.md` (this report). No other files
  changed; no source code touched, built, or benchmarked.

## Git/worktree state

- `/workspace`: branch `grammar-fast-forward`, HEAD `cab8cd3ae`. Still carries the `06-f` session's
  local, uncommitted `tracing::debug!` instrumentation (the source of the +9 line-number shift in
  `inputs_processor.rs` cited above); this report neither adds to nor removes it, and it is not part
  of this commit.
- This report was written and committed from `/tmp/ff-artifacts-wt`, branch `ff-demo-artifacts`,
  verified independent of `/workspace` (`git worktree list`, distinct device/inode, consistent with
  every prior session's check).

Not proceeding further this session, per instruction.
