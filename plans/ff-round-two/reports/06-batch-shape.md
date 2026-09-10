# 06 -- Batch-shape measurement (2026-09-10, fifth session)

Status: **run, with the ON/OFF contrast now genuine and one unresolved equality anomaly flagged.**
This supersedes `06-batch-shape-blocked.md` for the same session's follow-up, after the user
established manual sequential control of the host server's lifecycle (VRAM on the 24 GB host does
not fit two concurrent PagedAttention instances -- `Num GPU blocks is 0` -- so ON and OFF were run
as two consecutive processes, not two concurrent ones). No Rust source touched, no rebuild, no
change to `ff_harness.py` or the fixtures. `/workspace` stayed on `grammar-fast-forward` at
`cab8cd3ae` throughout; this report and its raw data were produced and committed from a separate
`git worktree` at `/tmp/ff-artifacts-wt` (branch `ff-demo-artifacts`), verified independent of
`/workspace` before use.

**Read this alongside its limitations section before citing any number here.** Dimension 3 (the
one instrumentation addition the plan permits) was not added -- out of scope for this session.
Dimension 4 (splice-width distribution / per-batch minimum) is only partially recoverable from the
existing counters. And correctness/equality did **not** cleanly hold across every run -- see
"Correctness / equality" below, which is the most important section in this report.

## Server configurations

Both processes: `unsloth/Qwen3.5-4B-GGUF`, quant 4, paged attention, `--max-batch-size 8
--max-seqs 8`, port 1234, `target/debug/mistralrs` built from `grammar-fast-forward` at `cab8cd3ae`
(the same binary the user had already built for the prior plan-05 checkpoint). Both started fresh
by the user for this session (confirmed via `/metrics` immediately after each restart -- near-zero
counters, exactly one `cuda_graph_dispatch_total{mode="skipped",reason="prefill"}` from the health
probe -- before this session sent any sweep traffic), so neither run inherited warm state from the
other.

**ON:**
```
MISTRALRS_GRAMMAR_FAST_FORWARD=1 target/debug/mistralrs serve \
  --model-id unsloth/Qwen3.5-4B-GGUF --quant 4 --paged-attn on \
  --max-batch-size 8 --max-seqs 8 --no-ui --port 1234 -v
```
Confirmed via `/metrics`: `mistralrs_grammar_ff_support_total{supported="true",reason="enabled"} 1`.

**OFF:**
```
target/debug/mistralrs serve \
  --model-id unsloth/Qwen3.5-4B-GGUF --quant 4 --paged-attn on \
  --max-batch-size 8 --max-seqs 8 --no-ui --port 1234 -v
```
(`MISTRALRS_GRAMMAR_FAST_FORWARD` unset.) Confirmed via `/metrics`:
`mistralrs_grammar_ff_support_total{supported="false",reason="flag_disabled"} 1`.

Before each sweep, one small unconstrained warmup request (`max_tokens=16`) was sent directly and
discarded, to absorb first-request CUDA-graph capture overhead symmetrically on both sides.

## Harness invocation

`ff_harness.py run --mode concurrency`, from this container, against the live host server, using
the same technique as `05-harness-live-validation-checkpoint.md`: a harmless placeholder
`--server-cmd "sleep 300"` (satisfies the harness's own `Popen`/`terminate` lifecycle without
touching the real server) with the real traffic routed via `--server-host host.docker.internal
--server-port 1234`.

Shared flags across all 24 runs:
```
python3 ff_harness.py run --mode concurrency \
  --model-id unsloth/Qwen3.5-4B-GGUF --flag {on|off} \
  --server-cmd "sleep 300" \
  --server-host host.docker.internal --server-port 1234 \
  --startup-timeout-seconds 5 --max-tokens 64 --plan 06 \
  --metric-prefix mistralrs_ --out-dir plans/ff-round-two/reports/06-sweep \
  --label <label> --num-requests <N> --schema-set <identical|differing> \
  --unconstrained-fraction <0.0|0.5> --seed 42 [--stagger-seconds 0.05]
```

Three workloads x four values of N, run once per flag setting (24 runs total, all within this one
session, ON batch first then OFF batch after the user's manual restart):

- **A -- identical schemas.** `--schema-set identical --unconstrained-fraction 0.0`. Fixture:
  `partial.schema.json` for every request.
- **B -- differing schemas.** `--schema-set differing --unconstrained-fraction 0.0`. Round-robins
  a seeded-shuffle order over `nested.schema.json`, `partial.schema.json`, `wide.schema.json`,
  `narrow.schema.json`.
- **C -- mixed.** `--schema-set identical --unconstrained-fraction 0.5` (the constrained half uses
  `partial.schema.json`; the plan does not specify a schema set for the constrained half of the
  mixed workload, so `identical` was chosen for simplicity -- noted as a scope decision, not a
  plan requirement).

`--stagger-seconds 0.05` was applied to the A/N=4 run (both ON and OFF) to satisfy the plan's
requirement that at least one variant not use synchronized starts; all other runs used simultaneous
submission (`ThreadPoolExecutor`, all N requests fired together).

All 24 runs: **0 failed requests** (`count_failed: 0` in every report), server stayed healthy
throughout both sweeps (`/health` returned 200 after the OFF sweep completed).

## Results table (all 24 runs)

| combo | flag | ok | wall_s | decode_tok | tok/s (decode_tok/wall_s) | staged | fed | splice_drops(batch_shape) | tokens_dropped(batch_shape) |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|
| A-identical-n1 | on  | 1 | 0.271 | 44  | 162.5 | 3  | 4 | 0  | 0  |
| A-identical-n1 | off | 1 | 0.249 | 44  | 176.4 | 0  | 0 | 0  | 0  |
| A-identical-n2 | on  | 2 | 0.330 | 78  | 236.5 | 7  | 2 | 6  | 7  |
| A-identical-n2 | off | 2 | 0.340 | 78  | 229.3 | 0  | 0 | 0  | 0  |
| A-identical-n4 | on  | 4 | 0.587 | 176 | 299.6 | 16 | 1 | 15 | 19 |
| A-identical-n4 | off | 4 | 0.585 | 176 | 301.1 | 0  | 0 | 0  | 0  |
| A-identical-n8 | on  | 8 | 0.846 | 352 | 416.0 | 32 | 0 | 32 | 40 |
| A-identical-n8 | off | 8 | 0.848 | 352 | 414.9 | 0  | 0 | 0  | 0  |
| B-differing-n1 | on  | 1 | 0.366 | 63  | 172.2 | 1  | 1 | 0  | 0  |
| B-differing-n1 | off | 1 | 0.256 | 46  | 179.9 | 0  | 0 | 0  | 0  |
| B-differing-n2 | on  | 2 | 0.370 | 90  | 243.1 | 13 | 0 | 13 | 23 |
| B-differing-n2 | off | 2 | 0.401 | 90  | 224.4 | 0  | 0 | 0  | 0  |
| B-differing-n4 | on  | 4 | 0.480 | 146 | 304.1 | 28 | 0 | 28 | 57 |
| B-differing-n4 | off | 4 | 0.478 | 146 | 305.5 | 0  | 0 | 0  | 0  |
| B-differing-n8 | on  | 8 | 0.891 | 326 | 365.9 | 40 | 0 | 40 | 80 |
| B-differing-n8 | off | 8 | 0.852 | 309 | 362.8 | 0  | 0 | 0  | 0  |
| C-mixed-n1 | on  | 1 | 0.242 | 44  | 181.6 | 3 | 4 | 0 | 0 |
| C-mixed-n1 | off | 1 | 0.250 | 44  | 175.7 | 0 | 0 | 0 | 0 |
| C-mixed-n2 | on  | 2 | 0.306 | 66  | 216.0 | 3 | 2 | 2 | 2 |
| C-mixed-n2 | off | 2 | 0.328 | 66  | 201.1 | 0 | 0 | 0 | 0 |
| C-mixed-n4 | on  | 4 | 0.423 | 132 | 312.1 | 6 | 4 | 4 | 4 |
| C-mixed-n4 | off | 4 | 0.429 | 122 | 284.7 | 0 | 0 | 0 | 0 |
| C-mixed-n8 | on  | 8 | 0.685 | 264 | 385.7 | 12 | 8 | 8 | 8 |
| C-mixed-n8 | off | 8 | 0.669 | 264 | 394.5 | 0 | 0 | 0 | 0 |

`splices_staged_total`, `tokens_fed_total`, `splice_drops_total{reason="batch_shape"}` and
`tokens_dropped_total{reason="batch_shape"}` were all exactly 0 for every OFF run and every
`mistralrs_grammar_ff_attempts_total` outcome under OFF was `unsupported` (flag disabled, gate A
fails deliberately -- this is the expected, structural OFF signature, not a bug). No drop reason
other than `"batch_shape"` was observed in any run, on either flag setting -- so `batch_shape` is
not being inflated by preemption or realloc reasons here; there simply weren't any.

## Dimension 1 -- splice discard rate by reason (ON only; OFF is structurally 0/0)

| combo | staged | dropped(batch_shape) | discard rate |
|---|--:|--:|--:|
| A-identical-n1 | 3  | 0  | 0%   |
| A-identical-n2 | 7  | 6  | 86%  |
| A-identical-n4 | 16 | 15 | 94%  |
| A-identical-n8 | 32 | 32 | 100% |
| B-differing-n1 | 1  | 0  | 0%   |
| B-differing-n2 | 13 | 13 | 100% |
| B-differing-n4 | 28 | 28 | 100% |
| B-differing-n8 | 40 | 40 | 100% |
| C-mixed-n1 | 3  | 0 | 0%  |
| C-mixed-n2 | 3  | 2 | 67% |
| C-mixed-n4 | 6  | 4 | 67% |
| C-mixed-n8 | 12 | 8 | 67% |

At N=1 there is by definition no batch-composition mismatch possible (one row, no peers to
disagree with), so the discard rate is 0% in every N=1 run. As soon as N>=2, discard rate rises
sharply -- 86-100% for workloads A and B, which is exactly the failure mode
`06-batch-shape-measurement.md` describes: `completion_batch_indices` admits rows on token budget
alone with no splice-width homogeneity test, so `resolve_pending_ff_batch` finds a non-homogeneous
batch on the very next step and discards every splice in it. Workload C (mixed
constrained/unconstrained) discards at a lower but still substantial 67% from N=2 onward, not 100%
-- worth noting since the plan expected C's per-batch minimum to be zero "by construction" for the
unconstrained-vs-constrained split; the 67%-not-100% figure suggests some splices in the
constrained subset are still landing in same-width sub-batches by chance at this N, not that the
mechanism doesn't apply.

## Dimension 2 -- forced tokens lost

| combo | tokens_fed | tokens_dropped(batch_shape) | fraction lost |
|---|--:|--:|--:|
| A-identical-n1 | 4 | 0  | 0%   |
| A-identical-n2 | 2 | 7  | 78%  |
| A-identical-n4 | 1 | 19 | 95%  |
| A-identical-n8 | 0 | 40 | 100% |
| B-differing-n1 | 1 | 0  | 0%   |
| B-differing-n2 | 0 | 23 | 100% |
| B-differing-n4 | 0 | 57 | 100% |
| B-differing-n8 | 0 | 80 | 100% |
| C-mixed-n1 | 4 | 0 | 0%  |
| C-mixed-n2 | 2 | 2 | 50% |
| C-mixed-n4 | 4 | 4 | 50% |
| C-mixed-n8 | 8 | 8 | 50% |

Token-loss fraction tracks splice-loss fraction closely, and for A/B reaches 100% at N>=2 (A/N=2)
or N>=2 (B/N>=2): once concurrency is present, essentially every forced token this workload could
have contributed is discarded. This directly answers the plan's concern that "a high splice-drop
rate on short splices matters much less than a low one on long splices" -- here the two rates move
together, so there is no evidence in this data of a workload where many small splices are lost but
few tokens are lost, or vice versa.

## Dimension 3 -- batch composition histogram

**Not measured.** This is the one source instrumentation change `06-batch-shape-measurement.md`
explicitly permits, but this session was instructed not to make Rust source changes, so no
per-decode-step histogram of splice-carrying vs splice-free rows exists. `splice_drops_total` and
`tokens_dropped_total` (dimensions 1-2 above) are a downstream proxy for how often a batch was
non-homogeneous, but they are not the same measurement -- they count discard events at the point
`resolve_pending_ff_batch` acts, not the underlying per-step row composition, and they say nothing
about which fraction of steps were all-splice, mixed, or none (the plan's specific ask).

## Dimension 4 -- splice-width distribution and per-batch minimum

**Only partially recoverable, and not to the resolution the plan asks for.** The harness/metrics
surface gives run-level aggregates only (`splices_staged_total`, `tokens_fed_total`,
`splice_drops_total`, `tokens_dropped_total`), not per-splice or per-batch-step values, so neither
a true width distribution nor `min(K_1..K_n)` per all-splice batch can be computed. What can be
derived is a coarse average successful-splice width per run, `tokens_fed_total /
(splices_staged_total - drops)`:

| combo | successful splices | tokens fed | avg width (fed / successful) |
|---|--:|--:|--:|
| A-identical-n1 | 3 | 4 | 1.33 |
| B-differing-n1 | 1 | 1 | 1.00 |
| C-mixed-n1 | 3 | 4 | 1.33 |
| C-mixed-n2 | 1 | 2 | 2.00 |
| C-mixed-n4 | 2 | 4 | 2.00 |
| C-mixed-n8 | 4 | 8 | 2.00 |

(Rows where `successful splices = 0`, i.e. 100% discard, are omitted -- there is nothing to
average.) These are small-sample run-level averages, not a distribution, and cannot support any
claim about per-batch minimum width or what a New-C-style truncation could recover. A true answer
to this dimension needs the same source instrumentation dimension 3 needs (or a superset of it),
which was out of scope here.

## Dimension 5 -- end-to-end effect: tokens/s and forward passes, flag on vs off

Using `mistralrs_decode_tokens_processed_total` delta over `wall_elapsed_s` as an aggregate
cluster-throughput proxy (see results table above for the full per-combo numbers):

| workload | N | tok/s ON | tok/s OFF | ON/OFF ratio |
|---|--:|--:|--:|--:|
| A | 1 | 162.5 | 176.4 | 0.92 |
| A | 2 | 236.5 | 229.3 | 1.03 |
| A | 4 | 299.6 | 301.1 | 0.99 |
| A | 8 | 416.0 | 414.9 | 1.00 |
| B | 1 | 172.2 | 179.9 | 0.96 |
| B | 2 | 243.1 | 224.4 | 1.08 |
| B | 4 | 304.1 | 305.5 | 1.00 |
| B | 8 | 365.9 | 362.8 | 1.01 |
| C | 1 | 181.6 | 175.7 | 1.03 |
| C | 2 | 216.0 | 201.1 | 1.07 |
| C | 4 | 312.1 | 284.7 | 1.10 |
| C | 8 | 385.7 | 394.5 | 0.98 |

No workload/N combination shows a large, consistent effect in either direction. Ratios sit in a
roughly 0.92-1.10 band around parity, with no clean monotonic trend against N and no combination
where ON is dramatically faster despite dimension-1 showing FF successfully staging and applying
splices at N=1 (A-n1, C-n1: 3 staged, 4 fed, 0 drops) and partially at C's higher N. This is
consistent with -- though this single-session sweep cannot prove -- the plan's architectural
finding that the scheduler gap discards most splices before they can matter once concurrency rises,
and that even where splices do land (small N, or C's partial success), the volume of forced tokens
recovered (1-8 tokens over a 44-352 token run) is too small a fraction of the run to move aggregate
throughput outside run-to-run noise at this sample size (one run per cell, no repeats).
`mistralrs_cuda_graph_dispatch_total{mode="replay"}` (forward-pass-via-CUDA-graph count) was not
included in this table but is present in the raw JSON alongside the same near-parity pattern.

**This is not a null result to be read as "FF has no effect"; it is an underpowered one.** One run
per cell, no repeated-seed variance estimate, and (see below) at least one run pair shows the
generated content itself was not identical between ON and OFF, which undermines a clean apples-to-
apples throughput comparison for that cell specifically.

## Dimension 6 -- complexity/risk inputs

No `mistralrs_paged_preemptions_total` (or any metric containing "preempt") appeared in `/metrics`
under either flag setting, at any point in this session -- either the counter was never registered
because no preemption occurred, or no such counter is exported by this build; this session did not
distinguish the two. `mistralrs_kv_cache_blocks_used` returned to 0 (fully idle, `available ==
total == 10546`) between sweeps and after the whole session, with no visible pressure at any N up
to 8. No starvation was visible in the `finish_reason` data beyond the equality anomaly below (no
request timed out, errored, or returned an unexpected status; `count_failed: 0` throughout). This
is a favorable-conditions run (max 8 concurrent requests against `--max-seqs 8`, i.e. never over
the configured cap) and says nothing about behavior at higher desired concurrency than the
scheduler's configured capacity.

## Correctness / equality

**This is the section most worth reading closely.** The plan's objective list requires
"correctness/equality remains intact between FF ON and FF OFF." Across the 24 runs, that held for
21 of the 24-run-equivalent request comparisons by `finish_reason`, but not for all of them:

1. **B-differing, N=1 (no batch-shape confound possible):** the single request against
   `nested.schema.json` finished with `finish_reason="length"` (hit `max_tokens=64` without the
   grammar naturally completing) under FF **ON**, but `finish_reason="stop"` (completed naturally,
   well under the token budget) under FF **OFF**. `decode_tokens_processed_total` delta for this
   one request was 63 (ON) vs 46 (OFF). N=1 means there is no concurrent peer row for this request
   to disagree with in width, so this is not an instance of the batch-shape mechanism the rest of
   this report is about -- it is either FF taking a different, numerically non-identical code path
   even for an uncontested single sequence (a plausible mechanism: FF's verification/splice step
   may dispatch different kernels than pure autoregressive decode, and floating-point results are
   not guaranteed bit-identical across different kernel paths, which can flip an argmax choice at
   a close tie), or some other source of run-to-run variation this session did not isolate. This
   session did not repeat this specific cell with a different seed to check reproducibility, so it
   cannot say whether this divergence is deterministic-and-repeatable or incidental.
2. **B-differing, N=8, same schema, same slot pattern:** index 0 of this run (also
   `nested.schema.json`) reproduces the same ON=`length`/OFF=`stop` split as the N=1 case. But
   index 4 (a second `nested.schema.json` request within the *same* N=8 run, same flag) is
   `length` under **both** ON and OFF -- and the OFF run's own index-0-vs-index-4 split (`stop` vs
   `length` for the same schema, same flag, same run) shows that this schema's finish behavior is
   not stable across batch position even with the flag held constant. This weakens (does not
   settle) the case that dimension-1's finding is the whole explanation for item 1 above: some of
   the instability here is present under OFF alone, i.e. concurrent-batching itself, independent of
   FF, is a source of variation for this schema.
3. **C-mixed, N=4:** all four `finish_reason`s matched (`stop` x4 on both sides), but
   `decode_tokens_processed_total` delta was 132 (ON) vs 122 (OFF) for the same nominal workload,
   same seed. Matching `finish_reason` labels do not guarantee matching content length here.

**What this does and does not establish:** this session did not capture full response text (the
harness's `concurrency` mode records `finish_reason` and latency, not the decoded string), so none
of the three items above can be confirmed as a token-for-token divergence versus a metric-counting
artifact -- only that the *signals available* (finish reason, aggregate decode-token count) are not
identical between ON and OFF for these specific cells, including one (item 1) with no batch-shape
confound at all. This is exactly the kind of finding `06-batch-shape-measurement.md`'s human
checkpoint after this plan exists to weigh; it is reported here, not resolved or explained away.
Nothing in `05-harness.md`'s `equality` mode (the mode actually designed to catch token-for-token
divergence, via `Which.Plain` and a fixed seed) was available to cross-check this, for the same
reasons `05-harness-live-validation-checkpoint.md` already recorded: no `mistralrs` package, no
top-level `tokenizer.json` for this GGUF repo, from this container.

## Files added

- `plans/ff-round-two/reports/06-batch-shape.md` (this report)
- `plans/ff-round-two/reports/06-sweep/06-concurrency-{on,off}-*.json` (24 raw harness reports,
  one per workload/N/flag combination)

## Git/worktree state

- `/workspace`: branch `grammar-fast-forward`, HEAD `cab8cd3ae`, clean throughout this session.
  Never checked out `ff-demo-artifacts` there, no source edited, no rebuild.
- This report and its raw data were produced in, and (if committed) committed from, a separate
  `git worktree` at `/tmp/ff-artifacts-wt`, branch `ff-demo-artifacts`.
- The host server was restarted twice by the user during this session (ON, then OFF), both times
  to the exact configuration recorded above; this container never started, stopped, or had control
  over either process, per instruction.

## Summary against the plan's deliverable checklist

- Dimensions 1, 2: measured, with real ON-only data (OFF is structurally zero). Discard/loss rates
  rise sharply with N for workloads A and B (86-100% by N=2-8), and more moderately for C (67%).
- Dimension 3: not measured (source change out of scope this session).
- Dimension 4: only a coarse run-level average width recoverable; true distribution/per-batch
  minimum not obtainable from existing counters.
- Dimension 5: measured across all 24 cells; no large or consistent ON/OFF throughput effect
  (0.92-1.10x band), but underpowered (one run per cell) and undermined for at least one cell by
  the equality anomaly in item 3 above.
- Dimension 6: no preemption signal found either way; no KV pressure observed at N up to 8; no
  request failures.
- **Correctness/equality:** did not cleanly hold in all cases -- see "Correctness / equality"
  above. This is flagged, not resolved, and should be read before any dimension-5 number is used
  for anything.

**Numbers only, as instructed -- no ranking of B/C/D1, no recommendation.** Not proceeding to
Plan 07.
