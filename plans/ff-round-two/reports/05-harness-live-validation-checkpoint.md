# 05 -- Harness/live-server validation checkpoint (2026-09-10)

Status: **partial pass, with a documented harness gap.** This is a checkpoint on whether the
existing `ff_harness.py` (as it stands at plan 05, unmodified) can drive the already-running live
host server and distinguish real FF execution from a control. It is **not** Plan 06's batch-shape
sweep and makes **no** performance claims.

Source activation commit: `cab8cd3ae`. Artifact base: `bfbc1505b`. No Rust source touched, no
server restart, no rebuild.

## Connectivity

From inside this container:

```
curl -s -o /dev/null -w "%{http_code}\n" http://host.docker.internal:1234/health
200
```

Live server (unchanged, launched before this checkpoint on the host):

```
MISTRALRS_GRAMMAR_FAST_FORWARD=1 target/debug/mistralrs serve \
  --model-id unsloth/Qwen3.5-4B-GGUF --quant 4 --paged-attn on \
  --max-batch-size 8 --max-seqs 8 --no-ui --port 1234 -v
```

## What the harness can and cannot do here

`ff_harness.py`'s `equality`/`routing-log`/`arms` modes (and therefore `compare`, which only accepts
those three) are **in-process only**: they build a `mistralrs.Runner` over `Which.Plain` inside the
harness's own Python process and never speak HTTP. Two independent blockers rule them out for this
checkpoint:

1. `import mistralrs` fails in this container (`ModuleNotFoundError: No module named 'mistralrs'`)
   -- confirmed directly, not assumed.
2. Even with the package installed, `Which.Plain` needs a top-level `tokenizer.json`, which
   `unsloth/Qwen3.5-4B-GGUF` does not expose -- already recorded as a known limitation in
   `05-harness.md`'s live-smoke-test section.
3. `compare --mode concurrency` is rejected outright by argparse (`invalid choice: 'concurrency'`,
   choices are `equality`/`routing-log`/`arms`) -- confirmed directly. There is no `compare` path
   for the one mode that does speak HTTP.

So the only harness mode that can reach the live server at all is `concurrency`, and its `compare`
counterpart does not exist.

`concurrency` mode itself is built to launch and own its server subprocess
(`subprocess.Popen(args.server_cmd, ...)`, then `proc.terminate()` in a `finally` block) -- it has no
"attach to an already-running server" path. Its actual request/metrics traffic, however, goes to
`--server-host`/`--server-port`, which is independent of what `--server-cmd` spawns. This checkpoint
exploited that decoupling rather than modifying the harness: `--server-cmd "sleep 120"` supplies a
harmless placeholder process (satisfying `Popen`/`terminate`), while `--server-host
host.docker.internal --server-port 1234` points the actual HTTP traffic and `/health`,
`/v1/chat/completions`, `/metrics` calls at the real live server, which was already healthy so the
poll succeeded immediately. This is a permitted use of an existing, documented CLI surface (two
independently-purposed flags), not a code change, and the placeholder process never touches the real
server.

**Hard limit, independent of the above:** `MISTRALRS_GRAMMAR_FAST_FORWARD` is read once through a
`OnceLock` at pipeline load (`perf_flags.rs:33-35`), so it is a process-wide load-time constant.
The live server was launched once, with the flag set to `1`, and this checkpoint was instructed not
to restart or rebuild it. There is therefore **no way to obtain a true FF-OFF observation from this
specific live process** -- not a harness defect, a structural consequence of the flag's design plus
this checkpoint's own "don't restart" constraint. `--flag on/off` on the harness's `run`/`compare`
subcommands only ever controls the env of a process the harness itself spawns; it has no effect on
an external, already-running server, confirmed by reading `run_concurrency`'s `env` handling, which
only ever reaches the harness's own `subprocess.Popen`.

## What was run

Three `concurrency`-mode invocations against the live server, `--num-requests 1` each (the cheapest
single-request signal), report artifacts alongside this file:

| Run | Command flags (beyond the shared base below) | File |
|---|---|---|
| A: constrained, run 1 | `--unconstrained-fraction 0.0 --seed 42 --label live-on` | `checkpoint-concurrency-on-live-on-20260910T171436Z.json` |
| B: unconstrained control | `--unconstrained-fraction 1.0 --seed 42 --label live-unconstrained` | `checkpoint-concurrency-on-live-unconstrained-20260910T171447Z.json` |
| C: constrained, run 2 (reproducibility) | `--unconstrained-fraction 0.0 --seed 43 --label live-on-rep2` | `checkpoint-concurrency-on-live-on-rep2-20260910T171502Z.json` |

Shared base:

```
python3 ff_harness.py run --mode concurrency \
  --model-id unsloth/Qwen3.5-4B-GGUF --flag on \
  --server-cmd "sleep 120" \
  --server-host host.docker.internal --server-port 1234 \
  --schema-set identical --max-tokens 64 --plan checkpoint \
  --metric-prefix mistralrs_ --out-dir plans/ff-round-two/reports
```

`--flag on` here is a no-op against the live process (see above); it was left `on` for
self-documentation rather than to claim it did anything.

## Results

### Run A -- constrained request against the live (flag-fixed-on) server

- HTTP 200, `finish_reason: stop`, latency 0.255 s.
- FF metric deltas (before -> after), all others in the `mistralrs_grammar_ff_` family zero:
  - `mistralrs_grammar_ff_attempts_total{outcome="empty_splice"}` +36
  - `mistralrs_grammar_ff_attempts_total{outcome="grammar_stopped"}` +2
  - `mistralrs_grammar_ff_attempts_total{outcome="staged"}` +3
  - `mistralrs_grammar_ff_splices_staged_total` +3
  - `mistralrs_grammar_ff_tokens_fed_total` +4
- Broader deltas (prefill/decode token counters, CUDA graph dispatch, queue histogram, etc.) also
  present and consistent with one served request; not FF-specific, recorded in the raw JSON.

### Run B -- unconstrained request against the same live server (achievable control)

- HTTP 200, `finish_reason: stop`, latency 0.131 s, `constrained: false`, no grammar sent.
- **Zero** `mistralrs_grammar_ff_*` deltas of any kind (attempts, staged, tokens fed all absent /
  0). No FF attempt path was reached at all, as expected for a request carrying no grammar.

### Run C -- constrained request, repeated (different seed)

- HTTP 200, `finish_reason: stop`.
- Identical FF deltas to Run A: `attempts{empty_splice}` +36, `attempts{grammar_stopped}` +2,
  `attempts{staged}` +3, `splices_staged_total` +3, `tokens_fed_total` +4.

The staged/attempts numbers reproducing exactly across A and C (same schema fixture, same
`max_tokens`, different seed) indicates the splice-staging count here is driven by the grammar
fixture's forced structure, not by run-to-run sampling noise at this length.

## Compare-mode exercise (step 5 of the checkpoint procedure)

Not run against these results: `ff_harness.py compare` only accepts `--mode
{equality,routing-log,arms}` and rejects `concurrency` at the argument-parsing stage (confirmed
above), and the accepted modes cannot reach this live server or this model at all in this
container (no `mistralrs` package, no `tokenizer.json` for the GGUF repo). This is the harness
limitation the checkpoint procedure asked to have documented rather than papered over: **there is
no existing `compare` path from inside this container to this live server.** The ON-vs-control
distinction above was established by manually diffing `metrics_delta` across two `run` reports
(the same arithmetic `compare_reports`/`metric_deltas` already does internally), not by invoking
`compare`.

## Conclusion

The existing harness's HTTP-capable mode (`concurrency`), run against the live host server through
its `--server-host`/`--server-port` flags, correctly and reproducibly observes real FF execution:
a constrained request stages 3 splices feeding 4 tokens, twice, with matching attempt-outcome
counts; an unconstrained request against the identical live process shows zero FF-metric movement.
That is a genuine, reproducible ON-vs-no-FF-activity distinction driven end-to-end from this
container through the live server.

What this checkpoint does **not** establish, and could not, given the constraints (no restart, no
rebuild) and the harness's actual capabilities (no `compare` support for `concurrency`, no
attach-to-existing-server mode by design, no in-process path usable against this GGUF model in this
container): a true FF-flag-OFF observation from this server process, or an automated `compare`
verdict rather than a manual one. Both are structural gaps -- one in the flag's process-wide
`OnceLock` design combined with the "don't restart" constraint, one in the harness's mode/compare
matrix -- not something papered over by this checkpoint's workaround (the harmless placeholder
`--server-cmd`, which only supplies a process for the harness to manage and terminate, and never
touches the real server or its flag).

**This is harness/FF-execution validation, not a quantitative speedup measurement, and not a
flag-on-vs-flag-off comparison** (the latter was structurally unavailable against a single
already-running process).

## Files added

- `plans/ff-round-two/reports/05-harness-live-validation-checkpoint.md` (this report)
- `plans/ff-round-two/reports/checkpoint-concurrency-on-live-on-20260910T171436Z.json`
- `plans/ff-round-two/reports/checkpoint-concurrency-on-live-unconstrained-20260910T171447Z.json`
- `plans/ff-round-two/reports/checkpoint-concurrency-on-live-on-rep2-20260910T171502Z.json`

No changes to `ff_harness.py`, `ff_bench.py`, or any file on `grammar-fast-forward`.
