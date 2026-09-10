# 06 -- FF-activation precondition: attempt and result (2026-09-10, second session)

Status: **precondition NOT discharged**. `mistralrs_grammar_ff_splices_staged_total` was not observed
in this session, non-zero or otherwise, because no request reached the instrumented call site at
all. Both routes this session could try were blocked by container/build capability gaps, not by any
FF-specific behaviour. Plan 06's batch-shape sweep was **not** started, per this task's scope.

## Environment, established first

This session runs in a container distinct from the one that produced the 2026-09-10 CUDA smoke test
in `reports/05-harness.md`. `git worktree list` shows a single worktree (`/workspace`,
`ff-demo-artifacts` checked out); there is no secondary worktree with `grammar-fast-forward` checked
out, so all source-reading below used `git show grammar-fast-forward:<path>` against the single
checkout rather than a second working tree.

Inventory taken before any run:

- **No GPU device passthrough.** `/proc/devices` lists `nvidia`, `nvidia-uvm`, etc. (the host has an
  NVIDIA driver), but no `/dev/nvidia*` nodes exist in this container and `nvidia-smi` is not
  installed. The GPU is not attached to this session.
- **No CUDA runtime on disk.** `ldconfig -p` has no `libcudart.so*` or `libcuda.so*` entry, and none
  exist anywhere on the filesystem.
- **No Rust toolchain.** `cargo`/`rustc` are not installed; no `~/.cargo`.
- **A leftover CUDA-linked debug binary exists**, `/workspace/target/debug/mistralrs` (build
  artifacts under `/workspace/target/` from a prior session sharing this disk), but it cannot start:
  `error while loading shared libraries: libcudart.so.12: cannot open shared object file`. Its
  `strings` output confirms it was built with the CUDA feature (`cudaMalloc`, `cudaLaunchKernelExC_ptsz`,
  etc.), so it is not a fallback CPU build.
- **A CPU-only Python wheel is committed to this branch**:
  `mistralrs-0.9.3-cp310-abi3-manylinux_2_35_x86_64.whl` (referenced already in `reports/05-harness.md`
  for its `.pyi`). `ldd` on the extracted `mistralrs.abi3.so` shows only `libgcc_s`, `libm`, `libc` --
  no CUDA dependency at all. It imports successfully under Python 3.11.2 with no `pip` needed
  (extracted and put on `PYTHONPATH`; no network install required for the module itself).
- No HF cache, no local GGUF files, no local `tokenizer.json` existed at session start. Outbound
  HTTPS to `pypi.org` and `huggingface.co` both work.

This means: the `concurrency` harness mode (needs a paged-attention CUDA/Metal *server* binary) has
no viable binary in this container, built or buildable, without provisioning a GPU and a Rust
toolchain -- both out of this task's scope. The only route available at all is the in-process
`equality`/`routing-log`/`arms` path via the committed CPU-only wheel.

## Attempt 1: non-GGUF HF checkpoint via `equality` mode (Python Runner path)

Per the task brief, `ff_harness.py --help` / `run --help` was read first (unchanged). Command:

```
PYTHONPATH=/tmp/whl_extract MISTRALRS_GRAMMAR_FAST_FORWARD=1 python3 ff_harness.py run \
  --mode equality --flag on --model-id Qwen/Qwen2.5-0.5B-Instruct --arch Qwen2 \
  --max-tokens 4 --seed 1 --schema-file plans/ff-round-two/fixtures/partial.schema.json \
  --plan attempt1 --label smoke
```

`Qwen/Qwen2.5-0.5B-Instruct` was chosen as a small, `Architecture.Qwen2`-mapped, non-GGUF checkpoint
with a real top-level `tokenizer.json`, matching plan 05's own suggested discriminator. No download
occurred before the failure below.

**Result: failed before inference**, every time, regardless of model:

```
ValueError: cannot seed the CPU rng with set_seed
```

raised inside `mistralrs.Runner(**runner_kwargs)` (`ff_harness.py:464`, called from
`build_plain_runner`). Full evidence and a minimal repro are in
`06-equality-on-cpu-blocked-20260910T152440Z.json`.

### Root cause (established, not guessed)

This is not model-, tokenizer-, or grammar-specific. Isolated repro:

```python
mistralrs.Runner(which=Which.Plain(model_id="Qwen/Qwen2.5-0.5B-Instruct", arch=Architecture.Qwen2),
                  seed=42)   # -> ValueError: cannot seed the CPU rng with set_seed, instantly
```

fails identically and instantly (no network activity). The wheel's compiled string table contains
the unconditional pair `cannot get the CPU rng seed with get_current_seed` /
`cannot seed the CPU rng with set_seed` -- this is candle-core's CPU `Device` having no settable RNG
seed in this build, not a mistral.rs-level check. Confirmed by bypassing `ff_harness.py` for
diagnosis only (not used for any report): `Runner(which=..., seed=None)` gets past construction and
starts pulling the model (an HF-token INFO log line and outbound network activity appeared before a
15s diagnostic timeout cut it off). So the failure is specifically "a concrete seed value + CPU
device", independent of everything else about the request.

`ff_harness.py`'s `run` subcommand always supplies a concrete seed (`--seed` defaults to `42`, and
`build_plain_runner` passes `seed=args.seed` unconditionally, `ff_harness.py:459`) -- there is no flag
to pass `seed=None` through the harness's existing interface, and per this task's constraints
`ff_harness.py` was not modified to add one. `routing-log` and `arms` share `build_plain_runner`, so
they are blocked identically; `concurrency` doesn't use this path but has no runnable binary here
(above).

**This blocks all three in-process harness modes in this container, for any model, any schema, any
seed value, any flag setting.** It is a container/build capability gap -- CPU-only wheel plus a
harness that always seeds plus no CUDA/Metal device to construct a seedable `Runner` -- not a finding
about `MISTRALRS_GRAMMAR_FAST_FORWARD` or fast-forward staging. No request of any kind reached
`consume_ff_tokens()` or the `tracing::debug!("fast-forward splice computed")` /
`mistralrs_grammar_ff_splices_staged_total` site at `mistralrs-core/src/pipeline/sampling.rs:1633-1637`
(re-read on `grammar-fast-forward` in this session to confirm the two fire together, still at the
same lines as reported in `05-harness.md`).

## Attempt 2: longer generations / more forced structure (`nested`/`wide` schemas)

**Not reached.** Attempt 2 only makes sense once Attempt 1 produces a live request to lengthen or
restructure; here Attempt 1 never got past `Runner` construction for any model, so there was nothing
to extend. Re-running the identical broken command with `--schema-file nested.schema.json` or
`wide.schema.json` would reproduce the exact same `ValueError` before touching the schema at all, so
it was not separately executed (would add a duplicate, uninformative artifact, not new evidence).

## What this does and does not establish

It does **not** establish that fast-forward is broken, and it does **not** establish that fast-forward
activates. No inference occurred; `mistralrs_grammar_ff_splices_staged_total` was not observed
because nothing ran that could increment it, in either direction.

Of the four candidate explanations `05-harness.md` listed for the CUDA smoke test's zero counters,
none is newly supported or ruled out by this session -- this session produced a different kind of
non-result (construction-time failure, zero requests) rather than a completed run with a zero
counter. A fifth, previously-unlisted explanation applies specifically to *this* session:

- **(new) Container capability gap**: this session's container has no GPU device passthrough, no CUDA
  runtime libraries, and no Rust toolchain, so neither the `concurrency` (server) path nor, it turns
  out, the in-process `equality`/`routing-log`/`arms` path (blocked by a CPU-only wheel's RNG
  limitation, independent of FF) could reach a live request at all. This is orthogonal to the four
  candidates in `05-harness.md`, which concern *why a completed request produced no splice*; this
  session never completed a request.

The two cheap discriminators `05-harness.md` proposed for a *next* session remain exactly as open as
before:

1. Non-GGUF HF checkpoint via `Which.Plain` -- attempted here, blocked before it could distinguish
   anything about tokenizer canonicality.
2. Longer/more-structured generations -- not reached.

## What a next session needs, concretely

Either of these would very likely unblock the precondition check without touching source,
`ff_harness.py`, or Plan 06:

- **A container with GPU passthrough** (the CUDA runtime and driver this task's environment note says
  the host has, per `/proc/devices`, just not attached here) so the already-committed CUDA-capable
  build path (or a freshly built one, on a session with a Rust toolchain) can run `equality` or
  `concurrency` mode, where `Runner(seed=...)`/the server's own seeding does not hit the CPU-only
  `set_seed` limitation.
- **Or** a CPU-only wheel/build where `Runner(seed=...)` on CPU is either fixed upstream or the
  caller can omit the seed -- neither of which this task authorises attempting here (would mean
  patching `ff_harness.py` or `mistralrs-core`/its `candle-core` dependency).

## Files added

- `plans/ff-round-two/reports/06-ff-activation-precondition.md` (this report)
- `plans/ff-round-two/reports/06-equality-on-cpu-blocked-20260910T152440Z.json` (failed-attempt
  evidence: environment inventory, exact command, traceback, minimal repro, root-cause strings)

No changes to `grammar-fast-forward`, `ff_harness.py`, or any Plan 06 document.

## Precondition status for Plan 06

**Not discharged.** Plan 06's workload sweep must not proceed from this session's evidence; dimensions
1, 2 and 4 remain *not yet measurable*, exactly as `05-harness.md` already required. This session adds
no new reason to relax that requirement and no new reason to believe FF is broken -- only a documented,
container-specific reason why this attempt could not test it either way.
