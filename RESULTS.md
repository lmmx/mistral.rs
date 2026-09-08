# Grammar fast-forward: demo results

Ad-hoc demo artifacts for the `grammar-fast-forward` branch (commits `e7bba90be`, `e9b0eac07`,
`d217eb32b` on top of v0.9.3 `d5ae0f18f`). Not part of that branch's history -- this is a separate
storage branch (`ff-demo-artifacts`) with no shared commits, so it can't interfere with it.

## Setup

- Model: `unsloth/Qwen3.5-4B-GGUF` / `Qwen3.5-4B-Q4_K_M.gguf` (cached locally).
- Hardware: 20-core CPU container, no GPU.
- Grammar: regex forcing one exact ~40-word fixed passage end-to-end (the fully-forceable case --
  see `ff_bench.py`), `enable_thinking=False`, `temperature=0.0`.
- Three installs compared, same query, 5 timed repeats each after 1 warmup:
  1. Published PyPI `mistralrs==0.9.3` (unpatched).
  2. This branch's wheel, `MISTRALRS_GRAMMAR_FAST_FORWARD` unset (flag off, the default).
  3. Same wheel, `MISTRALRS_GRAMMAR_FAST_FORWARD=1`.

## Results

| Build | Flag | Median | Range (5 repeats) |
|---|---|---:|---|
| Published 0.9.3 | n/a | 7.55s | 7.25s - 7.63s |
| This branch | off (default) | 7.39s | 7.25s - 8.35s |
| This branch | `=1` | **1.22s** | 1.18s - 1.34s |

- Flag off vs. published: within run-to-run noise -- no behavior change when the flag isn't set.
- Flag on: **~6.1-6.2x faster** than either baseline (7.55s / 1.22s = 6.19x; 7.39s / 1.22s = 6.06x),
  non-overlapping ranges. Matches the ~6.4x order of magnitude the sumac journal measured on a
  similarly fully-forceable grammar.
- Every run produced the exact same fixed passage byte-for-byte in all three configurations.

This is the ceiling case (100% of the completion is grammar-forced). A real tool-call grammar with
only a few forced scaffold tokens per response would see a much smaller (sumac's own journal
measured ~3%) but still real win -- not reproduced here.

## Files in this branch

- `ff_bench.py` -- the benchmark script.
- `build-branch.log` -- full `maturin build --release` output for the branch wheel.
- `mistralrs-0.9.3-cp310-abi3-manylinux_2_35_x86_64.whl` -- the actual wheel these numbers were
  measured against (built from the `grammar-fast-forward` branch, commit `d217eb32b`).

Not included: the two venvs used to run this (pure dependency installs, ~180MB combined, trivially
reproducible with `uv venv && uv pip install <wheel-or-mistralrs==0.9.3>`).
