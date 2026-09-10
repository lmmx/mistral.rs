# 05 -- The experimental harness

Status: implemented, locally validated at the Python-syntax/argument-parsing/helper-logic level.
Not run end-to-end against a live model -- no Rust toolchain, no built `mistralrs-cli`/server
binary, and no downloaded model in this session, consistent with every prior report in this plan
set.

## Where the harness lives, and why

The plan's own README states the branch split precisely: `grammar-fast-forward` (tip `4ba40ca9e`
at the time of this session) holds `mistralrs-core/` and no `docs/`, `plans/`, or `ff_bench.py` at
all; `ff-demo-artifacts` is the artifacts-only branch that holds those, with **no
`mistralrs-core/` source tree at all**. Plan 05 itself repeats this for the harness specifically:
"harness code lives beside `ff_bench.py` on the same branch" -- i.e. `ff-demo-artifacts`. Plan 05's
non-goals section is explicit: "No new inference features, no changes to `mistralrs-core` from
this plan."

This session verified both branches directly (`git ls-tree -r grammar-fast-forward --name-only`
has no `docs/journal`, `plans/`, or `ff_bench.py`; `git ls-tree -r ff-demo-artifacts --name-only`
has all of them, plus the wheel and `RESULTS.md`) and confirmed the two branches share no common
ancestor (`git merge-base` returns nothing) -- this is a deliberate two-repo split, not an
oversight. Given that, and given plan 05's own non-goals, **no change was made to
`grammar-fast-forward`** in this session. The task brief's generic "source worktree" framing
assumed `ff_bench.py` lived there; it does not, and the plan document is the authority per this
task's own instruction to "follow its defined interface rather than inventing a different one."

## What was implemented

One new file, `ff_harness.py`, committed to `ff-demo-artifacts` beside the existing `ff_bench.py`
(untouched), plus one fixture:

- `ff_harness.py` (new)
- `plans/ff-round-two/fixtures/partial.schema.json` (new) -- the committed partially-forcing JSON
  schema fixture the plan requires. It forces an object shape (`{"status": ..., "message": ...,
  "retry_count": ...}`, `additionalProperties: false`, so braces/keys/colons/ordering are forced)
  and leaves free spans open: `status` is a 4-member enum, `message` is a free string, `retry_count`
  is a free-choice integer in range.

`RESULTS.md` and `ff_bench.py` were not read for modification and were not touched, per the plan.

## Interface implemented

`python ff_harness.py run --mode {equality,routing-log,arms,concurrency} --flag {on,off,unset} ...`
runs exactly one mode at one flag setting in the current process and writes one JSON report to
`plans/ff-round-two/reports/<plan>-<mode>-<flag>[-<label>]-<timestamp>.json`, printing `wrote
<path>` to stderr.

`python ff_harness.py compare --mode {equality,routing-log,arms} ...` (no `--flag`: it sets one
itself in each child) spawns two `ff_harness.py run` invocations as **separate OS processes** --
one with `MISTRALRS_GRAMMAR_FAST_FORWARD` unset from the child env then set to `0`, one set to `1`
-- and diffs the two reports' token-id lists, reporting the first divergent index and both
continuations plus decoded content when they differ. This is the two-process design the plan
requires for the `OnceLock`-backed flag: nothing in this file toggles
`MISTRALRS_GRAMMAR_FAST_FORWARD` inside a running process; `run` reads it once, before constructing
`Runner`/launching the server subprocess, and never again.

### `equality`

Loader: `Which.Plain(model_id, arch, dtype, tokenizer_json)`, all from CLI args, so any HF
checkpoint works (not `Which.GGUF` like `ff_bench.py`). One streamed request (`stream=True`) with
the committed fixture as `grammar_type="json_schema"`, `temperature=0.0`, `enable_thinking=False`,
fixed `seed`. Each chunk is timestamped (`step_duration_s`, `elapsed_since_start_s`) to get
per-step timings without a separate timing loop. Token ids come from `logprobs=True,
top_logprobs=1`: at `temperature=0.0` the sampled token is the top logprob candidate, so
`ChunkChoice.logprobs.top_logprobs[0].token` is taken as that step's token id. This is a
documented assumption (recorded in a code comment), not something this session could verify against
a live model. If a chunk carries no logprobs (backend/model dependent), `token_ids_complete` is
`false` and `compare` reports "not comparable" rather than diffing a partial or None-padded list --
verified by a unit-level check (see Validation).

The JSON report carries: full token id list, decoded text, finish reason, per-step timings, and the
flag value as read from `os.environ` after the harness's own override (`flag_env_value`), plus the
request shape (model id, arch, dtype, seed, prompt, schema file path, max tokens).

### `routing-log`

`equality` plus best-effort capture: `os.dup2` redirects the real OS-level fd 2 (not Python's
`sys.stderr` object -- required because Rust's tracing subscriber writes to the process's actual
stderr fd) to a temp file for the duration of the request, then greps it against a regex (default
`EXPERT_IDX`, overridable via `--capture-pattern`) and stores matched lines plus a count in the
report's `capture` object.

**This currently captures nothing.** Reading `mistralrs-core/src/amoe/mod.rs` on
`grammar-fast-forward` (`MoeMlp::forward`, around the `topk(1)` call cited by
`reports/03-anymoe-divergence.md`) shows no `tracing`/`debug!` call at that site at this tip --
Deliverable 2 of plan 03/04's Part B, which would add it, was not run in any prior session (no Rust
toolchain). `routing-log` is the capture mechanism such instrumentation can be pointed at later; it
is not a claim that AnyMoE routing divergence is observed today, and adding that instrumentation to
`mistralrs-core` is out of scope for plan 05 by its own non-goals. It is also worth noting
explicitly: grammar fast-forward is currently *disabled* for AnyMoE (`fix: disable grammar fast
forward for AnyMoE`, `4ba40ca9e`), so there is presently no live divergence for this mode to find
even once instrumented -- its use is for a future, deliberate re-enable-and-measure experiment, not
for today's tree.

### `arms`

Same shape as `routing-log`, default pattern `\bARM\b`, for the nine recurrent-site arm messages
`reports/04-recurrent-site-audit.md` describes. Same caveat: that report's Deliverable 2 (adding the
`tracing::debug!` calls) was not run either, so this mode also captures nothing at this tip. Built
as the same generic stderr-capture mechanism as `routing-log` rather than a second implementation,
since the plan describes both as "`equality` plus collection of X" with no shape difference besides
what's being grepped for.

### `concurrency`

Launches an HTTP server as a subprocess via a user-supplied `--server-cmd` (the harness does not
hardcode `mistralrs-cli serve` flags -- this session could not verify the exact current flag names
against a built binary, so inventing them risked shipping a wrong command; the caller passes the
full command line instead). Polls `GET /health` until 200 or `--startup-timeout-seconds` elapses.
Builds `N` (`--num-requests`) request bodies, a `--unconstrained-fraction` of them with no grammar
at all and the rest carrying a grammar drawn from `--schema-files` (`identical`: all the same file;
`differing`: round-robin), fires them concurrently via `ThreadPoolExecutor` against
`POST /v1/chat/completions`, scrapes `GET /metrics` immediately before and after the burst, and
reports the deltas for every line matching the `mistralrs_grammar_ff_` prefix (covers
`_splices_staged_total`, `_tokens_fed_total`, `_splice_drops_total{reason=...}`, and
`_tokens_dropped_total{reason=...}` from plan 02, without hardcoding individual metric names so a
future counter is picked up automatically). The server subprocess is terminated in a `finally`
block regardless of outcome.

Not implemented, and explicitly out of scope: plan 06's "batch composition histogram" (dimension 3)
requires a new counter/histogram in `mistralrs-core` that plan 06 itself calls out as "the one code
change this plan permits" -- not this one. `concurrency` mode's report has everything plan 06 needs
that does not require that addition (dimensions 1, 2, and the raw ingredients for 5); dimension 4
(per-batch minimum splice width) is likewise not observable from outside the process without new
instrumentation and is left to plan 06.

## Files changed

- `ff_harness.py` (new, `ff-demo-artifacts`)
- `plans/ff-round-two/fixtures/partial.schema.json` (new, `ff-demo-artifacts`)
- `plans/ff-round-two/reports/05-harness.md` (this report, `ff-demo-artifacts`)
- No changes on `grammar-fast-forward` (none needed; see "Where the harness lives, and why" above)

## Validation performed (no Rust, no model)

All of the following ran in this session, with the exact commands used:

- `python3 -m py_compile ff_harness.py` -- passes.
- `python3 ff_harness.py --help`, `run --help`, `compare --help` -- render correctly; `import
  mistralrs` is deferred into the functions that need it (`build_plain_runner`,
  `build_chat_request`), so argument parsing and `--help` work without the package installed (it is
  not importable in this container -- `import mistralrs` resolves to the source-tree namespace
  package with no compiled extension, confirmed by inspecting `sys.modules`/`__path__`).
- `python3 ff_harness.py run --mode concurrency --flag on --model-id x` (no `--server-cmd`) --
  correctly errors `--server-cmd is required for --mode concurrency` via the `main()` guard.
- Inline unit checks (via a scratch script, not committed) against every pure-Python helper that
  does not require the `mistralrs` extension or a running server: `flag_env` for all three values;
  `report_path` filename shape against the plan's
  `<plan>-<mode>-<flag>-<timestamp>.json` pattern (label variant also checked); `METRIC_LINE_RE`
  against a synthetic Prometheus text body including a labeled counter
  (`..._splice_drops_total{reason="batch_shape"}`) and a non-`mistralrs_grammar_ff_` line, confirming
  the prefix filter and label-preserving key; `metric_deltas` arithmetic including a key present in
  only one snapshot; `compare_reports` for the equal case, the diverging case (correct
  `first_divergent_index`, tails, decoded content), and the "one side returned no token ids"
  case (`comparable: false`, no exception); `capture_stderr` actually captures bytes written to the
  real fd 2 during the `with` block into the temp file.

**Not run, and why:** any path that constructs a `Runner`, calls `send_chat_completion_request`, or
launches a server subprocess. All three require either the compiled `mistralrs` Python extension
(not built in this container) or a `mistralrs-cli`/server binary (requires the Rust toolchain, not
installed here per this task's own instructions not to provision one) plus a downloaded model. The
`token_ids_complete`/greedy-argmax assumption in `equality` mode, the AnyMoE/arm log line capture in
`routing-log`/`arms` (moot at this tip regardless, per above), and the `/metrics` scrape format in
`concurrency` are therefore unverified against a live process and are recorded as such rather than
assumed correct.

## Ready for plan 06?

`concurrency` mode covers what plan 06 needs from the harness (N concurrent requests, mixed
constrained/unconstrained, `/metrics` before/after deltas, on a `--server-cmd` the caller supplies)
for dimensions 1, 2, and the raw numbers behind 5. It does **not** cover dimension 3 (batch
composition histogram) or dimension 4 (per-batch minimum splice width) -- those need the
`mistralrs-core` instrumentation plan 06 reserves for itself, not something plan 05 should have
added. Plan 06 also needs a paged-attention-capable (CUDA/Metal) build to run `concurrency` mode at
all (`paged_attn_supported()` is a compile-time `false` on CPU), which this session cannot produce
or verify. `equality` and its two extensions are otherwise usable by plans 03 and 04 once a
toolchain, a model, and (for `routing-log`/`arms`) the missing `mistralrs-core` log lines exist.

## Environment note

No Rust toolchain (`cargo`/`rustc`) and no built `mistralrs` Python extension or server binary were
present in this container. All of the above was written and validated by reading source
(`mistralrs.pyi` extracted from the committed wheel `mistralrs-0.9.3-cp310-abi3-*.whl`, and
`mistralrs-core`/`mistralrs-server-core` source on `grammar-fast-forward` via `git show`) and by
exercising the harness's pure-Python logic directly, not by running inference.

## Source commit

No commit was made on `grammar-fast-forward` in this session (tip unchanged at `4ba40ca9e`); this
plan's work is entirely the `ff-demo-artifacts` commit associated with this report.
