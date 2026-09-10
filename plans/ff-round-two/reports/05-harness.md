# 05 -- The experimental harness

Status: implemented, locally validated at the Python level (syntax, argument parsing, helper logic,
and a stdlib-`http.server` stub for the HTTP paths). Not run end-to-end against a live model -- no
Rust toolchain, no built `mistralrs-cli`/server binary, and no downloaded model in any session so
far, consistent with every prior report in this plan set.

A later audit of this harness found five problems and they have since been fixed on this branch;
"Audit follow-up" at the end of this document lists them, and the sections below describe the
harness as it stands after those fixes. Nothing in that follow-up changes what is and is not
experimentally verified: still no live inference, no CUDA run, no AnyMoE routing evidence and no
recurrent-site arm counts.

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
- `plans/ff-round-two/fixtures/{narrow,nested,wide}.schema.json` (added by the audit follow-up) --
  the differing-forced-span set plan 06 workload B needs. Same partially-forcing shape, but the
  amount of literal text the grammar forces differs by roughly 9x across the set (structural
  characters: narrow 12, partial 37, nested 72, wide 112). `nested` additionally carries a forced
  run in the middle of the object, where the inner object closes and the next key begins, rather
  than only at the head.

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

`equality` plus best-effort capture: `os.dup2` redirects the real OS-level fds 1 *and* 2 (not
Python's `sys.stdout`/`sys.stderr` objects -- required because Rust's tracing subscriber writes to
real fds) to temp files for the duration of the request, then greps them against a regex (default
`EXPERT_IDX`, overridable via `--capture-pattern`) and stores the matched lines, each tagged with
the stream it came from, in the report's `capture` object.

Both streams, not just stderr, because which one the subscriber uses is *not* pinned down by
anything readable here: `initialize_logging` builds `tracing_subscriber::fmt()` and never calls
`.with_writer(...)` (`mistralrs-core/src/utils/debug.rs:47`), so the destination is that crate's
default `MakeWriter`, and `tracing-subscriber` 0.3.22's source is not vendored in this checkout
(no `cargo`, no registry, no `vendor/`) to confirm which fd that is. Capturing both makes the
question moot rather than guessed.

Matched lines are kept verbatim under `capture.lines`, each tagged with the stream it came from,
and aggregated under `capture.aggregate`: the emission-ordered expert sequence per `layer=L,row=R`
plus its collapsed counts, which is plan 05's "keyed by layer and batch row". `compare` diffs that
aggregate alongside the token ids, naming the first forward at which the two runs chose different
experts, and reports which row of plan 03's outcome table the pair lands on.

Two numbers in the report make a null result readable: `capture.lines_seen` (per stream) and
`capture.total_lines_seen`, alongside `capture.num_matched_lines`. Zero matches out of zero lines
seen means the capture never saw output at all; zero matches out of a few thousand lines means the
instrumentation genuinely emitted nothing matching. The `logging` block records `RUST_LOG`,
`MISTRALRS_DEBUG` and whether they were set before the `import mistralrs` that fixes the tracing
filter in a `OnceLock` (`mistralrs-pyo3/src/lib.rs:3223` calls `initialize_logging()` from the
`#[pymodule]` initialiser, so a later change to those variables does nothing).

The capture modes set `MISTRALRS_DEBUG=1` themselves, overridable with `--rust-log`. Without it the
filter is `warn` plus `mistralrs=info` (`utils/debug.rs:29-62`) and plan 03's routing line and plan
04's nine arm messages -- both `tracing::debug!` -- would be dropped before reaching any fd.

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

Same shape as `routing-log` (both streams, debug logging on, same `lines_seen` accounting, raw lines
kept), except the aggregate is plan 05's count per site per arm -- `capture.aggregate.counts[site][arm]`
-- and `compare` reports which site/arm counts moved between the two runs. Default pattern
`\bARM\b`, for the nine recurrent-site arm messages
`reports/04-recurrent-site-audit.md` describes. Same caveat: that report's Deliverable 2 (adding the
`tracing::debug!` calls) was not run either, so this mode also captures nothing at this tip. Built
as the same generic stderr-capture mechanism as `routing-log` rather than a second implementation,
since the plan describes both as "`equality` plus collection of X" with no shape difference besides
what's being grepped for.

### `concurrency`

Launches an HTTP server as a subprocess via a user-supplied `--server-cmd` (the harness does not
hardcode `mistralrs-cli serve` flags -- no session so far could verify the exact current flag names
against a built binary, so inventing them risked shipping a wrong command; the caller passes the
full command line instead, as one shell-quoted string that the harness splits with `shlex`). Polls
`GET /health` until 200 or `--startup-timeout-seconds` elapses.

`plan_concurrency_requests` then derives the whole schedule from `--seed`: which of the `N`
(`--num-requests`) slots go unconstrained (sampled, so they are not always the first ones submitted)
and which fixture each constrained slot carries -- `identical` uses the first file, `differing`
round-robins over a seeded shuffle of `--schema-files`, so every slot gets a distinct schema while
`N` fits the fixture set. `--stagger-seconds` delays request `i` by `i *` that value, covering plan
06's "stagger request start times in at least one variant"; at 0 the start is synchronised as
before. Requests fire concurrently via `ThreadPoolExecutor` against `POST /v1/chat/completions`.

Request bodies use the HTTP route's own shape, which is *not* the Python API's:
`mistralrs-server-core/src/openai.rs` has no `grammar_type` field and its `grammar` is
`#[serde(tag = "type", content = "value")]`, so the schema goes out as
`{"type": "json_schema", "value": {...}}`. `seed` is a real field on that struct (`openai.rs:1180`,
covered by its own test at `:2139`), so the run's seed reaches the server rather than only shaping
the client-side schedule.

`GET /metrics` is scraped immediately before and after the burst and the deltas reported for every
sample whose name starts with one of `--metric-prefix` (default `mistralrs_`). That default covers
plan 02's `mistralrs_grammar_ff_*` counters for dimensions 1-2 *and*
`mistralrs_decode_tokens_processed_total` / `mistralrs_prefill_tokens_processed_total` for dimension
5 and `mistralrs_paged_preemptions_total` / `mistralrs_kv_cache_blocks_*` for dimension 6. Label
sets stay in the sample key, so `_splice_drops_total{reason="batch_shape"}` remains separable from
the other drop reasons, which dimension 1 requires. Passing `--metric-prefix
mistralrs_grammar_ff_` narrows back to the fast-forward counters; passing `--metric-prefix` with no
value keeps everything. Individual metric names are never hardcoded, so a future counter is picked
up automatically. The server subprocess is terminated in a `finally` block regardless of outcome.

The report records the parameters needed to re-run it: seed (and the seed sent on the wire),
prompt, `unconstrained_fraction`, `stagger_seconds`, `schema_set`, the resolved `schema_files`, the
metric prefixes, `max_tokens`, the startup timeout, and both the argv list and the raw string of
`--server-cmd`. Per request it records the schema file used, the scheduled offset and the observed
start offset alongside latency, status and finish reason.

Not implemented, and explicitly out of scope: plan 06's "batch composition histogram" (dimension 3)
requires a new counter/histogram in `mistralrs-core` that plan 06 itself calls out as "the one code
change this plan permits" -- not this one. `concurrency` mode's report has everything plan 06 needs
that does not require that addition (dimensions 1, 2, and the raw ingredients for 5); dimension 4
(per-batch minimum splice width) is likewise not observable from outside the process without new
instrumentation and is left to plan 06.

Also still not enforced: plan 06 needs a build where `paged_attn_supported()` is true (CUDA or
Metal). The harness records `server_cmd` so a reader can tell what was launched, but it does not
check that the server it talked to was a paged build.

## Files changed

- `ff_harness.py` (new, `ff-demo-artifacts`)
- `test_ff_harness.py` (added by the audit follow-up, `ff-demo-artifacts`)
- `plans/ff-round-two/fixtures/partial.schema.json` (new, `ff-demo-artifacts`)
- `plans/ff-round-two/fixtures/{narrow,nested,wide}.schema.json` (audit follow-up,
  `ff-demo-artifacts`)
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
  correctly errors `--server-cmd is required for --mode concurrency` via the `main()` guard, which
  now also validates the command splits, the schema set can differ, and the request plan is
  constructible, all before anything is launched.
- `python3 -m unittest test_ff_harness` -- 94 tests, all passing, stdlib only (no pytest in the
  container). They cover: `parse_server_cmd` on a flagged command line and on quoted arguments;
  `parse_metrics_body` against a synthetic Prometheus body carrying the fast-forward counters, the
  token counters, the preemption and KV gauges, a labelled counter, a comment line, junk, and a
  non-`mistralrs_` family, under the default prefix, a narrowed prefix and an empty prefix list;
  `metric_deltas` including a key present in only one snapshot; `resolve_schema_files` for the
  differing/identical defaults, the one-fixture and duplicate-fixture rejections, and a missing
  file; the four fixtures' JSON validity, partially-forcing shape and distinct forced-span budgets;
  `build_concurrency_request_body`'s tagged grammar object and optional seed;
  `plan_concurrency_requests` for seed reproducibility, seed sensitivity, the unconstrained count
  and its non-head-biased placement, distinct-schema assignment, round-robin overflow, stagger
  offsets and parameter validation; `capture_std_streams` capturing both real fds and restoring
  them on the normal and exception paths; `scan_captured_streams` distinguishing zero-of-zero from
  zero-of-many; `configure_capture_logging` across the capture and non-capture modes, an explicit
  `--rust-log`, an operator-set `MISTRALRS_DEBUG`, and the already-imported hazard flag.
- For F5: `parse_tracing_fields` on quoted, bare, numeric and escaped values, on a field-less line,
  and on the shape `sampling.rs:1636` actually emits; arm aggregation per site per arm across all
  nine site identifiers, with wrong field names surfacing as unparsed rather than as an empty
  result; routing aggregation keyed by layer and row with emission order preserved; and `compare`
  over each row of plan 03's outcome table -- identical/identical, identical/differ (asserting the
  first divergent forward and both tails), differ/either -- plus keys seen in only one run, arm
  count deltas, a report with no capture at all, a capture diff surviving missing token ids, raw
  lines surviving alongside the aggregate, and the markdown summary naming the divergence.
- For F8: `build_anymoe_config` over the `FineTuned` string and object forms, `LoraAdapter` with its
  three fields, forgiving variant spellings, and rejection of an unknown variant, stray or missing
  variant fields, a missing required key, a mistyped optional key, and non-object inputs -- driven
  through a stub module shaped like the pyo3 classes, plus `AnyMoeApiSurfaceTest`, which parses
  `mistralrs/__init__.pyi` out of the wheel committed on this branch and asserts the harness's
  required/optional key lists and `LoraAdapter` field names still match the shipped signature. That
  last test was checked against two deliberate perturbations of the key lists and fails on both, so
  it is not passing vacuously.
- Four of those tests run against a stdlib `http.server` stub on a loopback port -- still no model
  and no `mistralrs` extension -- confirming `wait_for_health`, `scrape_metrics` over a real socket,
  that a staggered burst starts in order and still overlaps (four 0.25 s requests finishing in
  under 0.9 s), that the server receives the tagged grammar object and the seed, and that a
  connection failure is recorded on the result rather than raised.
- `python3 ff_harness.py run --mode concurrency ... --server-cmd "mistralrs serve -p 1234 -m foo"`
  -- reaches `Popen` and fails only with `FileNotFoundError: 'mistralrs'`, i.e. the flagged command
  line now parses.

**Not run, and why:** any path that constructs a `Runner`, calls `send_chat_completion_request`, or
launches a server subprocess. All three require either the compiled `mistralrs` Python extension
(not built in this container) or a `mistralrs-cli`/server binary (requires the Rust toolchain, not
installed here per this task's own instructions not to provision one) plus a downloaded model. The
`token_ids_complete`/greedy-argmax assumption in `equality` mode, the AnyMoE/arm log line capture in
`routing-log`/`arms` (moot at this tip regardless, per above), and the `/metrics` scrape format in
`concurrency` are therefore unverified against a live process and are recorded as such rather than
assumed correct. The stub server exercises the request/metrics *shapes* the harness sends and
parses; it does not verify that mistral.rs accepts them, only that they match what
`mistralrs-server-core/src/openai.rs` and `metrics.rs` say on this branch.

## Ready for plan 06?

`concurrency` mode covers what plan 06 needs from the harness (N concurrent requests, mixed
constrained/unconstrained, identical or genuinely differing schema sets, a seeded and optionally
staggered schedule, `/metrics` before/after deltas across the whole `mistralrs_` namespace, on a
`--server-cmd` the caller supplies) for dimensions 1, 2, 5 and the recorded inputs behind 6. It does **not** cover dimension 3 (batch
composition histogram) or dimension 4 (per-batch minimum splice width) -- those need the
`mistralrs-core` instrumentation plan 06 reserves for itself, not something plan 05 should have
added. Plan 06 also needs a paged-attention-capable (CUDA/Metal) build to run `concurrency` mode at
all (`paged_attn_supported()` is a compile-time `false` on CPU), which no session so far can produce
or verify. `equality` and its two extensions are otherwise usable by plans 03 and 04 once a
toolchain, a model, and (for `routing-log`/`arms`) the missing `mistralrs-core` log lines exist.

## Audit follow-up

A later session audited this harness against plan 05 and returned `PASS WITH ISSUES`. Seven
findings were fixed on `ff-demo-artifacts` across two follow-up rounds; no Rust source was touched
and no plan 06 work was done.

| Finding | Problem | Fix |
|---|---|---|
| F1 | `--server-cmd` used `nargs="+"`, which stops at the first `-`, so `mistralrs serve -p 1234 -m foo` could not be passed at all and concurrency mode could not launch a server | one shell-quoted string split with `shlex`, validated at parse time |
| F4 | metrics filtered to a hardcoded `mistralrs_grammar_ff_` prefix, dropping the token, preemption and KV metrics plan 06 dimensions 5 and 6 read | `--metric-prefix`, defaulting to the whole `mistralrs_` namespace, narrowable and disableable |
| F6 | `--schema-set differing` silently round-robined one fixture, producing workload A while the report said workload B | a four-fixture default set, and a hard error on any explicit set with fewer than two distinct files |
| F7 | `--seed` never reached concurrency mode, unconstrained requests were always the first slots, and none of the schedule was recorded | seeded schedule, `--stagger-seconds`, and the full parameter set in the report |
| F2 + F3 | capture watched fd 2 only and left logging at `info`, so a run that never had a chance to see a `tracing::debug!` line looked identical to a real negative | capture both fds, set `MISTRALRS_DEBUG=1` for the capture modes before the import that fixes the filter, record the logging config and the lines-seen totals |
| F5 | `routing-log`/`arms` kept only a flat list of matched lines and `compare` diffed token ids alone, so plan 05's "count per site per arm" and "diff routing as well as tokens" were unimplemented | parse the trailing tracing `key=value` fields, aggregate per site per arm and per layer/row, diff both in `compare`, keep the raw lines |
| F8 | `AnyMoeConfig(**json)` could never construct the `expert_type` pyo3 complex enum, so `--anymoe-config-json` raised `TypeError` for plan 03 Part B's only use of it | name the variant in JSON and construct `AnyMoeExpertType.FineTuned()` / `.LoraAdapter(...)`, with the key lists checked against the shipped stub |

One further defect was fixed because F6 and F7 are meaningless without it: the concurrency request
body sent the Python API's `grammar_type` + string `grammar`, which the HTTP route cannot
deserialise, so every constrained request would have been rejected before reaching the model.

Deliberately **not** done, per the audit's own ranking and the follow-up's scope: the
reproducibility extras -- build identity, commit, argv, schema hash -- and the smaller robustness
items (F9). Those remain open.

### What F5 does and does not claim

The aggregation consumes a rendering that provably exists on this branch -- `tracing_subscriber::fmt`
writes structured fields as trailing `key=value`, as at `sampling.rs:1636`
(`tracing::debug!(splice_len = splice.len(), ...)`) and `sequence.rs:1269` (`error = %e`). What does
*not* exist yet is anything to parse: plan 03's routing line and plan 04's nine arm messages are
both still unwritten. The field **names** are therefore CLI options (`--arm-site-field`,
`--arm-field`, `--routing-layer-field`, `--routing-row-field`, `--routing-expert-field`) with
documented defaults, not assumptions baked into the parser, and matched-but-unparsed lines are
counted and sampled in the report so a name mismatch shows up instead of reading as a clean zero.

For the same reason `compare` refuses to call two empty captures "equal": at this tip that is
"nothing to diff", and reporting it as agreement would manufacture exactly the plan 03 table row
("tokens identical, expert indices identical") that the plan says is the only one supporting a
re-enable. When there *are* observations, `compare` names the row reached and exits nonzero on
routing or arm divergence as well as on token divergence.

One more thing worth recording for whoever writes plan 03's instrumentation: `MoeMlp::forward`
(`amoe/mod.rs:258-284`) has no layer index in scope at all, so "keyed by layer" needs a layer id
threaded into `MoeMlp` before a log line can carry one. The harness will key on whatever field the
line does carry; it cannot invent the layer.

## Environment note

No Rust toolchain (`cargo`/`rustc`) and no built `mistralrs` Python extension or server binary were
present in this container. All of the above was written and validated by reading source
(`mistralrs.pyi` extracted from the committed wheel `mistralrs-0.9.3-cp310-abi3-*.whl`, and
`mistralrs-core`/`mistralrs-server-core`/`mistralrs-pyo3` source on `grammar-fast-forward`) and by
exercising the harness's pure-Python logic directly, not by running inference. One thing that could
*not* be settled by reading: `tracing-subscriber` 0.3.22 is a registry dependency with no vendored
copy here, so its default `MakeWriter` (stdout or stderr) is unverified -- which is why the capture
takes both fds rather than picking one.

## Source commit

No commit was made on `grammar-fast-forward` in this session (tip unchanged at `4ba40ca9e`); this
plan's work is entirely the `ff-demo-artifacts` commit associated with this report.
