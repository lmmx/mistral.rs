# 06-D -- source-level audit of `llguidance::consume_ff_tokens()` semantics (2026-09-10, eighth session)

Status: **the equivalence question mistralrs's own usage depends on is answered, precisely, from
source: llguidance computes the *content* of a forced splice via the exact same code path used by
ordinary per-token masked sampling, so the two paths are provably forced to agree on which token(s)
get emitted at a given grammar position. This decisively narrows the failure away from "llguidance
picks the wrong token at a nesting boundary" and toward mistralrs's own handling of the widened
decode step that replays a staged splice -- outside this audit's scope (llguidance only), and outside
this session's mandate (no mistralrs source changes, no Plan 07). No Rust source modified anywhere.
No rebuild. No live server requests.**

## 0. Correction to a claim in `06-c-root-cause.md`

`06-c-root-cause.md` Sec 1 and Sec 7 state this container has "no network access to fetch [llguidance's]
source." That was re-tested this session and is **not accurate for this container**: `curl`/`wget`/`git`
are present and outbound HTTPS to `crates.io`, `docs.rs`, `api.github.com`, and
`raw.githubusercontent.com` all succeeded (crates.io's crate-download endpoint alone returned `403`;
GitHub's raw/API endpoints did not). This audit used GitHub's raw source at the exact tagged commit
for the resolved crate version (see Sec 1) -- a handful of small file fetches (`matcher.rs`,
`tokenparser.rs`, `earley/parser.rs`, `api.rs`, `CHANGELOG.md`, one test file; no `cargo` invocation,
no dependency tree, no build). This satisfies the task's "cheap and already supported" bar; nothing
expensive was fetched or built. Per the memory instructions this correction is recorded here (not as
a persistent memory, since it is scoped entirely to this container instance) so it does not silently
contradict the prior report.

## 1. Exact llguidance version/revision

- `Cargo.toml:146` (workspace root): `llguidance = { version = "1.2.0", default-features = false,
  features = ["lark"] }` -- a caret requirement (`>=1.2.0, <2.0.0`), not a pin.
- `Cargo.lock:3385-3392` (resolved): `name = "llguidance"`, `version = "1.4.0"`, `source =
  "registry+https://github.com/rust-lang/crates.io-index"`, `checksum =
  "ebdf44cac6f32a127a275da89f542a358baad09938f8c01caa19241d8d26dcf7"`, deps `anyhow, derivre,
  indexmap 2.13.0, regex-syntax, serde, serde_json, toktrie`.
- **The checkout actually builds against 1.4.0**, not 1.2.0 -- confirming, not contradicting, the
  prior report's parenthetical ("workspace-pinned to 1.2.0, referenced as 1.4.0 in last year's
  review"); 1.4.0 is simply what `>=1.2.0` currently resolves to with this lockfile.
- GitHub tag `v1.4.0` in `guidance-ai/llguidance` resolves to commit `c1b69da46e2053ce0d40ff4416b6a860f7165143`.
  `parser/Cargo.toml` at that commit reads `name = "llguidance"`, `version = "1.4.0"` -- confirms the
  crates.io package `llguidance` is published from the `parser/` subdirectory of this monorepo, and
  that this commit is the correct source snapshot for the resolved dependency. All line numbers below
  are against that commit's `parser/src/*.rs`.

## 2. Exact `consume_ff_tokens()` source locations

There are **two** distinct `consume_ff_tokens`, at two layers. mistralrs calls the outer one.

- **`Matcher::consume_ff_tokens`** -- `parser/src/matcher.rs:146-152`:
  ```rust
  pub fn consume_ff_tokens(&mut self) -> Vec<TokenId> {
      let toks = self.compute_ff_tokens();
      if !toks.is_empty() {
          let _ = self.consume_tokens(&toks);
      }
      toks
  }
  ```
  This is the type mistralrs's `llg: Matcher` (per `sampling.rs:1629-1635` in `/workspace`) actually
  calls. `Matcher::compute_ff_tokens` (`matcher.rs:141-144`) delegates to
  `TokenParser::compute_ff_tokens`; `Matcher::consume_tokens` (`matcher.rs:74-83`) loops calling
  `TokenParser::consume_token` once per returned token id, then calls `TokenParser::check_stop()`
  once at the end.

- **`TokenParser::consume_ff_tokens`** -- `parser/src/tokenparser.rs:884-896` -- a lower-level,
  separate implementation with the same shape (compute, then loop-consume via `self.consume_token`),
  documented `/// Compute and then consume fast-forward tokens.` This one is **not** what mistralrs
  reaches (mistralrs holds a `Matcher`, not a raw `TokenParser`), but it independently confirms the
  same "compute via `ff_tokens()`, then consume each token through the ordinary single-token path"
  shape at the layer beneath `Matcher`.

Both layers bottom out in the same primitives: `TokenParser::compute_ff_tokens`
(`tokenparser.rs:875-881`) calls `self.ff_tokens()` (`tokenparser.rs:677-754`), and consumption of
whatever it returns goes through `TokenParser::consume_token` (`tokenparser.rs:792-833`) --
**the identical function used for every ordinary, per-step sampled token**, with identical
`apply_token` (`tokenparser.rs:513-...`), identical backtrack handling, identical error handling.
There is no separate/relaxed "bulk consume" code path anywhere in this crate; a forced-byte-derived
splice and a model-sampled token are consumed by the exact same three-line loop
(`matcher.rs:75-79`: `for &t in tokens { inner.parser.consume_token(t)?; ensure!(bt == 0, ...) }`).

## 3. Ordinary `consume_token()` path (for comparison)

`Matcher::consume_token` (`matcher.rs:85-87`) is a one-element call into `Matcher::consume_tokens`
(`matcher.rs:74-83`) -- the exact same function the FF splice replay uses internally, just with a
one-token slice instead of an N-token slice. Underneath, `TokenParser::consume_token`
(`tokenparser.rs:792-833`) does bookkeeping (`max_tokens_total`, EOS handling) and calls
`apply_token` (`tokenparser.rs:513-609`), which pushes the token into `llm_tokens`, decodes its bytes
via the token trie, and calls into the Earley parser's own `apply_token`
(`earley/parser.rs:2866-2870`, itself calling `ParserState::apply_token` under a lock via
`with_shared`). **This is identical machinery for every token, forced or not.** The only branch
point in `consume_token` that could behave differently per-call is backtracking
(`apply_token`'s `backtrack_bytes0 != 0` branch, `tokenparser.rs:581-624`), which both the FF splice
path and ordinary path handle the same way, and which mistralrs's own `consume_tokens` wrapper in
the FF case explicitly asserts against (`ensure!(bt == 0, "unexpected backtracking")`,
`matcher.rs:78`) -- consistent with `06-c`'s finding that this run's splice was cleanly staged and
fed with no discard.

## 4. The key structural fact: ordinary mask computation *also* consults `ff_tokens()`

This is the finding that reframes the investigation. `TokenParser::compute_mask_inner`
(`tokenparser.rs:463-507`), which backs `Matcher::compute_mask`/`compute_mask_or_eos`
(`matcher.rs:102-118`) -- the function mistralrs calls **on every decode step, in both FF-ON and
FF-OFF configurations**, per `06-c-root-cause.md` Sec 3's own citation of `sampling.rs:1561-1571`
(`llg.compute_mask_or_eos()`) -- itself computes `ff_tokens()` first:

```rust
// tokenparser.rs:463-482
fn compute_mask_inner(&mut self) -> Result<SimpleVob> {
    ...
    let prefix = if self.can_force_bytes() {
        let (ff_tokens, token_prefix) = self
            .ff_tokens_cache
            .take()
            .unwrap_or_else(|| self.ff_tokens());
        if !ff_tokens.is_empty() {
            let t = ff_tokens[0];
            infoln!(self, "forcing ff_token by mask: {}", t);
            let mask = self.tok_trie().singleton_token_set(t);
            self.last_step_stats = ParserStats::default();
            return Ok(mask);
        } else {
            token_prefix
        }
    } else { ... };
    ...
}
```

`self.ff_tokens()` here (`tokenparser.rs:677-754`) is the **exact same private method**
`compute_ff_tokens()`/`consume_ff_tokens()` call (Sec 2 above). So: whenever a position is
byte-forceable, `compute_mask_inner` collapses the returned mask to a **singleton set containing
exactly `ff_tokens[0]`** -- the model still runs a real forward pass and "samples," but under a mask
that permits only one token, so it is forced to emit that same token regardless. This branch is
**not gated by mistralrs's fast-forward flag at all** -- it is unconditional behavior of
`compute_mask()`, which mistralrs calls every step independent of whether it additionally invokes
`consume_ff_tokens()` afterward (`06-c` Sec 3, `sampling.rs:1629-1645`: `consume_token` +
conditionally `consume_ff_tokens`, but the *preceding* step's `compute_mask_or_eos()` call at
`sampling.rs:1561` runs unconditionally either way).

**Consequence:** at any single grammar position, FF-ON (splice, skip the forward pass, force the
token via `consume_ff_tokens`) and FF-OFF (real forward pass, mask collapsed to the same singleton by
`compute_mask_inner`, model has no choice but to emit it) are guaranteed by construction to force the
**identical token id**, given identical preceding grammar/matcher state. They differ only in whether
a dedicated forward pass is spent computing KV for that position before the next mask is computed --
a mistralrs/model-serving concern that `llguidance` has no visibility into or opinion on (this crate
has no concept of "forward pass" or "KV cache" anywhere in `matcher.rs`/`tokenparser.rs`).

## 5. Contract / precondition of `consume_ff_tokens()`

From Sec 2-4, the contract is not "documented" in doc comments beyond the one-line
`/// Compute and then consume fast-forward tokens.` (`tokenparser.rs:883`) and
`/// Check if there are any tokens to fast-forward, forced by the current parser state.`
(`tokenparser.rs:873-874`), but it is precisely *inferable* from the shared-implementation fact in
Sec 4:

- **Content equivalence is guaranteed by construction**, not merely "documented": the token(s)
  `consume_ff_tokens()` returns are exactly the token(s) that an ordinary `compute_mask()` +
  masked-sample + `consume_token()` step would independently arrive at for the same matcher state,
  because both call sites resolve to the same `ff_tokens()` computation. This answers the audit's
  central question for the *token-selection* half of it: **yes, equivalent, for what gets chosen**,
  not merely "assumed equivalent."
- **The precondition that is *not* enforced inside `llguidance` at all**: nothing in `Matcher` or
  `TokenParser` tracks, requires, or waits for the caller to have obtained real model computation
  (logits, KV cache entries) for the forced token(s) before advancing its own internal parser state.
  `Matcher::consume_ff_tokens` (Sec 2) synchronously advances the matcher the instant it is called --
  matching `06-c`'s inherited citation. `06-c` treated this fact as established but unverified this
  session; **it is now independently re-verified directly from source** (Sec 2 above), not merely
  inherited.
- This makes the crate's implicit precondition on its caller: *"if you skip a forward pass and use
  `consume_ff_tokens()` to synchronously advance grammar state ahead of the model, you (the caller)
  are responsible for ensuring the model eventually computes correct, correctly-positioned KV state
  for those tokens before you rely on any subsequent logits for real sampling."* `llguidance` cannot
  check this; it has no model-serving concepts. This matches the task's conclusion category **1**:
  *"`consume_ff_tokens()` is explicitly designed to be equivalent [in token content], but mistralrs
  violates a precondition [about downstream KV/logit bookkeeping]"* -- not category 3 (llguidance
  defect) or category 2 (mistralrs misusing an API with different/weaker semantics than assumed; the
  semantics are, per Sec 4, exactly as strong as assumed for token content).

## 6. JSON nesting-transition analysis: no nesting-aware code in the forcing mechanism

The only mechanism that decides *what* is forceable is `forced_byte()`
(`earley/parser.rs:1659-1725`), called in a loop by `force_bytes()`
(`earley/parser.rs:1408-1415`, and the outer wrapper at `earley/parser.rs:2827-2845`):

- `forced_byte()` returns `None` immediately if `self.is_accepting()` (`earley/parser.rs:1663-1666`)
  -- a genuinely-completable parse state is never forced past.
- Otherwise it does a fast check via the lexer's `next_byte()` hint, and failing a definitive answer,
  falls back to a **brute-force, speculative-execution search over all 256 byte values**
  (`earley/parser.rs:1696-1714`): for each candidate byte `b`, it actually pushes `b` onto a
  speculative copy of parser state (`r.try_push_byte(b)`) and pops it back off, counting how many of
  the 256 values are accepted. A byte is "forced" only if **exactly one** value is accepted
  (`byte_sym.is_some()` guard returns `None` the moment a *second* accepted byte is found, line
  1703-1706).
- This routine is generic over the compiled grammar automaton (`ParserRecognizer` /
  `try_push_byte`/`pop_bytes`) and contains **no reference to JSON, object/array nesting, lexer
  stack depth, or nonterminal identity** anywhere in its body. It is the same primitive regardless of
  whether the current position is inside a top-level scalar property, a nested object's property, or
  any other grammar construct.
- `currently_forced_bytes()` (`earley/parser.rs:2921-2923`) and `ff_tokens()`
  (`tokenparser.rs:677-754`, converting forced bytes to token ids via re-tokenization,
  `tokenize_bytes_marker`/`tokenize_and_chop`) are likewise generic over grammar shape; nesting depth
  only affects the automaton `try_push_byte` walks against, not the forcing algorithm itself.

**This directly contradicts `06-c-root-cause.md` Sec 5 candidate 2** ("the matcher's parser-stack
position after the splice does not actually match the stated contract... forgetting to push a stack
frame for the new nesting level while still reporting the byte/token as consumed"): there is no
stack-frame-specific logic in this forcing path to fail to execute. `consume_tokens`
(`matcher.rs:74-83`) calls the ordinary `parser.consume_token(t)` for every forced token, which goes
through the full Earley `apply_token` (Sec 3) -- the same state-advancement code used for every
non-forced token, including ones that themselves open nested objects. Nothing here is skipped or
special-cased at a nesting boundary. This candidate should be considered ruled out by direct source
reading, not merely unresolved.

`06-c` Sec 5 candidate 1 ("the splice's token content is itself wrong... a whitespace token that the
grammar's canonicalized JSON-formatting rules treat as forced but that does not actually correspond
to a safe, resumable parser position for a nested object") is **also weakened** by Sec 4 above: if
the *content* of the forced token were wrong specifically for the nested-object transition, the
identical wrong token would also be forced onto FF-OFF via `compute_mask_inner`'s singleton mask at
the same grammar position (same `ff_tokens()` call, same input state) -- and FF-OFF's output at this
exact fixture is confirmed correct 5/5 (`06-b-n1-repro.md`). A content-only defect that is wrong in
FF-ON but not in FF-OFF, while both compute the forced token through the same function given the same
preceding grammar state, is not a coherent failure mode.

## 7. The failing `"run": { ... }` case, in light of Sec 4 and 6

`06-c` established: exactly one splice, width 1 token, staged at some point between the `"run":` key
(itself produced by ordinary masked sampling, since 62/63 attempts were `empty_splice`) and the point
where generation degenerates. Given Sec 4-6:

- Whatever single token `consume_ff_tokens()` forced at that point is **provably the same token**
  FF-OFF's `compute_mask_inner` would have (and, per the correct FF-OFF output, did) force via its
  singleton mask at the identical grammar position. The forced token's *content* is therefore not a
  viable locus for this bug -- it is shared code, shared input state, and the shared-path output
  (FF-OFF) is correct.
- The two configurations diverge in exactly one respect that `llguidance` has no visibility into or
  responsibility for: whether/how the model's forward pass computes real, correctly-positioned KV
  state for that forced token before the *next* real (non-forced) sampling decision is made. FF-OFF
  pays for a dedicated forward pass at that position before ever computing a further mask. FF-ON, by
  design, defers that computation into mistralrs's widened decode window
  (`06-c` Sec 3: `PagedAttentionScheduler::completion_token_cost`,
  `paged_attention/scheduler.rs:540-545`; `num_computed_tokens` advancement,
  `engine/mod.rs:2000-2008, 2109-2117`) -- machinery `06-c` Sec 7 explicitly flagged as **not fully
  traced** ("I could not confirm whether that path is unconditionally excluded for any
  grammar-constrained sequence... `account_cuda_decode_rows`/`continue_cuda_decode_batch`").
- This session's findings make that untraced mistralrs-side window/row-accounting path -- specifically
  whether the **first real (non-forced) sampling decision immediately following a staged splice**
  reads its logits from the KV/row position that actually corresponds to *after* the spliced token,
  rather than some other row in the widened batch -- the single most concrete remaining candidate.
  A wrong-row read would explain the observed symptom precisely: the model's real sampling decision
  right after `"run": ` (whitespace) would be conditioned on logits computed for the wrong context,
  producing an out-of-distribution argmax at temperature 0 that could plausibly and deterministically
  land on a degenerate repeated token (`\r`) rather than the schema-required `{`, and -- since the
  matcher's own state was already correctly advanced past the (correct) forced token independent of
  this -- the grammar mask on subsequent steps would still legitimately allow whitespace-class bytes
  (ordinary JSON permits `\s*` before a value), letting the mistaken `\r` preference repeat forever
  without the mask itself ever forbidding it. This is consistent with, and narrower than, `06-c` Sec
  6's "too-permissive mask" explanation -- the mask permissiveness there is ordinary/expected JSON
  grammar behavior, not a bug; the new candidate is *which row of logits* gets sampled under that
  (correctly permissive) mask.
- I did not, and could not within this session's llguidance-only scope, instrument or trace the
  mistralrs-side widened-decode row-selection code to confirm this directly. It is the outcome this
  audit points to, not a proven finding.

## 8. Conclusion: which of the four categories

- **Not** a `llguidance` defect (category 3): Sec 4 and 6 show, by direct code reading, that forced
  token content is shared/identical between FF-ON and FF-OFF at the same grammar position, and that
  the byte-forcing algorithm contains no nesting-aware logic that could selectively misbehave for
  nested-object transitions. Nothing found in `parser/src/matcher.rs`, `tokenparser.rs`, or
  `earley/parser.rs`'s forcing/consumption machinery is inconsistent with its own use elsewhere in
  the same crate (including the crate's own `sample_parser` test suite, which exercises a
  structurally similar nested-object schema for **validation** correctness --
  `sample_parser/tests/test_json_objects.rs:49-60`'s `NESTED_SCHEMA` -- though those tests check
  schema-acceptance, not `ff_tokens()`/forced-splice content specifically, so they do not directly
  cover this code path).
- **Best current fit: category 1** -- `consume_ff_tokens()` is (by construction, not merely
  documentation) equivalent to ordinary per-token consumption for *what* gets consumed; the defect
  most plausibly lives in mistralrs's precondition-upholding responsibility for *when/how the model
  actually computes KV state* for a token whose grammar-consumption was made synchronous ahead of
  that computation. This is outside `llguidance`'s crate boundary entirely -- it has no model,
  no KV cache, no notion of a "decode step."
- Residual uncertainty: this audit did not instrument the actual widened-decode-window
  logits/row-selection code in mistralrs (out of scope: no Rust source changes, no rebuild). Category
  4 ("insufficient evidence to distinguish") would be the honest fallback specifically for *that*
  final link in the causal chain -- everything upstream of it (llguidance's token-selection
  equivalence) is now resolved with high confidence from source, not inferred.

## 9. Narrowest useful next experiment

Two independent, small, source-preserving options, in order of cost:

1. **Llguidance-only, no GPU, no mistralrs**: build a tiny standalone harness (new script/crate, not
   a change to `mistralrs-core` or the vendored `llguidance` source) using `llguidance`'s own public
   `Matcher`/`TokenParser` API against `plans/ff-round-two/fixtures/nested.schema.json`'s compiled
   grammar and this model's real tokenizer, replaying the exact token prefix
   `{"run":` (established in `06-c`/`06-b`) and calling `compute_ff_tokens()` directly to print the
   exact forced token id(s)/bytes at that point, with no model in the loop. This would convert
   Sec 4-7's structural argument (forced content must be correct because it's shared with FF-OFF)
   into a directly observed value, closing the last bit of inference in this audit at near-zero cost
   (crate compiles standalone; `parser/Cargo.toml` shows no GPU/mistralrs dependency).
2. **mistralrs-side instrumentation (requires the Rust source change and rebuild this session was
   told to avoid)**: log the KV/row index and the resulting sampled token id used for the first real
   (non-forced) decode step immediately following a staged splice, and compare it against a control
   step where the equivalent token was produced via the ordinary FF-OFF masked path. This is the
   experiment that would actually confirm or refute Sec 7's leading candidate, but it requires
   exactly the kind of instrumented rebuild and live-model run this investigation is scoped to defer
   to a later, explicitly-authorized session (not Plan 07 -- this is narrower and mistralrs-specific,
   not a new investigation phase).

Neither was run this session; both are recorded as next steps only.

## 10. What evidence would be required before implementing a fix

- Direct confirmation (experiment 1 above, or equivalent) that the forced token(s) at the
  `"run": {` boundary, for this exact fixture and tokenizer, decode to bytes that are actually a
  correct, schema-consistent continuation (i.e., ruling out, not just structurally arguing against,
  a content-level defect).
- Direct confirmation (experiment 2 above) of which KV/logit row mistralrs's widened decode step
  actually samples from for the first real token after a splice, and whether it is the row
  corresponding to the position immediately after the spliced token -- specifically for a splice that
  immediately precedes a grammar-forced entry into a deeper nesting level, since that is the only
  structural trigger established so far (`06-c` Sec 8: the three flat fixtures never reproduced this).
- If evidence points at mistralrs's row-accounting: a minimal, reviewable diff isolated to that one
  accounting path, verifiable against exactly `06-b-n1-repro.md`'s reproduction (N=1,
  `nested.schema.json`, seed 42, 5-repeat determinism check) before being declared correct, per `06-c`
  Sec 8's existing invariant list, which this audit found no reason to revise.
- If evidence instead points at `llguidance` itself (not the leading hypothesis after this audit, but
  not fully excluded without experiment 1): a minimal reproduction using `llguidance`'s own public API
  directly (no mistralrs), filed upstream, since a fix would then belong in the vendored crate version
  bump, not in `mistralrs-core`.

## Files added

- `plans/ff-round-two/reports/06-d-llguidance-audit.md` (this report). No other files changed.
- `/tmp/llguidance-src/` (this session's scratch fetch of `llguidance` v1.4.0 source from GitHub at
  tag `v1.4.0` / commit `c1b69da46e2053ce0d40ff4416b6a860f7165143`) is outside both
  `/workspace` and `/tmp/ff-artifacts-wt`, not committed, and not referenced by path from the
  committed report -- all citations above are against the public commit hash, independently
  re-fetchable.

## Git/worktree state

- `/workspace`: branch `grammar-fast-forward`, HEAD `cab8cd3ae`, clean, untouched this session
  (read-only `grep`/`Read` only, no edits).
- This report was written and committed from the separate worktree at `/tmp/ff-artifacts-wt`, branch
  `ff-demo-artifacts`.
- No server requests were made this session; no host server state was touched; network access this
  session was limited to fetching public `llguidance` source files from GitHub (see Sec 0).

Not proceeding to Plan 07.
