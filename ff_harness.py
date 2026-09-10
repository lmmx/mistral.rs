"""Experimental harness for grammar fast-forward round-two plans 03, 04 and 06.

`ff_bench.py` (same directory) is the reproducibility record for the 6.1-6.2x demo measurement and
is not touched by this file -- see plans/ff-round-two/05-harness.md for why it cannot answer the
open questions those plans ask.

`perf_flags::grammar_fast_forward_enabled` reads MISTRALRS_GRAMMAR_FAST_FORWARD through a
`OnceLock`, so a single process can only ever exercise one flag setting. Every mode below therefore
sets (or unsets) the flag once, before the model loads, and produces one JSON report for that one
setting. `compare` runs two of those as separate OS processes and diffs their reports; nothing in
this file toggles the flag inside a running process.

Modes:
  equality      One fixed request under a partially-forcing JSON schema grammar. Records the full
                token id list, decoded text, finish reason, and per-token timings.
  routing-log   `equality` plus best-effort capture of AnyMoE expert-routing lines from the
                process's own stderr (tracing output), if the running tip logs any. At the tip this
                harness was built against, no such log line exists yet (see
                plans/ff-round-two/reports/03-anymoe-divergence.md); this mode is the capture
                mechanism a future instrumentation change can feed, not a claim that it finds
                anything today.
  arms          `equality` plus best-effort capture of the nine recurrent-site arm messages from
                stderr, same caveat as routing-log (see reports/04-recurrent-site-audit.md).
  concurrency   Launches an HTTP server subprocess, fires N concurrent chat completions (a mix of
                grammar-constrained and unconstrained), and scrapes /metrics before and after.
                Requires a paged-attention-capable (CUDA or Metal) server build.

See plans/ff-round-two/05-harness.md for the full mode contracts.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import random
import re
import shlex
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

FF_ENV_VAR = "MISTRALRS_GRAMMAR_FAST_FORWARD"
REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_REPORT_DIR = REPO_ROOT / "plans" / "ff-round-two" / "reports"
FIXTURE_DIR = REPO_ROOT / "plans" / "ff-round-two" / "fixtures"
DEFAULT_SCHEMA_FIXTURE = FIXTURE_DIR / "partial.schema.json"
# Plan 06 workload B wants differing splice widths, so these differ in how much literal text the
# grammar forces: narrow (~12 structural chars) < partial (~37) < nested (~72) < wide (~112).
DIFFERING_SCHEMA_FIXTURES = (
    FIXTURE_DIR / "narrow.schema.json",
    DEFAULT_SCHEMA_FIXTURE,
    FIXTURE_DIR / "nested.schema.json",
    FIXTURE_DIR / "wide.schema.json",
)
DEFAULT_PROMPT = (
    "Report the status of the last deployment. Respond with only the requested JSON object."
)
EQUALITY_MODES = ("equality", "routing-log", "arms")
ALL_MODES = EQUALITY_MODES + ("concurrency",)

DEFAULT_ROUTING_LOG_PATTERN = r"EXPERT_IDX"
DEFAULT_ARMS_PATTERN = r"\bARM\b"

# Plan 06 reads dimensions 1-2 off `mistralrs_grammar_ff_*` but dimension 5 (tokens/s, forward
# passes) off `mistralrs_decode_tokens_processed_total` / `mistralrs_prefill_tokens_processed_total`
# and dimension 6 (preemption, KV pressure) off `mistralrs_paged_preemptions_total` /
# `mistralrs_kv_cache_blocks_*`, so the default has to be the whole `mistralrs_` namespace.
DEFAULT_METRIC_PREFIXES = ("mistralrs_",)
FF_METRIC_PREFIX = "mistralrs_grammar_ff_"

# `name{label="v",other="w"} 1.5` -- Prometheus writes no space inside the label set, so the whole
# name-plus-labels is one \S+ run and stays intact as the report key.
METRIC_LINE_RE = re.compile(r"^([A-Za-z_:][^\s{]*(?:\{[^\s]*\})?)\s+(\S+)\s*$")


def parse_server_cmd(raw: str) -> list[str]:
    """Split a `--server-cmd` string into an argv list.

    One shell-quoted string rather than `nargs="+"`: argparse stops consuming a `+` list at the
    first token starting with `-`, so a normal server command line (`mistralrs serve -p 1234 -m X`)
    could not be passed at all.
    """
    argv = shlex.split(raw)
    if not argv:
        raise ValueError("--server-cmd is empty after shell splitting")
    return argv


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def flag_env(value: str) -> dict[str, str] | None:
    """Return an env-var override for `value` in {"on", "off", "unset"}, or None for unset."""
    if value == "on":
        return {FF_ENV_VAR: "1"}
    if value == "off":
        return {FF_ENV_VAR: "0"}
    if value == "unset":
        return {}
    raise ValueError(f"unknown flag value: {value!r}")


def resolve_schema_files(schema_set: str, schema_files: list[str] | None) -> list[str]:
    """Pick the concurrency schema set, refusing a `differing` run that cannot actually differ.

    Silently round-robining one fixture would produce workload A while the report claimed workload
    B, which is the shape plan 06 specifically needs to tell apart.
    """
    if schema_files:
        chosen = list(schema_files)
    elif schema_set == "differing":
        chosen = [str(p) for p in DIFFERING_SCHEMA_FIXTURES]
    else:
        chosen = [str(DEFAULT_SCHEMA_FIXTURE)]

    if schema_set == "differing":
        distinct = {Path(p).resolve() for p in chosen}
        if len(distinct) < 2:
            raise ValueError(
                "--schema-set differing needs at least 2 distinct --schema-files, got "
                f"{len(distinct)}: {chosen}"
            )
    missing = [p for p in chosen if not Path(p).is_file()]
    if missing:
        raise ValueError(f"schema fixture(s) not found: {missing}")
    return chosen


def report_path(out_dir: Path, plan: str, mode: str, flag: str, label: str | None) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    parts = [plan, mode, flag]
    if label:
        parts.append(label)
    parts.append(utc_stamp())
    return out_dir / ("-".join(parts) + ".json")


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {path}", file=sys.stderr)


# --------------------------------------------------------------------------------------
# equality / routing-log / arms modes: in-process Runner, one streamed request
# --------------------------------------------------------------------------------------


@contextlib.contextmanager
def capture_stderr():
    """Redirect OS-level fd 2 to a temp file for the duration of the block and yield its path.

    Rust's tracing subscriber writes to the process's real stderr fd, not Python's `sys.stderr`
    object, so capturing this way (rather than `contextlib.redirect_stderr`) is required to see
    anything the mistralrs-core tracing calls emit.
    """
    tmp = tempfile.NamedTemporaryFile(prefix="ff_harness_stderr_", suffix=".log", delete=False)
    tmp_path = Path(tmp.name)
    tmp.close()
    saved_fd = os.dup(2)
    target_fd = os.open(tmp_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
    sys.stderr.flush()
    os.dup2(target_fd, 2)
    os.close(target_fd)
    try:
        yield tmp_path
    finally:
        sys.stderr.flush()
        os.dup2(saved_fd, 2)
        os.close(saved_fd)


def build_plain_runner(args: argparse.Namespace):
    import mistralrs

    dtype = getattr(mistralrs.ModelDType, args.dtype.upper()) if args.dtype != "auto" else (
        mistralrs.ModelDType.Auto
    )
    arch = getattr(mistralrs.Architecture, args.arch) if args.arch else None

    which_kwargs: dict[str, Any] = dict(
        model_id=args.model_id,
        arch=arch,
        tokenizer_json=args.tokenizer_json,
        dtype=dtype,
    )
    which = mistralrs.Which.Plain(**which_kwargs)

    runner_kwargs: dict[str, Any] = dict(which=which, seed=args.seed)
    if args.anymoe_config_json:
        anymoe_cfg = json.loads(Path(args.anymoe_config_json).read_text())
        runner_kwargs["anymoe_config"] = mistralrs.AnyMoeConfig(**anymoe_cfg)

    return mistralrs.Runner(**runner_kwargs)


def build_chat_request(args: argparse.Namespace):
    import mistralrs

    schema_path = Path(args.schema_file)
    grammar = schema_path.read_text()

    return mistralrs.ChatCompletionRequest(
        model="default",
        messages=[{"role": "user", "content": args.prompt}],
        max_tokens=args.max_tokens,
        temperature=0.0,
        enable_thinking=False,
        grammar_type="json_schema",
        grammar=grammar,
        logprobs=True,
        top_logprobs=1,
        stream=True,
    )


def run_equality_like(args: argparse.Namespace) -> dict[str, Any]:
    env_override = flag_env(args.flag)
    for key, value in env_override.items():
        os.environ[key] = value
    if args.flag == "unset":
        os.environ.pop(FF_ENV_VAR, None)

    runner = build_plain_runner(args)
    request = build_chat_request(args)

    capture_needed = args.mode in ("routing-log", "arms")
    pattern = re.compile(
        args.capture_pattern
        or (DEFAULT_ROUTING_LOG_PATTERN if args.mode == "routing-log" else DEFAULT_ARMS_PATTERN)
    )

    steps: list[dict[str, Any]] = []
    finish_reason = None
    start = time.perf_counter()
    last_t = start

    stderr_ctx = capture_stderr() if capture_needed else contextlib.nullcontext(None)
    with stderr_ctx as stderr_path:
        for chunk in runner.send_chat_completion_request(request):
            now = time.perf_counter()
            choice = chunk.choices[0]
            if choice.finish_reason:
                finish_reason = choice.finish_reason
            delta = choice.delta.content or ""
            token_id = None
            token_str = None
            if choice.logprobs is not None and choice.logprobs.top_logprobs:
                token_id = choice.logprobs.top_logprobs[0].token
                token_str = choice.logprobs.token
            if delta or token_id is not None:
                steps.append(
                    {
                        "token_id": token_id,
                        "token_str": token_str,
                        "delta": delta,
                        "elapsed_since_start_s": now - start,
                        "step_duration_s": now - last_t,
                    }
                )
            last_t = now

    captured_lines: list[str] = []
    if capture_needed and stderr_path is not None:
        raw = stderr_path.read_text(errors="replace")
        captured_lines = [line for line in raw.splitlines() if pattern.search(line)]
        stderr_path.unlink(missing_ok=True)

    content = "".join(s["delta"] for s in steps)
    token_ids = [s["token_id"] for s in steps]
    token_ids_complete = all(t is not None for t in token_ids) and len(token_ids) > 0

    report: dict[str, Any] = {
        "plan": args.plan,
        "mode": args.mode,
        "flag_requested": args.flag,
        "flag_env_value": os.environ.get(FF_ENV_VAR, "<unset>"),
        "timestamp_utc": utc_stamp(),
        "model_id": args.model_id,
        "arch": args.arch,
        "dtype": args.dtype,
        "seed": args.seed,
        "prompt": args.prompt,
        "schema_file": str(args.schema_file),
        "max_tokens": args.max_tokens,
        "response": {
            "content": content,
            "finish_reason": finish_reason,
            "token_ids": token_ids if token_ids_complete else None,
            "token_ids_complete": token_ids_complete,
            "num_steps": len(steps),
        },
        "steps": steps,
        "total_elapsed_s": last_t - start,
    }
    if capture_needed:
        report["capture"] = {
            "pattern": pattern.pattern,
            "num_matched_lines": len(captured_lines),
            "lines": captured_lines,
        }
    return report


# --------------------------------------------------------------------------------------
# compare driver: two OS processes, one flag setting each
# --------------------------------------------------------------------------------------


def run_child(args: argparse.Namespace, flag: str) -> Path:
    cmd = [
        args.python,
        str(Path(__file__).resolve()),
        "run",
        "--mode", args.mode,
        "--flag", flag,
        "--model-id", args.model_id,
        "--dtype", args.dtype,
        "--prompt", args.prompt,
        "--schema-file", str(args.schema_file),
        "--max-tokens", str(args.max_tokens),
        "--seed", str(args.seed),
        "--plan", args.plan,
        "--out-dir", str(args.out_dir),
    ]
    if args.arch:
        cmd += ["--arch", args.arch]
    if args.tokenizer_json:
        cmd += ["--tokenizer-json", args.tokenizer_json]
    if args.anymoe_config_json:
        cmd += ["--anymoe-config-json", args.anymoe_config_json]
    if args.capture_pattern:
        cmd += ["--capture-pattern", args.capture_pattern]
    if args.label:
        cmd += ["--label", args.label]

    env = dict(os.environ)
    env.pop(FF_ENV_VAR, None)
    env.update(flag_env(flag))

    print(f"--- spawning child, flag={flag}: {' '.join(cmd)} ---", file=sys.stderr)
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    sys.stderr.write(proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"child (flag={flag}) exited {proc.returncode}")

    reported_path = None
    for line in proc.stderr.splitlines():
        if line.startswith("wrote "):
            reported_path = Path(line[len("wrote "):].strip())
    if reported_path is None:
        raise RuntimeError(f"child (flag={flag}) did not report a written path on stderr")
    return reported_path


def compare_reports(report_a: dict[str, Any], report_b: dict[str, Any]) -> dict[str, Any]:
    ids_a = report_a["response"]["token_ids"]
    ids_b = report_b["response"]["token_ids"]

    if ids_a is None or ids_b is None:
        return {
            "comparable": False,
            "reason": "one or both runs did not return a complete token id list "
            "(model/backend may not have returned logprobs)",
        }

    first_divergence = None
    for i, (a, b) in enumerate(zip(ids_a, ids_b)):
        if a != b:
            first_divergence = i
            break
    else:
        if len(ids_a) != len(ids_b):
            first_divergence = min(len(ids_a), len(ids_b))

    equal = first_divergence is None
    result: dict[str, Any] = {
        "comparable": True,
        "equal": equal,
        "len_a": len(ids_a),
        "len_b": len(ids_b),
    }
    if not equal:
        result["first_divergent_index"] = first_divergence
        result["tail_a"] = ids_a[first_divergence:]
        result["tail_b"] = ids_b[first_divergence:]
        result["content_a"] = report_a["response"]["content"]
        result["content_b"] = report_b["response"]["content"]
    return result


def write_compare_summary(path: Path, flag_a: str, flag_b: str, path_a: Path, path_b: Path,
                           diff: dict[str, Any]) -> None:
    lines = [
        f"# Compare: {flag_a} vs {flag_b}",
        "",
        f"- report {flag_a}: `{path_a}`",
        f"- report {flag_b}: `{path_b}`",
        "",
    ]
    if not diff.get("comparable", False):
        lines.append(f"**Not comparable:** {diff['reason']}")
    elif diff["equal"]:
        lines.append(f"**Token-id equal** across {diff['len_a']} tokens.")
    else:
        lines += [
            "**Token ids diverge.**",
            "",
            f"- first divergent index: {diff['first_divergent_index']}",
            f"- length a / b: {diff['len_a']} / {diff['len_b']}",
            f"- tail a: `{diff['tail_a']}`",
            f"- tail b: `{diff['tail_b']}`",
            "",
            f"- content a: {diff['content_a']!r}",
            f"- content b: {diff['content_b']!r}",
        ]
    path.write_text("\n".join(lines) + "\n")
    print(f"wrote {path}", file=sys.stderr)


def do_compare(args: argparse.Namespace) -> None:
    if args.mode not in EQUALITY_MODES:
        raise SystemExit(f"compare only supports modes {EQUALITY_MODES}, got {args.mode!r}")

    path_off = run_child(args, "off")
    path_on = run_child(args, "on")

    report_off = json.loads(path_off.read_text())
    report_on = json.loads(path_on.read_text())
    diff = compare_reports(report_off, report_on)

    summary_path = path_on.with_name(
        path_on.name.replace(".json", "") + "-vs-off-compare.md"
    )
    write_compare_summary(summary_path, "off", "on", path_off, path_on, diff)

    print(json.dumps(diff, indent=2))
    if diff.get("comparable") and not diff["equal"]:
        sys.exit(1)


# --------------------------------------------------------------------------------------
# concurrency mode: HTTP server subprocess, N concurrent requests, /metrics scrape
# --------------------------------------------------------------------------------------


def wait_for_health(base_url: str, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    last_err: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{base_url}/health", timeout=5) as resp:
                if resp.status == 200:
                    return
        except (urllib.error.URLError, ConnectionError, TimeoutError) as exc:
            last_err = exc
        time.sleep(1.0)
    raise TimeoutError(f"server did not become healthy within {timeout_s}s: {last_err}")


def parse_metrics_body(body: str, prefixes: Sequence[str]) -> dict[str, float]:
    """Parse a Prometheus text exposition body, keeping samples whose name matches any prefix.

    An empty `prefixes` keeps every sample. Label sets stay in the key, so
    `..._splice_drops_total{reason="batch_shape"}` is distinct from the same counter's other
    reasons, which is what plan 06 dimension 1 asks for.
    """
    out: dict[str, float] = {}
    for line in body.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = METRIC_LINE_RE.match(line)
        if not m:
            continue
        key = m.group(1)
        if prefixes and not key.startswith(tuple(prefixes)):
            continue
        try:
            out[key] = float(m.group(2))
        except ValueError:
            continue
    return out


def scrape_metrics(base_url: str, prefixes: Sequence[str]) -> dict[str, float]:
    with urllib.request.urlopen(f"{base_url}/metrics", timeout=10) as resp:
        body = resp.read().decode("utf-8", errors="replace")
    return parse_metrics_body(body, prefixes)


def metric_deltas(before: dict[str, float], after: dict[str, float]) -> dict[str, float]:
    keys = set(before) | set(after)
    return {k: after.get(k, 0.0) - before.get(k, 0.0) for k in sorted(keys)}


@dataclass
class PlannedRequest:
    index: int
    constrained: bool
    schema_index: int | None
    scheduled_offset_s: float


@dataclass
class ConcurrencyRequestResult:
    index: int
    constrained: bool
    ok: bool
    status: int | None
    latency_s: float
    schema_index: int | None = None
    scheduled_offset_s: float = 0.0
    start_offset_s: float = 0.0
    finish_reason: str | None = None
    error: str | None = None


def plan_concurrency_requests(
    num_requests: int,
    unconstrained_fraction: float,
    schema_set: str,
    num_schemas: int,
    seed: int,
    stagger_seconds: float,
) -> list[PlannedRequest]:
    """Decide, deterministically from `seed`, which slots go unconstrained and which schema each
    constrained slot carries.

    Slot choice is seeded rather than "first k are unconstrained" so the unconstrained requests are
    not always the ones submitted first. Schemas are handed out round-robin over a seeded shuffle of
    the fixture order, so a `differing` run gives every slot a distinct schema while `num_requests`
    fits in the fixture set -- which is what plan 06 workload B needs.
    """
    if num_requests < 1:
        raise ValueError(f"--num-requests must be >= 1, got {num_requests}")
    if not 0.0 <= unconstrained_fraction <= 1.0:
        raise ValueError(
            f"--unconstrained-fraction must be in [0, 1], got {unconstrained_fraction}"
        )
    if stagger_seconds < 0.0:
        raise ValueError(f"--stagger-seconds must be >= 0, got {stagger_seconds}")
    if num_schemas < 1:
        raise ValueError("need at least one schema fixture")

    rng = random.Random(seed)
    n_unconstrained = round(num_requests * unconstrained_fraction)
    unconstrained_slots = set(rng.sample(range(num_requests), n_unconstrained))

    order = list(range(num_schemas))
    if schema_set == "differing":
        rng.shuffle(order)

    plan: list[PlannedRequest] = []
    constrained_seen = 0
    for i in range(num_requests):
        constrained = i not in unconstrained_slots
        schema_index = None
        if constrained:
            schema_index = 0 if schema_set == "identical" else order[constrained_seen % num_schemas]
            constrained_seen += 1
        plan.append(PlannedRequest(i, constrained, schema_index, i * stagger_seconds))
    return plan


def build_concurrency_request_body(
    prompt: str, max_tokens: int, schema: dict[str, Any] | None, seed: int | None = None
) -> dict[str, Any]:
    """Build one OpenAI-route body.

    The HTTP route is not the Python `ChatCompletionRequest`: `openai.rs` has no `grammar_type`
    field and its `grammar` is `#[serde(tag = "type", content = "value")]`, so the schema goes in
    as a nested object under `{"type": "json_schema", "value": ...}`. `seed` is a real field on
    that struct (`openai.rs:1180`, covered by its own test at `:2139`).
    """
    body: dict[str, Any] = {
        "model": "default",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "enable_thinking": False,
    }
    if seed is not None:
        body["seed"] = seed
    if schema is not None:
        body["grammar"] = {"type": "json_schema", "value": schema}
    return body


def fire_one(
    base_url: str, planned: PlannedRequest, body: dict[str, Any], wall_start: float
) -> ConcurrencyRequestResult:
    remaining = planned.scheduled_offset_s - (time.perf_counter() - wall_start)
    if remaining > 0:
        time.sleep(remaining)

    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.perf_counter()
    common = dict(
        index=planned.index,
        constrained=planned.constrained,
        schema_index=planned.schema_index,
        scheduled_offset_s=planned.scheduled_offset_s,
        start_offset_s=start - wall_start,
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        latency = time.perf_counter() - start
        finish_reason = payload.get("choices", [{}])[0].get("finish_reason")
        return ConcurrencyRequestResult(
            ok=True, status=resp.status, latency_s=latency, finish_reason=finish_reason, **common
        )
    except Exception as exc:  # noqa: BLE001 - reported in the JSON report, not raised
        latency = time.perf_counter() - start
        status = getattr(exc, "code", None)
        return ConcurrencyRequestResult(
            ok=False, status=status, latency_s=latency, error=str(exc), **common
        )


def run_concurrency(args: argparse.Namespace) -> dict[str, Any]:
    env = dict(os.environ)
    env.pop(FF_ENV_VAR, None)
    env.update(flag_env(args.flag))

    base_url = f"http://{args.server_host}:{args.server_port}"
    metric_prefixes = list(args.metric_prefix)
    schema_files = resolve_schema_files(args.schema_set, args.schema_files)
    plan = plan_concurrency_requests(
        args.num_requests,
        args.unconstrained_fraction,
        args.schema_set,
        len(schema_files),
        args.seed,
        args.stagger_seconds,
    )

    server_cmd = parse_server_cmd(args.server_cmd)
    print(f"--- launching server: {shlex.join(server_cmd)} ---", file=sys.stderr)
    proc = subprocess.Popen(server_cmd, env=env)
    try:
        wait_for_health(base_url, args.startup_timeout_seconds)

        schemas = [json.loads(Path(p).read_text()) for p in schema_files]

        n = args.num_requests
        bodies = [
            build_concurrency_request_body(
                args.prompt,
                args.max_tokens,
                None if planned.schema_index is None else schemas[planned.schema_index],
                seed=args.seed,
            )
            for planned in plan
        ]
        n_unconstrained = sum(1 for planned in plan if not planned.constrained)

        metrics_before = scrape_metrics(base_url, metric_prefixes)
        wall_start = time.perf_counter()
        results: list[ConcurrencyRequestResult] = []
        with ThreadPoolExecutor(max_workers=n) as pool:
            futures = [
                pool.submit(fire_one, base_url, planned, body, wall_start)
                for planned, body in zip(plan, bodies)
            ]
            for fut in as_completed(futures):
                results.append(fut.result())
        wall_elapsed = time.perf_counter() - wall_start
        metrics_after = scrape_metrics(base_url, metric_prefixes)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=30)

    results.sort(key=lambda r: r.index)
    ok_results = [r for r in results if r.ok]
    latencies = [r.latency_s for r in ok_results]

    report: dict[str, Any] = {
        "plan": args.plan,
        "mode": "concurrency",
        "flag_requested": args.flag,
        "flag_env_value": env.get(FF_ENV_VAR, "<unset>"),
        "timestamp_utc": utc_stamp(),
        "server_cmd": server_cmd,
        "server_cmd_raw": args.server_cmd,
        "base_url": base_url,
        "num_requests": n,
        "num_unconstrained": n_unconstrained,
        "unconstrained_fraction": args.unconstrained_fraction,
        "schema_set": args.schema_set,
        "schema_files": schema_files,
        "stagger_seconds": args.stagger_seconds,
        "seed": args.seed,
        "request_seed_sent": args.seed,
        "prompt": args.prompt,
        "max_tokens": args.max_tokens,
        "startup_timeout_seconds": args.startup_timeout_seconds,
        "wall_elapsed_s": wall_elapsed,
        "requests": [
            {
                "index": r.index,
                "constrained": r.constrained,
                "schema_index": r.schema_index,
                "schema_file": None if r.schema_index is None else schema_files[r.schema_index],
                "scheduled_offset_s": r.scheduled_offset_s,
                "start_offset_s": r.start_offset_s,
                "ok": r.ok,
                "status": r.status,
                "latency_s": r.latency_s,
                "finish_reason": r.finish_reason,
                "error": r.error,
            }
            for r in results
        ],
        "latency_summary_s": {
            "count_ok": len(ok_results),
            "count_failed": len(results) - len(ok_results),
            "median": statistics.median(latencies) if latencies else None,
            "mean": statistics.mean(latencies) if latencies else None,
            "min": min(latencies) if latencies else None,
            "max": max(latencies) if latencies else None,
        },
        "metric_prefixes": metric_prefixes,
        "metrics_before": metrics_before,
        "metrics_after": metrics_after,
        "metrics_delta": metric_deltas(metrics_before, metrics_after),
    }
    return report


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def add_common_request_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model-id", required=True)
    p.add_argument("--arch", default=None, help="mistralrs.Architecture member name, e.g. Qwen3")
    p.add_argument("--dtype", default="auto")
    p.add_argument("--tokenizer-json", default=None)
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--schema-file", default=str(DEFAULT_SCHEMA_FIXTURE))
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--plan", default="05")
    p.add_argument("--out-dir", default=str(DEFAULT_REPORT_DIR))
    p.add_argument("--label", default=None)
    p.add_argument("--anymoe-config-json", default=None,
                    help="routing-log mode only: JSON file with AnyMoeConfig kwargs")
    p.add_argument("--capture-pattern", default=None,
                    help="routing-log/arms mode only: regex over captured stderr lines "
                         "(default depends on mode)")


def cmd_run(args: argparse.Namespace) -> None:
    if args.mode in EQUALITY_MODES:
        report = run_equality_like(args)
    elif args.mode == "concurrency":
        report = run_concurrency(args)
    else:
        raise SystemExit(f"unknown mode {args.mode!r}")

    path = report_path(Path(args.out_dir), args.plan, args.mode, args.flag, args.label)
    write_report(path, report)


def cmd_compare(args: argparse.Namespace) -> None:
    do_compare(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="run one mode at one flag setting in this process")
    p_run.add_argument("--mode", choices=ALL_MODES, required=True)
    p_run.add_argument("--flag", choices=("on", "off", "unset"), required=True)
    add_common_request_args(p_run)
    p_run.add_argument("--server-cmd", default=None,
                        help="concurrency mode only: the full server command line as ONE "
                             "shell-quoted string, split with shlex, e.g. "
                             "--server-cmd 'mistralrs serve -p 1234 -m <model> --paged-attn'")
    p_run.add_argument("--server-host", default="127.0.0.1")
    p_run.add_argument("--server-port", type=int, default=11434)
    p_run.add_argument("--startup-timeout-seconds", type=float, default=120.0)
    p_run.add_argument("--num-requests", type=int, default=8)
    p_run.add_argument("--schema-set", choices=("identical", "differing"), default="identical")
    p_run.add_argument("--schema-files", nargs="+", default=None,
                        help="concurrency mode only: fixtures to draw grammars from. Defaults to "
                             "the single equality fixture for --schema-set identical and to the "
                             "four differing-forced-span fixtures for --schema-set differing.")
    p_run.add_argument("--unconstrained-fraction", type=float, default=0.25)
    p_run.add_argument("--stagger-seconds", type=float, default=0.0,
                        help="concurrency mode only: delay request i by i * this, so requests sit "
                             "at different grammar positions while still overlapping (plan 06 "
                             "asks for at least one staggered variant). Keep it well under the "
                             "per-request latency or the requests stop overlapping.")
    p_run.add_argument("--metric-prefix", nargs="*", default=list(DEFAULT_METRIC_PREFIXES),
                        metavar="PREFIX",
                        help="concurrency mode only: keep only /metrics samples whose name starts "
                             "with one of these prefixes (default: %(default)s). Pass "
                             f"'{FF_METRIC_PREFIX}' to narrow to the fast-forward counters, or no "
                             "value at all to keep every sample.")
    p_run.set_defaults(func=cmd_run)

    p_cmp = sub.add_parser("compare", help="run flag=off then flag=on as two OS processes and diff")
    p_cmp.add_argument("--mode", choices=EQUALITY_MODES, required=True)
    add_common_request_args(p_cmp)
    p_cmp.add_argument("--python", default=sys.executable)
    p_cmp.set_defaults(func=cmd_compare)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "run" and args.mode == "concurrency":
        if not args.server_cmd:
            parser.error("--server-cmd is required for --mode concurrency")
        try:
            parse_server_cmd(args.server_cmd)
            files = resolve_schema_files(args.schema_set, args.schema_files)
            plan_concurrency_requests(
                args.num_requests, args.unconstrained_fraction, args.schema_set,
                len(files), args.seed, args.stagger_seconds,
            )
        except ValueError as exc:
            parser.error(str(exc))

    args.func(args)


if __name__ == "__main__":
    main()
