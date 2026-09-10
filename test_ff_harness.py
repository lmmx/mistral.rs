"""Pure-Python checks for `ff_harness.py`.

Everything here runs without the compiled `mistralrs` extension, a Rust toolchain, a model or a
server: only the harness's own argument parsing, fixture handling, scheduling and report helpers.
Nothing here verifies live inference, CUDA behaviour, or that any tracing instrumentation exists.

Run with: python3 -m unittest test_ff_harness -v
"""

from __future__ import annotations

import collections
import concurrent.futures
import contextlib
import http.server
import importlib.util
import io
import json
import socket
import sys
import threading
import time
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _load_harness():
    spec = importlib.util.spec_from_file_location("ff_harness", HERE / "ff_harness.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["ff_harness"] = module
    spec.loader.exec_module(module)
    return module


ffh = _load_harness()


class ServerCmdTest(unittest.TestCase):
    """F1: a server command carrying its own flags must survive argument parsing."""

    def test_flagged_command_parses(self):
        argv = ffh.parse_server_cmd("mistralrs serve -p 1234 -m Qwen/Qwen3-0.6B --paged-attn")
        self.assertEqual(
            argv,
            ["mistralrs", "serve", "-p", "1234", "-m", "Qwen/Qwen3-0.6B", "--paged-attn"],
        )

    def test_quoting_is_respected(self):
        argv = ffh.parse_server_cmd("'./my server' --chat-template 'a b.json'")
        self.assertEqual(argv, ["./my server", "--chat-template", "a b.json"])

    def test_empty_command_rejected(self):
        with self.assertRaises(ValueError):
            ffh.parse_server_cmd("   ")

    def test_cli_accepts_flagged_server_cmd(self):
        parser = ffh.build_parser()
        args = parser.parse_args([
            "run", "--mode", "concurrency", "--flag", "on", "--model-id", "x",
            "--server-cmd", "mistralrs serve -p 1234 -m foo",
        ])
        self.assertEqual(
            ffh.parse_server_cmd(args.server_cmd),
            ["mistralrs", "serve", "-p", "1234", "-m", "foo"],
        )


# A trimmed but faithful sample of what mistral.rs's Prometheus exporter renders: the fast-forward
# counters plan 06 dimensions 1-2 read, the token counters dimension 5 reads, and the preemption /
# KV gauges dimension 6 reads, plus one non-mistralrs family that must never be picked up.
SAMPLE_METRICS_BODY = """\
# HELP mistralrs_grammar_ff_splices_staged_total staged splices
# TYPE mistralrs_grammar_ff_splices_staged_total counter
mistralrs_grammar_ff_splices_staged_total 42
mistralrs_grammar_ff_splice_drops_total{reason="batch_shape"} 7
mistralrs_grammar_ff_splice_drops_total{reason="realloc"} 1
mistralrs_grammar_ff_tokens_fed_total 311
mistralrs_decode_tokens_processed_total 9001
mistralrs_prefill_tokens_processed_total 512
mistralrs_paged_preemptions_total 3
mistralrs_kv_cache_blocks_used 128
http_requests_total{route="/v1/chat/completions"} 8
not a metric line at all
"""


class MetricFilterTest(unittest.TestCase):
    """F4: plan 06 needs more than the fast-forward counters, and needs them narrowable."""

    def test_default_prefix_keeps_every_plan_06_metric(self):
        got = ffh.parse_metrics_body(SAMPLE_METRICS_BODY, ffh.DEFAULT_METRIC_PREFIXES)
        for needed in (
            "mistralrs_grammar_ff_splices_staged_total",
            "mistralrs_grammar_ff_tokens_fed_total",
            "mistralrs_decode_tokens_processed_total",
            "mistralrs_prefill_tokens_processed_total",
            "mistralrs_paged_preemptions_total",
            "mistralrs_kv_cache_blocks_used",
        ):
            self.assertIn(needed, got, needed)
        self.assertEqual(got["mistralrs_decode_tokens_processed_total"], 9001.0)

    def test_labels_are_kept_distinct_in_the_key(self):
        got = ffh.parse_metrics_body(SAMPLE_METRICS_BODY, ffh.DEFAULT_METRIC_PREFIXES)
        self.assertEqual(got['mistralrs_grammar_ff_splice_drops_total{reason="batch_shape"}'], 7.0)
        self.assertEqual(got['mistralrs_grammar_ff_splice_drops_total{reason="realloc"}'], 1.0)

    def test_comments_and_junk_are_skipped(self):
        got = ffh.parse_metrics_body(SAMPLE_METRICS_BODY, ffh.DEFAULT_METRIC_PREFIXES)
        self.assertFalse([k for k in got if k.startswith("#")])
        self.assertNotIn("not", got)

    def test_non_mistralrs_families_are_dropped_by_default(self):
        got = ffh.parse_metrics_body(SAMPLE_METRICS_BODY, ffh.DEFAULT_METRIC_PREFIXES)
        self.assertFalse([k for k in got if k.startswith("http_requests_total")])

    def test_narrowing_to_the_fast_forward_prefix_still_works(self):
        got = ffh.parse_metrics_body(SAMPLE_METRICS_BODY, [ffh.FF_METRIC_PREFIX])
        self.assertEqual(len(got), 4)
        self.assertTrue(all(k.startswith(ffh.FF_METRIC_PREFIX) for k in got))

    def test_empty_prefix_list_keeps_everything(self):
        got = ffh.parse_metrics_body(SAMPLE_METRICS_BODY, [])
        self.assertIn('http_requests_total{route="/v1/chat/completions"}', got)

    def test_deltas_handle_a_key_present_on_one_side_only(self):
        before = {"a": 1.0, "b": 5.0}
        after = {"a": 4.0, "c": 2.0}
        self.assertEqual(ffh.metric_deltas(before, after), {"a": 3.0, "b": -5.0, "c": 2.0})

    def test_cli_exposes_the_prefix_knob(self):
        parser = ffh.build_parser()
        base = ["run", "--mode", "concurrency", "--flag", "on", "--model-id", "x",
                "--server-cmd", "srv"]
        self.assertEqual(parser.parse_args(base).metric_prefix, list(ffh.DEFAULT_METRIC_PREFIXES))
        narrowed = parser.parse_args(base + ["--metric-prefix", ffh.FF_METRIC_PREFIX])
        self.assertEqual(narrowed.metric_prefix, [ffh.FF_METRIC_PREFIX])
        self.assertEqual(parser.parse_args(base + ["--metric-prefix"]).metric_prefix, [])


def _structural_chars(schema: dict) -> int:
    """Characters a JSON-schema grammar forces outright: braces, quoted keys, colons, commas."""
    props = schema.get("properties", {})
    total = 2 + max(0, len(props) - 1)
    for name, sub in props.items():
        total += len(name) + 3
        if sub.get("type") == "object":
            total += _structural_chars(sub)
    return total


class FixtureTest(unittest.TestCase):
    """F6: workload B is only workload B if the fixtures really force different amounts of text."""

    def test_every_differing_fixture_exists_and_is_valid_json(self):
        for path in ffh.DIFFERING_SCHEMA_FIXTURES:
            self.assertTrue(path.is_file(), path)
            json.loads(path.read_text())

    def test_fixtures_are_partially_forcing(self):
        for path in ffh.DIFFERING_SCHEMA_FIXTURES:
            schema = json.loads(path.read_text())
            self.assertEqual(schema["type"], "object", path)
            # forced spans: closed object, every property required
            self.assertFalse(schema["additionalProperties"], path)
            self.assertEqual(sorted(schema["required"]), sorted(schema["properties"]), path)
            # free spans: the model still chooses somewhere
            leaves = _leaf_types(schema)
            self.assertTrue(leaves & {"string", "integer"}, path)

    def test_forced_span_budgets_are_all_distinct(self):
        budgets = {p.name: _structural_chars(json.loads(p.read_text()))
                   for p in ffh.DIFFERING_SCHEMA_FIXTURES}
        self.assertEqual(len(set(budgets.values())), len(budgets), budgets)
        spread = max(budgets.values()) / min(budgets.values())
        self.assertGreater(spread, 4.0, budgets)


def _leaf_types(schema: dict) -> set[str]:
    out = set()
    for sub in schema.get("properties", {}).values():
        if sub.get("type") == "object":
            out |= _leaf_types(sub)
        else:
            out.add(sub.get("type"))
    return out


class SchemaSetTest(unittest.TestCase):
    """F6: `differing` must fail loudly rather than silently becoming `identical`."""

    def test_differing_defaults_to_the_full_fixture_set(self):
        chosen = ffh.resolve_schema_files("differing", None)
        self.assertEqual(len(chosen), len(ffh.DIFFERING_SCHEMA_FIXTURES))

    def test_identical_defaults_to_the_single_equality_fixture(self):
        self.assertEqual(ffh.resolve_schema_files("identical", None),
                         [str(ffh.DEFAULT_SCHEMA_FIXTURE)])

    def test_differing_with_one_fixture_is_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            ffh.resolve_schema_files("differing", [str(ffh.DEFAULT_SCHEMA_FIXTURE)])
        self.assertIn("at least 2", str(ctx.exception))

    def test_differing_with_the_same_fixture_twice_is_rejected(self):
        dup = str(ffh.DEFAULT_SCHEMA_FIXTURE)
        with self.assertRaises(ValueError):
            ffh.resolve_schema_files("differing", [dup, dup])

    def test_identical_with_one_fixture_is_fine(self):
        ffh.resolve_schema_files("identical", [str(ffh.DEFAULT_SCHEMA_FIXTURE)])

    def test_missing_fixture_is_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            ffh.resolve_schema_files("identical", ["/nonexistent/nope.schema.json"])
        self.assertIn("not found", str(ctx.exception))

    def test_cli_rejects_a_differing_run_that_cannot_differ(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            ffh.main([
                "run", "--mode", "concurrency", "--flag", "on", "--model-id", "x",
                "--server-cmd", "srv --port 1", "--schema-set", "differing",
                "--schema-files", str(ffh.DEFAULT_SCHEMA_FIXTURE),
            ])


class RequestBodyTest(unittest.TestCase):
    """The HTTP route's grammar field is a tagged object, not the Python API's flat string pair."""

    def test_constrained_body_uses_the_tagged_grammar_object(self):
        schema = json.loads(ffh.DEFAULT_SCHEMA_FIXTURE.read_text())
        body = ffh.build_concurrency_request_body("hi", 16, schema)
        self.assertEqual(body["grammar"], {"type": "json_schema", "value": schema})
        self.assertNotIn("grammar_type", body)

    def test_unconstrained_body_carries_no_grammar(self):
        body = ffh.build_concurrency_request_body("hi", 16, None)
        self.assertNotIn("grammar", body)

    def test_body_is_json_serialisable(self):
        schema = json.loads(ffh.DEFAULT_SCHEMA_FIXTURE.read_text())
        json.dumps(ffh.build_concurrency_request_body("hi", 16, schema, seed=7))

    def test_seed_is_only_sent_when_given(self):
        self.assertEqual(ffh.build_concurrency_request_body("hi", 16, None, seed=7)["seed"], 7)
        self.assertNotIn("seed", ffh.build_concurrency_request_body("hi", 16, None))


class ConcurrencyPlanTest(unittest.TestCase):
    """F7: the schedule must be reproducible from --seed and must actually stagger."""

    def plan(self, **kw):
        params = dict(num_requests=8, unconstrained_fraction=0.25, schema_set="differing",
                      num_schemas=4, seed=42, stagger_seconds=0.0)
        params.update(kw)
        return ffh.plan_concurrency_requests(**params)

    def test_same_seed_gives_the_same_plan(self):
        self.assertEqual(self.plan(), self.plan())

    def test_a_different_seed_gives_a_different_plan(self):
        seeds = {tuple((p.constrained, p.schema_index) for p in self.plan(seed=s))
                 for s in range(12)}
        self.assertGreater(len(seeds), 1)

    def test_unconstrained_count_follows_the_fraction(self):
        for frac, expected in ((0.0, 0), (0.25, 2), (0.5, 4), (1.0, 8)):
            plan = self.plan(unconstrained_fraction=frac)
            self.assertEqual(sum(1 for p in plan if not p.constrained), expected, frac)

    def test_unconstrained_slots_are_not_always_the_first_ones(self):
        # the pre-fix behaviour put every unconstrained request at the head of the batch
        head_only = 0
        for seed in range(20):
            plan = self.plan(seed=seed, unconstrained_fraction=0.5)
            idx = [p.index for p in plan if not p.constrained]
            if idx == list(range(len(idx))):
                head_only += 1
        self.assertLess(head_only, 20)

    def test_differing_gives_every_slot_a_distinct_schema_when_it_fits(self):
        plan = self.plan(num_requests=4, unconstrained_fraction=0.0, num_schemas=4)
        used = [p.schema_index for p in plan]
        self.assertEqual(sorted(used), [0, 1, 2, 3])

    def test_differing_round_robins_when_it_does_not_fit(self):
        plan = self.plan(num_requests=8, unconstrained_fraction=0.0, num_schemas=4)
        counts = collections.Counter(p.schema_index for p in plan)
        self.assertEqual(sorted(counts.values()), [2, 2, 2, 2])

    def test_identical_always_uses_the_first_schema(self):
        plan = self.plan(schema_set="identical", num_schemas=1, unconstrained_fraction=0.0)
        self.assertEqual({p.schema_index for p in plan}, {0})

    def test_unconstrained_slots_carry_no_schema(self):
        for p in self.plan(unconstrained_fraction=0.5):
            self.assertEqual(p.schema_index is None, not p.constrained)

    def test_stagger_offsets_are_monotonic_and_scaled(self):
        plan = self.plan(num_requests=4, stagger_seconds=0.25)
        self.assertEqual([p.scheduled_offset_s for p in plan], [0.0, 0.25, 0.5, 0.75])

    def test_zero_stagger_means_a_synchronised_start(self):
        self.assertEqual({p.scheduled_offset_s for p in self.plan(stagger_seconds=0.0)}, {0.0})

    def test_invalid_parameters_are_rejected(self):
        for kw in ({"num_requests": 0}, {"unconstrained_fraction": 1.5},
                   {"unconstrained_fraction": -0.1}, {"stagger_seconds": -1.0},
                   {"num_schemas": 0}):
            with self.assertRaises(ValueError, msg=kw):
                self.plan(**kw)

    def test_cli_exposes_seed_and_stagger(self):
        parser = ffh.build_parser()
        args = parser.parse_args([
            "run", "--mode", "concurrency", "--flag", "on", "--model-id", "x",
            "--server-cmd", "srv", "--seed", "7", "--stagger-seconds", "0.5",
        ])
        self.assertEqual(args.seed, 7)
        self.assertEqual(args.stagger_seconds, 0.5)


class _StubHandler(http.server.BaseHTTPRequestHandler):
    """Minimal stand-in for the mistral.rs HTTP surface. No model, no inference."""

    bodies: list = []

    def log_message(self, *_args):
        pass

    def _send(self, code, payload, content_type="application/json"):
        raw = payload.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        if self.path == "/health":
            self._send(200, "{}")
        elif self.path == "/metrics":
            self._send(200, SAMPLE_METRICS_BODY, "text/plain")
        else:
            self._send(404, "{}")

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        type(self).bodies.append(json.loads(self.rfile.read(length)))
        time.sleep(0.25)  # long enough that staggered requests still overlap
        self._send(200, json.dumps({"choices": [{"finish_reason": "stop"}]}))


class StubServerTest(unittest.TestCase):
    """Exercises fire_one / scrape_metrics against a real socket. Still no model, no mistralrs."""

    def setUp(self):
        _StubHandler.bodies = []
        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _StubHandler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.base_url = "http://127.0.0.1:%d" % self.server.server_address[1]

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()

    def test_health_wait_and_metric_scrape(self):
        ffh.wait_for_health(self.base_url, 5.0)
        metrics = ffh.scrape_metrics(self.base_url, ffh.DEFAULT_METRIC_PREFIXES)
        self.assertEqual(metrics["mistralrs_decode_tokens_processed_total"], 9001.0)

    def test_stagger_delays_starts_but_requests_still_overlap(self):
        plan = ffh.plan_concurrency_requests(4, 0.0, "identical", 1, 42, stagger_seconds=0.05)
        body = ffh.build_concurrency_request_body("hi", 4, None, seed=42)
        wall_start = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(
                lambda pr: ffh.fire_one(self.base_url, pr, body, wall_start), plan))
        wall_elapsed = time.perf_counter() - wall_start

        self.assertTrue(all(r.ok for r in results), [r.error for r in results])
        starts = [r.start_offset_s for r in results]
        self.assertEqual(starts, sorted(starts))
        self.assertGreater(starts[-1] - starts[0], 0.1)   # the stagger really happened
        # overlapping: serialising 4 x 0.25s requests would take >= 1.0s
        self.assertLess(wall_elapsed, 0.9)

    def test_the_server_receives_the_tagged_grammar_and_the_seed(self):
        schema = json.loads(ffh.DEFAULT_SCHEMA_FIXTURE.read_text())
        plan = ffh.plan_concurrency_requests(1, 0.0, "identical", 1, 42, 0.0)
        body = ffh.build_concurrency_request_body("hi", 4, schema, seed=99)
        result = ffh.fire_one(self.base_url, plan[0], body, time.perf_counter())
        self.assertTrue(result.ok)
        received = _StubHandler.bodies[0]
        self.assertEqual(received["grammar"]["type"], "json_schema")
        self.assertEqual(received["grammar"]["value"], schema)
        self.assertEqual(received["seed"], 99)

    def test_a_failed_request_is_recorded_not_raised(self):
        plan = ffh.plan_concurrency_requests(1, 0.0, "identical", 1, 42, 0.0)
        dead = "http://127.0.0.1:%d" % _free_port()
        result = ffh.fire_one(dead, plan[0], {"model": "default"}, time.perf_counter())
        self.assertFalse(result.ok)
        self.assertIsNotNone(result.error)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


if __name__ == "__main__":
    unittest.main()
