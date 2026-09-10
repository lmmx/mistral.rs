"""Pure-Python checks for `ff_harness.py`.

Everything here runs without the compiled `mistralrs` extension, a Rust toolchain, a model or a
server: only the harness's own argument parsing, fixture handling, scheduling and report helpers.
Nothing here verifies live inference, CUDA behaviour, or that any tracing instrumentation exists.

Run with: python3 -m unittest test_ff_harness -v
"""

from __future__ import annotations

import importlib.util
import sys
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


if __name__ == "__main__":
    unittest.main()
