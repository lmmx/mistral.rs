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


if __name__ == "__main__":
    unittest.main()
