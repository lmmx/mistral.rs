import os
import re
import statistics
import sys
import time

from mistralrs import ChatCompletionRequest, Runner, Which

# A fixed passage the regex grammar forces byte-for-byte, so every token after the first is
# fully determined by the grammar alone -- the "fully forceable" case from the sumac journal's
# own ff_bench.py, adapted for this repo's demo.
PASSAGE = (
    "The butter is kept in the refrigerator door, on the second shelf from the top, "
    "right next to the jam and the mustard. It should be soft enough to spread within "
    "about ten minutes if you take it out ahead of time."
)

REPEATS = int(os.environ.get("FF_BENCH_REPEATS", "8"))
WARMUP = int(os.environ.get("FF_BENCH_WARMUP", "2"))


def build_runner() -> Runner:
    return Runner(
        which=Which.GGUF(
            quantized_model_id="unsloth/Qwen3.5-4B-GGUF",
            quantized_filename="Qwen3.5-4B-Q4_K_M.gguf",
        ),
    )


def run_once(runner: Runner) -> tuple[float, str]:
    start = time.perf_counter()
    res = runner.send_chat_completion_request(
        ChatCompletionRequest(
            model="default",
            messages=[{"role": "user", "content": "Where is the butter?"}],
            max_tokens=len(PASSAGE.split()) + 40,
            temperature=0.0,
            enable_thinking=False,
            grammar_type="regex",
            grammar=re.escape(PASSAGE),
        )
    )
    elapsed = time.perf_counter() - start
    return elapsed, res.choices[0].message.content


def main() -> None:
    label = sys.argv[1] if len(sys.argv) > 1 else "run"
    ff_env = os.environ.get("MISTRALRS_GRAMMAR_FAST_FORWARD", "<unset>")
    print(f"=== {label} (MISTRALRS_GRAMMAR_FAST_FORWARD={ff_env}) ===", flush=True)

    runner = build_runner()

    for i in range(WARMUP):
        elapsed, content = run_once(runner)
        print(f"warmup {i}: {elapsed:.2f}s", flush=True)

    times = []
    contents = set()
    for i in range(REPEATS):
        elapsed, content = run_once(runner)
        times.append(elapsed)
        contents.add(content)
        print(f"repeat {i}: {elapsed:.2f}s", flush=True)

    print(f"\n--- {label} summary ---")
    print(f"median: {statistics.median(times):.2f}s")
    print(f"mean:   {statistics.mean(times):.2f}s")
    print(f"range:  {min(times):.2f}s - {max(times):.2f}s")
    print(f"distinct completions across {REPEATS} repeats: {len(contents)}")
    for c in contents:
        matches_passage = c.strip() == PASSAGE.strip()
        print(f"  matches fixed passage: {matches_passage} | len={len(c)}")


if __name__ == "__main__":
    main()
