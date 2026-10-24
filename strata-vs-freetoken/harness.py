#!/usr/bin/env python3
"""One OpenAI-streaming client for the Strata / FreeToken head-to-head.

The head-to-head invocation takes no arguments. It reads prompts from
./prompts, appends every run to ./results.csv, and labels the engine from
the served model id. A second mode, --breakdown, is the FreeToken TTFT
series only.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROMPT_DIR = ROOT / "prompts"
CSV_PATH = ROOT / "results.csv"
BREAK_CSV = ROOT / "ttft-breakdown.csv"
TOKENIZER = Path(
    os.environ.get(
        "TOKENIZER",
        "/opt/freetoken/models/qwen/Qwen3.8-Flash-Next-NVFP4-ftw-b12x/tokenizer.json",
    )
)
BASE = os.environ.get("BENCH_URL", "http://127.0.0.1:12345/v1")
API_KEY = os.environ.get("API_KEY", "local")
SIZES = (2048, 32768, 131072, 187860)
MEASURED = 3
MAX_TOKENS = 512
SEED = 1
TIMEOUT_S = 1800

CSV_FIELDS = [
    "engine",
    "model",
    "prompt_size",
    "run",
    "kind",
    "ttft_s",
    "prefill_tok_s",
    "decode_tok_s",
    "prompt_tokens",
    "cached_tokens",
    "completion_tokens",
    "total_s",
    "power_limit_w",
    "power_max_w",
    "power_mean_w",
    "sm_clock_mhz",
    "mem_clock_mhz",
    "temp_max_c",
    "vram_mib",
    "other_gpu_pids",
    "error",
]


def engine_name(model: str) -> str:
    low = model.lower()
    if "strata" in low or low.endswith("-local") or "fp4-local" in low:
        return "strata"
    return "freetoken"


def gpu_snapshot() -> dict:
    query = (
        "power.limit,power.draw,clocks.sm,clocks.mem,temperature.gpu,"
        "memory.used,utilization.gpu"
    )
    out = subprocess.check_output(
        [
            "nvidia-smi",
            f"--query-gpu={query}",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip().splitlines()[0]
    parts = [p.strip() for p in out.split(",")]

    def num(value: str) -> float:
        try:
            return float(value)
        except ValueError:
            return float("nan")

    apps = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_gpu_memory",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip()
    return {
        "power_limit_w": num(parts[0]),
        "power_draw_w": num(parts[1]),
        "sm_clock_mhz": num(parts[2]),
        "mem_clock_mhz": num(parts[3]),
        "temp_c": num(parts[4]),
        "vram_mib": num(parts[5]),
        "util": num(parts[6]),
        "apps": apps,
    }


class GpuSampler:
    def __init__(self) -> None:
        self.samples: list[dict] = []
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self.stop.is_set():
            try:
                self.samples.append(gpu_snapshot())
            except Exception as exc:  # recorded later if the row has no samples
                self.samples.append({"error": str(exc)})
            self.stop.wait(0.5)

    def __enter__(self) -> "GpuSampler":
        self.thread.start()
        return self

    def __exit__(self, *_) -> None:
        self.stop.set()
        self.thread.join(timeout=2)

    def summary(self) -> dict:
        good = [s for s in self.samples if "power_draw_w" in s]
        if not good:
            return {}
        draws = [s["power_draw_w"] for s in good]
        temps = [s["temp_c"] for s in good]
        last = good[-1]
        return {
            "power_limit_w": f"{last['power_limit_w']:.0f}",
            "power_max_w": f"{max(draws):.0f}",
            "power_mean_w": f"{(sum(draws) / len(draws)):.0f}",
            "sm_clock_mhz": f"{max(s['sm_clock_mhz'] for s in good):.0f}",
            "mem_clock_mhz": f"{max(s['mem_clock_mhz'] for s in good):.0f}",
            "temp_max_c": f"{max(temps):.0f}",
            "vram_mib": f"{max(s['vram_mib'] for s in good):.0f}",
            "other_gpu_pids": last["apps"].replace("\n", " | "),
        }


def other_compute_pids() -> str:
    out = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip()
    return out


def discover_model() -> str:
    req = urllib.request.Request(
        BASE + "/models",
        headers={"Authorization": f"Bearer {API_KEY}"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        payload = json.load(resp)
    return payload["data"][0]["id"]


def ensure_prompts() -> None:
    missing = []
    for size in SIZES:
        for run in range(MEASURED + 1):
            path = PROMPT_DIR / str(size) / f"run{run}.txt"
            if not path.exists():
                missing.append(path)
    if not missing and (PROMPT_DIR / "64" / "run0.txt").exists():
        return
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(TOKENIZER))
    PROMPT_DIR.mkdir(parents=True, exist_ok=True)

    def exact(n: int, salt: str) -> str:
        salt_n = len(tok.encode(salt).ids)
        if salt_n >= n:
            raise SystemExit(f"salt {salt!r} is {salt_n} tokens, need < {n}")
        text = salt + " fox" * (n - salt_n)
        got = len(tok.encode(text).ids)
        if got != n:
            raise SystemExit(f"{salt!r} encoded to {got}, wanted {n}")
        return text

    for size in list(SIZES) + [64]:
        folder = PROMPT_DIR / str(size)
        folder.mkdir(parents=True, exist_ok=True)
        runs = 5 if size == 64 else MEASURED + 1
        # 2k also needs five distinct bodies for the TTFT series.
        if size == 2048:
            runs = max(runs, 5)
        for run in range(runs):
            path = folder / f"run{run}.txt"
            if path.exists():
                continue
            path.write_text(exact(size, f"RUN {size}-{run}\n"), encoding="utf-8")
            print(f"wrote {path} tokens={size}", flush=True)
    # One shared body for the identical warm series. It matches run0's length.
    ident = PROMPT_DIR / "2048" / "identical.txt"
    if not ident.exists():
        ident.write_text(exact(2048, "RUN 2048-identical\n"), encoding="utf-8")


def chat(text: str, max_tokens: int = MAX_TOKENS) -> dict:
    body = json.dumps(
        {
            "model": chat.model,
            "messages": [{"role": "user", "content": text}],
            "temperature": 0,
            "seed": SEED,
            "max_tokens": max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
    ).encode()
    req = urllib.request.Request(
        BASE + "/chat/completions",
        data=body,
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
    )
    started = time.perf_counter()
    first = None
    usage = None
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                event = json.loads(data)
                if event.get("usage"):
                    usage = event["usage"]
                choices = event.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                piece = delta.get("content") or delta.get("reasoning_content") or ""
                if piece and first is None:
                    first = time.perf_counter() - started
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:500]
        return {"error": f"HTTP {exc.code}: {detail}", "total_s": time.perf_counter() - started}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}", "total_s": time.perf_counter() - started}
    total = time.perf_counter() - started
    prompt_tokens = cached = completion = None
    if usage:
        prompt_tokens = usage.get("prompt_tokens")
        completion = usage.get("completion_tokens")
        details = usage.get("prompt_tokens_details") or {}
        cached = details.get("cached_tokens")
    ttft = first
    prefill = decode = None
    if ttft and prompt_tokens:
        prefill = prompt_tokens / ttft
    if ttft and completion and total > ttft:
        decode = completion / (total - ttft)
    return {
        "ttft_s": ttft,
        "total_s": total,
        "prompt_tokens": prompt_tokens,
        "cached_tokens": cached,
        "completion_tokens": completion,
        "prefill_tok_s": prefill,
        "decode_tok_s": decode,
        "error": "",
    }


def append_row(path: Path, row: dict) -> None:
    new = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if new:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


def run_one(engine: str, size: int, run: int, kind: str, text: str, path: Path, max_tokens: int = MAX_TOKENS) -> dict:
    print(f"start {engine} size={size} {kind} run={run}", flush=True)
    with GpuSampler() as sampler:
        result = chat(text, max_tokens=max_tokens)
    row = {
        "engine": engine,
        "model": chat.model,
        "prompt_size": size,
        "run": run,
        "kind": kind,
        **{k: ("" if result.get(k) is None else result.get(k)) for k in (
            "ttft_s", "prefill_tok_s", "decode_tok_s", "prompt_tokens",
            "cached_tokens", "completion_tokens", "total_s",
        )},
        "error": result.get("error") or "",
    }
    row.update(sampler.summary())
    # Format floats to a stable precision when they are numbers.
    for key in ("ttft_s", "prefill_tok_s", "decode_tok_s", "total_s"):
        value = row.get(key)
        if isinstance(value, float):
            row[key] = f"{value:.4f}"
    append_row(path, row)
    print(
        f"done {engine} size={size} {kind} run={run} "
        f"ttft={row['ttft_s']} prefill={row['prefill_tok_s']} "
        f"decode={row['decode_tok_s']} prompt={row['prompt_tokens']} "
        f"cached={row['cached_tokens']} err={row['error']}",
        flush=True,
    )
    Path("/tmp/bench-heartbeat").write_text(str(time.time()), encoding="utf-8")
    return row


def head_to_head() -> None:
    ensure_prompts()
    chat.model = discover_model()
    engine = engine_name(chat.model)
    apps = other_compute_pids()
    print(f"model={chat.model} engine={engine}", flush=True)
    print(f"gpu_apps={apps}", flush=True)
    for size in SIZES:
        for run in range(MEASURED + 1):
            text = (PROMPT_DIR / str(size) / f"run{run}.txt").read_text(encoding="utf-8")
            kind = "warmup" if run == 0 else "measured"
            run_one(engine, size, run, kind, text, CSV_PATH)
    print("HARNESS_DONE", flush=True)


def breakdown() -> None:
    ensure_prompts()
    chat.model = discover_model()
    engine = engine_name(chat.model)
    print(f"breakdown model={chat.model}", flush=True)
    cold = (PROMPT_DIR / "2048" / "run0.txt").read_text(encoding="utf-8")
    run_one(engine, 2048, 0, "cold", cold, BREAK_CSV, max_tokens=32)
    identical = (PROMPT_DIR / "2048" / "identical.txt").read_text(encoding="utf-8")
    for i in range(5):
        run_one(engine, 2048, i + 1, "warm_identical", identical, BREAK_CSV, max_tokens=32)
    for i in range(5):
        text = (PROMPT_DIR / "2048" / f"run{i}.txt").read_text(encoding="utf-8")
        run_one(engine, 2048, i + 1, "warm_unique", text, BREAK_CSV, max_tokens=32)
    for i in range(5):
        text = (PROMPT_DIR / "64" / f"run{i}.txt").read_text(encoding="utf-8")
        run_one(engine, 64, i + 1, "overhead_64", text, BREAK_CSV, max_tokens=32)
    print("BREAKDOWN_DONE", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--breakdown", action="store_true")
    args = parser.parse_args()
    if args.breakdown:
        breakdown()
    else:
        head_to_head()


if __name__ == "__main__":
    main()
