"""Benchmark a running UFM demo over loopback with warm-up and saved provenance."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import platform
from pathlib import Path
import threading
import time
import urllib.error
import urllib.request

import numpy as np
import psutil


def get_json(url: str) -> dict | list:
    with urllib.request.urlopen(url, timeout=30) as response:
        return json.load(response)


def request(url: str, payload: dict, barrier: threading.Barrier | None = None) -> dict:
    if barrier is not None:
        barrier.wait(timeout=10)
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url + "/api/recommend", body,
                                 {"Content-Type": "application/json"})
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=900) as response:
            result = json.load(response)
            return {"status": response.status, "seconds": time.perf_counter() - started,
                    "results": len(result["results"]), "model": result["model"]}
    except urllib.error.HTTPError as error:
        return {"status": error.code, "seconds": time.perf_counter() - started,
                "error": json.loads(error.read())["error"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8771")
    parser.add_argument("--server-pid", type=int, required=True)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.warmup < 1 or args.repeats < 5:
        raise ValueError("Use a fresh output, >=1 warm-up and >=5 measured requests")
    url = args.url.rstrip("/")
    if url not in {"http://127.0.0.1:8771", "http://localhost:8771"}:
        raise ValueError("This acceptance benchmark is loopback port 8771 only")
    status, examples = get_json(url + "/api/status"), get_json(url + "/api/examples")
    if status["backend"] != "UFM Rec" or status["items"] != 767045 or status["data"]["counts_simulated"]:
        raise ValueError("Expected full real-data UFM backend")
    process = psutil.Process(args.server_pid)
    payload = {"history": [examples[0]["asin"]], "regime": "all", "k": 5}
    rss_samples = []
    stop = threading.Event()

    def sample_memory() -> None:
        while not stop.wait(0.1):
            try:
                rss_samples.append(process.memory_info().rss)
            except psutil.NoSuchProcess:
                return

    rss_samples.append(process.memory_info().rss)
    observer = threading.Thread(target=sample_memory, daemon=True)
    observer.start()
    try:
        warmup = [request(url, payload) for _ in range(args.warmup)]
        measured = [request(url, payload) for _ in range(args.repeats)]
        barrier = threading.Barrier(2)
        with ThreadPoolExecutor(max_workers=2) as pool:
            concurrent = list(pool.map(lambda _: request(url, payload, barrier), range(2)))
    finally:
        stop.set()
        observer.join(timeout=2)
    if any(x["status"] != 200 or x["model"] != status["model"] or x["results"] != 5
           for x in warmup + measured):
        raise ValueError("Sequential inference failed or model changed")
    if sorted(x["status"] for x in concurrent) != [200, 429]:
        raise ValueError("Expected one accepted request and one busy response at concurrency 2")
    seconds = np.array([x["seconds"] for x in measured])
    report = {
        "observed_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "hardware": {"platform": platform.platform(), "logical_cpu": psutil.cpu_count(),
                     "physical_ram_mib": round(psutil.virtual_memory().total / 2**20, 1)},
        "request": payload,
        "warmup_requests": args.warmup,
        "measured_requests": args.repeats,
        "latency_seconds": {"p50": float(np.percentile(seconds, 50)),
                            "p95": float(np.percentile(seconds, 95)),
                            "p99": float(np.percentile(seconds, 99)),
                            "min": float(seconds.min()), "max": float(seconds.max())},
        "sequential_qps": float(len(seconds) / seconds.sum()),
        "server_rss_mib": {"at_start": round(rss_samples[0] / 2**20, 1),
                           "observed_max": round(max(rss_samples) / 2**20, 1)},
        "concurrent_two_request_statuses": [x["status"] for x in concurrent],
        "limitations": ["One Windows host and one history/K setting; no inference VRAM on CPU.",
                        "Observed RSS during this run is not training peak RAM.",
                        "Server allows one active inference and returns HTTP 429 to the other request.",
                        "Sequential throughput is not concurrent throughput."],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"p50_s": report["latency_seconds"]["p50"],
                      "p95_s": report["latency_seconds"]["p95"],
                      "rss_peak_mib": report["server_rss_mib"]["observed_max"],
                      "concurrent": report["concurrent_two_request_statuses"]}))


if __name__ == "__main__":
    main()
