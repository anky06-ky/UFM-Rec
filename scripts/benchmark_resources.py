"""Benchmark RAM, VRAM, latency and throughput for the final report.

Usage:
    python scripts/benchmark_resources.py --backend content   # TF-IDF only
    python scripts/benchmark_resources.py --backend ufm       # Full UFM
    python scripts/benchmark_resources.py --backend ufm --device cuda  # GPU inference

Outputs a JSON report to reports/benchmark_resources_<backend>.json.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def get_process_memory_mb() -> float:
    """Resident Set Size of the current process in MiB (cross-platform)."""
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / 2**20
    except ImportError:
        pass
    # Fallback for Linux /proc
    try:
        with open(f'/proc/{os.getpid()}/status') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    return int(line.split()[1]) / 1024  # kB → MiB
    except FileNotFoundError:
        pass
    return -1


def measure_vram() -> dict:
    """Snapshot of CUDA memory usage; returns empty dict on CPU-only systems."""
    try:
        import torch
        if not torch.cuda.is_available():
            return {}
        return {
            'allocated_MiB': round(torch.cuda.memory_allocated() / 2**20, 2),
            'reserved_MiB': round(torch.cuda.memory_reserved() / 2**20, 2),
            'peak_allocated_MiB': round(torch.cuda.max_memory_allocated() / 2**20, 2),
            'peak_reserved_MiB': round(torch.cuda.max_memory_reserved() / 2**20, 2),
            'device_name': torch.cuda.get_device_name(0),
            'device_total_MiB': round(torch.cuda.get_device_properties(0).total_mem / 2**20, 2),
        }
    except Exception:
        return {}


def benchmark_latency(backend, catalog, queries, k=10, warmup=3, repeats=30):
    """Measure per-query latency for various history lengths."""
    results = []
    for label, history in queries:
        # Warmup: ignore first N calls (JIT, caches, etc.)
        for _ in range(warmup):
            backend.recommend(history, k)
        times = []
        for _ in range(repeats):
            gc.collect()
            start = time.perf_counter()
            recs = backend.recommend(history, k)
            elapsed_ms = (time.perf_counter() - start) * 1000
            times.append(elapsed_ms)
        times.sort()
        results.append({
            'query': label,
            'history_length': len(history),
            'top_k': k,
            'repeats': repeats,
            'warmup': warmup,
            'result_count': len(recs),
            'latency_ms': {
                'min': round(min(times), 2),
                'p50': round(times[len(times) // 2], 2),
                'p95': round(times[int(len(times) * 0.95)], 2),
                'p99': round(times[int(len(times) * 0.99)], 2),
                'max': round(max(times), 2),
                'mean': round(statistics.mean(times), 2),
                'std': round(statistics.stdev(times), 2) if len(times) > 1 else 0,
            },
        })
    return results


def benchmark_throughput(backend, catalog, history, k=10, duration_seconds=10):
    """How many queries/second the backend can sustain."""
    # Warmup
    for _ in range(3):
        backend.recommend(history, k)
    count = 0
    start = time.perf_counter()
    deadline = start + duration_seconds
    while time.perf_counter() < deadline:
        backend.recommend(history, k)
        count += 1
    elapsed = time.perf_counter() - start
    return {
        'queries': count,
        'elapsed_seconds': round(elapsed, 2),
        'qps': round(count / elapsed, 3),
        'history_length': len(history),
        'top_k': k,
    }


def estimate_feature_table_size_mb(data_path: Path) -> dict:
    """Estimate memory footprint of the CLIP feature tables."""
    sizes = {}
    for name in ['text.npy', 'image.npy', 'modalities.npy']:
        path = data_path / f'foundation_clip_b32_v1/{name}'
        if path.exists():
            arr = np.load(path, mmap_mode='r')
            sizes[name] = {
                'shape': list(arr.shape),
                'dtype': str(arr.dtype),
                'size_MiB': round(arr.nbytes / 2**20, 2),
            }
    if sizes:
        sizes['total_MiB'] = round(sum(v['size_MiB'] for v in sizes.values()), 2)
    return sizes


def estimate_model_size(model) -> dict:
    """Count parameters and estimate model memory footprint."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # Estimate bytes: float32 by default
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    return {
        'total_parameters': total,
        'trainable_parameters': trainable,
        'param_MiB': round(param_bytes / 2**20, 2),
        'buffer_MiB': round(buffer_bytes / 2**20, 2),
        'total_model_MiB': round((param_bytes + buffer_bytes) / 2**20, 2),
    }


def estimate_training_memory(model_size_mib: float, batch_size: int = 256,
                             feature_table_mib: float = 0) -> dict:
    """Rough estimate of peak training memory."""
    # Model weights + optimizer states (AdamW: 2 states per param) + gradients
    optimizer_mib = model_size_mib * 2  # momentum + variance
    gradient_mib = model_size_mib
    # Activations: rough heuristic — proportional to batch size
    activation_mib = model_size_mib * (batch_size / 32)  # rough scaling
    total = model_size_mib + optimizer_mib + gradient_mib + activation_mib + feature_table_mib
    return {
        'model_weights_MiB': round(model_size_mib, 2),
        'optimizer_states_MiB': round(optimizer_mib, 2),
        'gradients_MiB': round(gradient_mib, 2),
        'activations_estimate_MiB': round(activation_mib, 2),
        'feature_tables_MiB': round(feature_table_mib, 2),
        'estimated_peak_MiB': round(total, 2),
        'estimated_peak_GiB': round(total / 1024, 2),
        'within_16GiB_budget': total < 16 * 1024,
        'batch_size': batch_size,
        'note': 'Rough estimate; actual peak depends on AMP, gradient checkpointing, CUDA fragmentation.',
    }


def build_queries(catalog) -> list:
    """Build a set of representative benchmark queries."""
    queries = []
    # Empty history (new user)
    queries.append(('new_user_no_history', []))
    # Build queries from popular items
    popular = np.argsort(-catalog.counts, kind='stable')[:20]
    asins = [catalog.asins[int(i)] for i in popular]
    for length in [1, 5, 10, 20]:
        queries.append((f'warm_user_{length}_items', asins[:length]))
    return queries


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--backend', choices=['content', 'ufm'], default='content')
    p.add_argument('--data', type=Path,
                   default=ROOT / 'data/processed/toys_games_full_temporal')
    p.add_argument('--run', type=Path, default=ROOT / 'runs/ufm_full_v1')
    p.add_argument('--features', type=Path, default=None,
                   help='Relocated CLIP cache (optional).')
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu',
                   help='Device for UFM inference.')
    p.add_argument('--top-k', type=int, default=10)
    p.add_argument('--latency-repeats', type=int, default=30)
    p.add_argument('--throughput-seconds', type=int, default=10)
    p.add_argument('--output', type=Path, default=None)
    args = p.parse_args()

    if args.output is None:
        args.output = ROOT / f'reports/benchmark_resources_{args.backend}.json'

    print(f'=== Resource Benchmark: {args.backend} backend ===', flush=True)

    # ── Memory baseline ─────────────────────────────────────────────────
    gc.collect()
    ram_before = get_process_memory_mb()
    print(f'RAM before loading: {ram_before:.1f} MiB', flush=True)

    # ── Load backend ─────────────────────────────────────────────────────
    from recommend_catalog import Catalog, ContentBackend, UFMBackend

    print('Loading catalog...', flush=True)
    catalog = Catalog(args.data)
    ram_after_catalog = get_process_memory_mb()

    print(f'Loading {args.backend} backend...', flush=True)
    load_start = time.perf_counter()
    if args.backend == 'content':
        backend = ContentBackend(catalog)
    else:
        backend = UFMBackend(catalog, args.run, features=args.features)
    load_elapsed = time.perf_counter() - load_start
    ram_after_backend = get_process_memory_mb()

    print(f'Backend loaded in {load_elapsed:.2f}s', flush=True)
    print(f'RAM after loading: {ram_after_backend:.1f} MiB', flush=True)

    # ── VRAM snapshot ────────────────────────────────────────────────────
    vram = measure_vram()
    if vram:
        print(f'VRAM peak: {vram["peak_allocated_MiB"]:.1f} MiB', flush=True)

    # ── Feature table sizes ──────────────────────────────────────────────
    feature_sizes = estimate_feature_table_size_mb(args.data)

    # ── Model size (UFM only) ────────────────────────────────────────────
    model_info = {}
    training_estimate = {}
    if args.backend == 'ufm':
        model_info = estimate_model_size(backend.model)
        print(f'Model: {model_info["trainable_parameters"]:,} trainable params '
              f'({model_info["total_model_MiB"]:.2f} MiB)', flush=True)
        feature_total = feature_sizes.get('total_MiB', 0)
        training_estimate = estimate_training_memory(
            model_info['total_model_MiB'], batch_size=256,
            feature_table_mib=feature_total)
        budget_status = '✅ WITHIN' if training_estimate['within_16GiB_budget'] else '❌ EXCEEDS'
        print(f'Estimated peak training: {training_estimate["estimated_peak_GiB"]:.2f} GiB '
              f'{budget_status} 16 GiB budget', flush=True)

    # ── Build queries ────────────────────────────────────────────────────
    queries = build_queries(catalog)
    print(f'\nRunning latency benchmark ({args.latency_repeats} repeats per query)...', flush=True)

    # ── Latency benchmark ────────────────────────────────────────────────
    latency_results = benchmark_latency(
        backend, catalog, queries,
        k=args.top_k, repeats=args.latency_repeats)

    for r in latency_results:
        print(f'  {r["query"]}: p50={r["latency_ms"]["p50"]:.1f}ms  '
              f'p95={r["latency_ms"]["p95"]:.1f}ms  '
              f'p99={r["latency_ms"]["p99"]:.1f}ms', flush=True)

    # ── Throughput benchmark ─────────────────────────────────────────────
    # Use a representative query (5 items in history)
    throughput_history = queries[2][1] if len(queries) > 2 else []
    print(f'\nRunning throughput benchmark ({args.throughput_seconds}s)...', flush=True)
    throughput = benchmark_throughput(
        backend, catalog, throughput_history,
        k=args.top_k, duration_seconds=args.throughput_seconds)
    print(f'  Throughput: {throughput["qps"]:.1f} queries/second', flush=True)

    # ── Final RAM/VRAM ───────────────────────────────────────────────────
    gc.collect()
    vram_final = measure_vram()
    ram_final = get_process_memory_mb()

    # ── Report ───────────────────────────────────────────────────────────
    report = {
        'backend': args.backend,
        'device': args.device if args.backend == 'ufm' else 'cpu',
        'catalog_items': len(catalog.asins),
        'load_time_seconds': round(load_elapsed, 2),
        'memory': {
            'ram_baseline_MiB': round(ram_before, 2),
            'ram_after_catalog_MiB': round(ram_after_catalog, 2),
            'ram_after_backend_MiB': round(ram_after_backend, 2),
            'ram_final_MiB': round(ram_final, 2),
            'ram_backend_delta_MiB': round(ram_after_backend - ram_before, 2),
            'vram_at_load': vram,
            'vram_final': vram_final,
        },
        'feature_tables': feature_sizes,
        'model': model_info,
        'training_memory_estimate': training_estimate,
        'latency': latency_results,
        'throughput': throughput,
        'budget_16GiB': {
            'limit_MiB': 16 * 1024,
            'ram_used_MiB': round(ram_after_backend, 2),
            'within_budget': ram_after_backend < 16 * 1024,
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    tmp.replace(args.output)

    print(f'\n=== Report saved to {args.output} ===', flush=True)
    print(json.dumps({
        'ram_MiB': report['memory']['ram_backend_delta_MiB'],
        'p50_ms': latency_results[2]['latency_ms']['p50'] if len(latency_results) > 2 else None,
        'p95_ms': latency_results[2]['latency_ms']['p95'] if len(latency_results) > 2 else None,
        'qps': throughput['qps'],
        'within_16GiB': report['budget_16GiB']['within_budget'],
    }, indent=2), flush=True)


if __name__ == '__main__':
    main()
