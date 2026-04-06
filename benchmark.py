"""Benchmark: PyTorch (HF Transformers) vs TensorRT inference for PP-OCRv5."""

from __future__ import annotations

import argparse
import time

import numpy as np


def bench_pytorch(model_name: str, shape: tuple, warmup: int, iterations: int):
    """Benchmark PyTorch HF model."""
    import torch
    from pp_ocrv5_trt.export import _load_hf_model

    model, model_type = _load_hf_model(model_name)
    model.eval()
    model.cuda()

    dummy = torch.randn(*shape, device="cuda")

    # Warmup
    print(f"  PyTorch warmup ({warmup} iters)...")
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy)
    torch.cuda.synchronize()

    # Benchmark
    print(f"  PyTorch benchmark ({iterations} iters)...")
    times = []
    with torch.no_grad():
        for _ in range(iterations):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(dummy)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

    return np.array(times) * 1000


def bench_trt(engine_path: str, shape: tuple, warmup: int, iterations: int):
    """Benchmark TensorRT engine."""
    from pp_ocrv5_trt.runtime import TrtModel

    model = TrtModel(engine_path)
    dummy = np.random.randn(*shape).astype(np.float32)

    # Warmup
    print(f"  TRT warmup ({warmup} iters)...")
    for _ in range(warmup):
        model(dummy)

    # Benchmark
    print(f"  TRT benchmark ({iterations} iters)...")
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        model(dummy)
        times.append(time.perf_counter() - t0)

    return np.array(times) * 1000


def verify_accuracy(model_name: str, engine_path: str, shape: tuple):
    """Compare PyTorch vs TRT outputs for numerical accuracy."""
    import torch
    from pp_ocrv5_trt.export import _load_hf_model
    from pp_ocrv5_trt.runtime import TrtModel

    model, _ = _load_hf_model(model_name)
    model.eval()

    dummy_np = np.random.randn(*shape).astype(np.float32)
    dummy_pt = torch.from_numpy(dummy_np)

    # PyTorch output
    with torch.no_grad():
        pt_out = model(dummy_pt)
        if hasattr(pt_out, "last_hidden_state"):
            pt_result = pt_out.last_hidden_state.numpy()
        else:
            pt_result = pt_out[0].numpy()

    # TRT output
    trt_model = TrtModel(engine_path)
    trt_result = trt_model(dummy_np)

    # Compare
    max_diff = np.max(np.abs(pt_result - trt_result))
    mean_diff = np.mean(np.abs(pt_result - trt_result))

    # Cosine similarity
    pt_flat = pt_result.flatten()
    trt_flat = trt_result.flatten()
    cos_sim = np.dot(pt_flat, trt_flat) / (
        np.linalg.norm(pt_flat) * np.linalg.norm(trt_flat) + 1e-8
    )

    return max_diff, mean_diff, cos_sim


def print_stats(name: str, times: np.ndarray):
    print(f"  {name}:")
    print(f"    Mean:       {times.mean():.2f} ms")
    print(f"    Median:     {np.median(times):.2f} ms")
    print(f"    P95:        {np.percentile(times, 95):.2f} ms")
    print(f"    P99:        {np.percentile(times, 99):.2f} ms")
    print(f"    Min:        {times.min():.2f} ms")
    print(f"    Max:        {times.max():.2f} ms")
    print(f"    Throughput: {1000 / times.mean():.1f} infer/s")


def main():
    parser = argparse.ArgumentParser(description="Benchmark PyTorch vs TRT")
    parser.add_argument(
        "--model",
        default="server_det",
        help="HF model shorthand or ID (default: server_det)",
    )
    parser.add_argument("--engine", required=True, help="TRT engine path")
    parser.add_argument(
        "--shape", default=None, help="Input shape, e.g. '1,3,640,640'"
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument(
        "--no-accuracy", action="store_true", help="Skip accuracy comparison"
    )
    args = parser.parse_args()

    from pp_ocrv5_trt.export import _model_type

    mt = _model_type(args.model)
    if args.shape:
        shape = tuple(int(x) for x in args.shape.split(","))
    else:
        shape = (1, 3, 640, 640) if mt == "det" else (1, 3, 48, 320)

    print(f"Model: {args.model} ({mt})")
    print(f"Input shape: {shape}")
    print(f"Engine: {args.engine}")
    print()

    # --- Accuracy ---
    if not args.no_accuracy:
        print("=== Accuracy Comparison ===")
        max_diff, mean_diff, cos_sim = verify_accuracy(
            args.model, args.engine, shape
        )
        print(f"  Max absolute diff:  {max_diff:.6f}")
        print(f"  Mean absolute diff: {mean_diff:.6f}")
        print(f"  Cosine similarity:  {cos_sim:.6f}")
        print()

    # --- Latency ---
    print("=== Latency Benchmark ===")
    print()

    pt_times = bench_pytorch(args.model, shape, args.warmup, args.iterations)
    print()
    trt_times = bench_trt(args.engine, shape, args.warmup, args.iterations)
    print()

    print("--- Results ---")
    print_stats("PyTorch (GPU)", pt_times)
    print()
    print_stats("TensorRT (FP16)", trt_times)
    print()

    speedup = np.median(pt_times) / np.median(trt_times)
    print(f"  Speedup (median): {speedup:.2f}x")
    print(
        f"  Throughput gain:  {1000 / trt_times.mean():.1f} vs "
        f"{1000 / pt_times.mean():.1f} infer/s"
    )


if __name__ == "__main__":
    main()
