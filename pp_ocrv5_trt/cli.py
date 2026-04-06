"""CLI for PP-OCRv5 TensorRT export and inference."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        prog="pp-ocrv5-trt",
        description="PP-OCRv5 TensorRT export and inference",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable debug logging"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- export ---
    p_export = sub.add_parser("export", help="Export HF model to TRT engine")
    p_export.add_argument(
        "--model",
        required=True,
        help="HF model ID, shorthand (server_det/mobile_det/server_rec/mobile_rec), or local path",
    )
    p_export.add_argument("--output", required=True, help="Output directory")
    p_export.add_argument(
        "--precision", default="fp16", choices=["fp32", "fp16"], help="TRT precision"
    )
    p_export.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    p_export.add_argument(
        "--workspace", type=float, default=1.0, help="TRT workspace (GB)"
    )
    p_export.add_argument(
        "--keep-onnx", action="store_true", help="Keep intermediate ONNX file"
    )
    p_export.add_argument(
        "--no-verify", action="store_true", help="Skip ONNX verification"
    )

    # --- infer ---
    p_infer = sub.add_parser("infer", help="Run single-model inference")
    p_infer.add_argument("--engine", required=True, help="TRT engine path")
    p_infer.add_argument("--image", required=True, help="Input image path")
    p_infer.add_argument(
        "--char-list", help="Character list file (required for rec models)"
    )

    # --- e2e ---
    p_e2e = sub.add_parser("e2e", help="End-to-end OCR (det + rec)")
    p_e2e.add_argument("--det-engine", required=True, help="Detection TRT engine")
    p_e2e.add_argument("--rec-engine", required=True, help="Recognition TRT engine")
    p_e2e.add_argument("--char-list", required=True, help="Character list file")
    p_e2e.add_argument("--image", required=True, help="Input image path")
    p_e2e.add_argument(
        "--threshold", type=float, default=0.3, help="Det binarization threshold"
    )
    p_e2e.add_argument(
        "--box-threshold", type=float, default=0.6, help="Det box confidence threshold"
    )

    # --- bench ---
    p_bench = sub.add_parser("bench", help="Benchmark TRT engine latency")
    p_bench.add_argument("--engine", required=True, help="TRT engine path")
    p_bench.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    p_bench.add_argument(
        "--iterations", type=int, default=100, help="Benchmark iterations"
    )
    p_bench.add_argument(
        "--shape",
        help="Input shape, e.g. '1,3,640,640'",
    )

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.command == "export":
        _cmd_export(args)
    elif args.command == "infer":
        _cmd_infer(args)
    elif args.command == "e2e":
        _cmd_e2e(args)
    elif args.command == "bench":
        _cmd_bench(args)


def _cmd_export(args):
    from .export import export

    engine_path = export(
        model_name_or_path=args.model,
        output_dir=args.output,
        precision=args.precision,
        opset_version=args.opset,
        workspace_gb=args.workspace,
        verify=not args.no_verify,
        keep_onnx=args.keep_onnx,
    )
    print(f"Engine saved: {engine_path}")


def _cmd_infer(args):
    import cv2
    import numpy as np

    from .export import _model_type
    from .processing import postprocess_det, postprocess_rec, preprocess_det, preprocess_rec
    from .runtime import TrtModel

    model = TrtModel(args.engine)
    image = cv2.imread(args.image)
    if image is None:
        print(f"Error: cannot read image {args.image}", file=sys.stderr)
        sys.exit(1)

    model_type = _model_type(args.engine)

    if model_type == "det":
        tensor, orig_size, scale_factor = preprocess_det(image)
        output = model(tensor)
        result = postprocess_det(output, orig_size, scale_factor)
        print(f"Detected {len(result['boxes'])} text regions:")
        for i, (box, score) in enumerate(zip(result["boxes"], result["scores"])):
            print(f"  [{i}] score={score:.3f} box={box.tolist()}")
    else:
        if not args.char_list:
            print("Error: --char-list required for rec models", file=sys.stderr)
            sys.exit(1)
        from .pipeline import OCRPipeline

        chars = OCRPipeline._load_character_list(args.char_list)
        tensor = preprocess_rec([image])
        output = model(tensor)
        results = postprocess_rec(output, chars)
        for text, conf in results:
            print(f"  text='{text}' confidence={conf:.3f}")


def _cmd_e2e(args):
    import cv2

    from .pipeline import OCRPipeline

    image = cv2.imread(args.image)
    if image is None:
        print(f"Error: cannot read image {args.image}", file=sys.stderr)
        sys.exit(1)

    pipeline = OCRPipeline(
        det_engine=args.det_engine,
        rec_engine=args.rec_engine,
        character_list_path=args.char_list,
        det_threshold=args.threshold,
        det_box_threshold=args.box_threshold,
    )

    t0 = time.perf_counter()
    results = pipeline(image)
    elapsed = time.perf_counter() - t0

    print(f"OCR completed in {elapsed*1000:.1f}ms, {len(results)} text regions:")
    for r in results:
        print(f"  '{r['text']}' (conf={r['confidence']:.3f})")


def _cmd_bench(args):
    import numpy as np

    from .runtime import TrtModel

    model = TrtModel(args.engine)

    if args.shape:
        shape = tuple(int(x) for x in args.shape.split(","))
    else:
        # Default based on model type
        from .export import _model_type

        mt = _model_type(args.engine)
        shape = (1, 3, 640, 640) if mt == "det" else (1, 3, 48, 320)

    dummy = np.random.randn(*shape).astype(np.float32)

    # Warmup
    print(f"Warming up ({args.warmup} iterations)...")
    for _ in range(args.warmup):
        model(dummy)

    # Benchmark
    print(f"Benchmarking ({args.iterations} iterations, shape={shape})...")
    times = []
    for _ in range(args.iterations):
        t0 = time.perf_counter()
        model(dummy)
        times.append(time.perf_counter() - t0)

    times = np.array(times) * 1000  # ms
    print(f"  Mean:   {times.mean():.2f} ms")
    print(f"  Median: {np.median(times):.2f} ms")
    print(f"  P95:    {np.percentile(times, 95):.2f} ms")
    print(f"  P99:    {np.percentile(times, 99):.2f} ms")
    print(f"  Min:    {times.min():.2f} ms")
    print(f"  Max:    {times.max():.2f} ms")
    print(f"  Throughput: {1000/times.mean():.1f} infer/s")


if __name__ == "__main__":
    main()
