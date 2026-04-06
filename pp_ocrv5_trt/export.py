"""Export PP-OCRv5 HuggingFace models to ONNX and TensorRT."""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# HF Hub model IDs
MODEL_IDS = {
    "server_det": "PaddlePaddle/PP-OCRv5_server_det_safetensors",
    "mobile_det": "PaddlePaddle/PP-OCRv5_mobile_det_safetensors",
    "server_rec": "PaddlePaddle/PP-OCRv5_server_rec_safetensors",
    "mobile_rec": "PaddlePaddle/PP-OCRv5_mobile_rec_safetensors",
}

# Default TRT optimization profile shapes: (min, opt, max)
# Format: (batch, channels, height, width)
DEFAULT_SHAPES = {
    "det": {
        "min": (1, 3, 320, 320),
        "opt": (1, 3, 640, 640),
        "max": (4, 3, 960, 960),
    },
    "rec": {
        "min": (1, 3, 48, 32),
        "opt": (1, 3, 48, 320),
        "max": (32, 3, 48, 3200),
    },
}


def _model_type(model_name: str) -> str:
    """Return 'det' or 'rec' based on model name."""
    name_lower = model_name.lower()
    if "det" in name_lower:
        return "det"
    elif "rec" in name_lower:
        return "rec"
    raise ValueError(f"Cannot determine model type from: {model_name}")


def export_onnx(
    model_name_or_path: str,
    output_path: str | Path,
    opset_version: int = 17,
) -> Path:
    """Export a PP-OCRv5 HuggingFace model to ONNX.

    Args:
        model_name_or_path: HF model ID (e.g. "server_det" shorthand or full
            "PaddlePaddle/PP-OCRv5_server_det_safetensors") or local path.
        output_path: Where to save the .onnx file.
        opset_version: ONNX opset version.

    Returns:
        Path to the saved ONNX model.
    """
    import torch

    model, model_type = _load_hf_model(model_name_or_path)
    model.eval()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if model_type == "det":
        dummy_input = torch.randn(1, 3, 640, 640)
        dynamic_axes = {
            "pixel_values": {0: "batch", 2: "height", 3: "width"},
            "output": {0: "batch", 2: "height", 3: "width"},
        }
    else:
        dummy_input = torch.randn(1, 3, 48, 320)
        dynamic_axes = {
            "pixel_values": {0: "batch", 3: "width"},
            "output": {0: "batch", 1: "seq_len"},
        }

    logger.info("Exporting %s to ONNX: %s", model_name_or_path, output_path)

    torch.onnx.export(
        model,
        (dummy_input,),
        str(output_path),
        input_names=["pixel_values"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
    )

    logger.info("ONNX export complete: %s", output_path)
    return output_path


def verify_onnx(
    onnx_path: str | Path,
    model_name_or_path: str | None = None,
    atol: float = 1e-3,
) -> bool:
    """Verify ONNX model output matches PyTorch.

    Returns True if outputs match within tolerance.
    """
    import numpy as np
    import onnxruntime as ort
    import torch

    onnx_path = Path(onnx_path)
    model_type = _model_type(str(onnx_path))

    if model_name_or_path:
        model, _ = _load_hf_model(model_name_or_path)
    else:
        # Try to infer from filename
        stem = onnx_path.stem
        model, _ = _load_hf_model(stem)
    model.eval()

    if model_type == "det":
        dummy = torch.randn(1, 3, 640, 640)
    else:
        dummy = torch.randn(1, 3, 48, 320)

    with torch.no_grad():
        pt_output = model(dummy)
        if hasattr(pt_output, "last_hidden_state"):
            pt_result = pt_output.last_hidden_state.numpy()
        else:
            pt_result = pt_output[0].numpy()

    sess = ort.InferenceSession(str(onnx_path))
    ort_result = sess.run(None, {"pixel_values": dummy.numpy()})[0]

    max_diff = np.max(np.abs(pt_result - ort_result))
    logger.info("ONNX verification: max diff = %.6f (atol=%.6f)", max_diff, atol)
    return max_diff < atol


def build_trt_engine(
    onnx_path: str | Path,
    engine_path: str | Path,
    precision: str = "fp16",
    workspace_gb: float = 1.0,
    min_shape: tuple | None = None,
    opt_shape: tuple | None = None,
    max_shape: tuple | None = None,
) -> Path:
    """Build a TensorRT engine from an ONNX model.

    Args:
        onnx_path: Path to the ONNX model.
        engine_path: Where to save the .trt engine.
        precision: "fp32" or "fp16".
        workspace_gb: Max workspace in GB.
        min_shape: Min input shape for optimization profile.
        opt_shape: Optimal input shape.
        max_shape: Max input shape.

    Returns:
        Path to the saved TRT engine.
    """
    import tensorrt as trt

    onnx_path = Path(onnx_path)
    engine_path = Path(engine_path)
    engine_path.parent.mkdir(parents=True, exist_ok=True)

    model_type = _model_type(str(onnx_path))
    defaults = DEFAULT_SHAPES[model_type]
    min_shape = min_shape or defaults["min"]
    opt_shape = opt_shape or defaults["opt"]
    max_shape = max_shape or defaults["max"]

    trt_logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(trt_logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, trt_logger)

    logger.info("Parsing ONNX: %s", onnx_path)
    # Use parse_from_file so TRT can find external weight data files
    # (e.g. model.onnx.data) relative to the ONNX file path.
    if not parser.parse_from_file(str(onnx_path.resolve())):
        for i in range(parser.num_errors):
            logger.error("ONNX parse error: %s", parser.get_error(i))
        raise RuntimeError("Failed to parse ONNX model")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, int(workspace_gb * (1 << 30))
    )

    if precision == "fp16":
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            logger.info("FP16 enabled")
        else:
            logger.warning("FP16 not supported on this GPU, falling back to FP32")

    profile = builder.create_optimization_profile()
    profile.set_shape("pixel_values", min_shape, opt_shape, max_shape)
    config.add_optimization_profile(profile)

    logger.info(
        "Building TRT engine (precision=%s, workspace=%.1fGB)...", precision, workspace_gb
    )
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TRT engine build failed")

    with open(engine_path, "wb") as f:
        f.write(serialized)

    size_mb = engine_path.stat().st_size / (1 << 20)
    logger.info("TRT engine saved: %s (%.1f MB)", engine_path, size_mb)
    return engine_path


def export(
    model_name_or_path: str,
    output_dir: str | Path,
    precision: str = "fp16",
    opset_version: int = 17,
    workspace_gb: float = 1.0,
    verify: bool = True,
    keep_onnx: bool = False,
) -> Path:
    """Full export pipeline: HF model → ONNX → TRT engine.

    Args:
        model_name_or_path: HF model ID or local path.
        output_dir: Directory to save outputs.
        precision: TRT precision ("fp32" or "fp16").
        opset_version: ONNX opset version.
        workspace_gb: TRT workspace in GB.
        verify: If True, verify ONNX matches PyTorch before TRT build.
        keep_onnx: If True, keep the intermediate ONNX file.

    Returns:
        Path to the TRT engine file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine a clean name for the output files
    name = _resolve_name(model_name_or_path)
    onnx_path = output_dir / f"{name}.onnx"
    engine_path = output_dir / f"{name}.trt"

    # Step 1: ONNX export
    export_onnx(model_name_or_path, onnx_path, opset_version)

    # Step 2: Verify (optional)
    if verify:
        ok = verify_onnx(onnx_path, model_name_or_path)
        if not ok:
            logger.warning("ONNX verification FAILED — TRT engine may produce wrong results")
        else:
            logger.info("ONNX verification passed")

    # Step 3: Build TRT engine
    engine = build_trt_engine(onnx_path, engine_path, precision, workspace_gb)

    # Cleanup
    if not keep_onnx:
        onnx_path.unlink()
        logger.info("Removed intermediate ONNX file")

    return engine


def _load_hf_model(model_name_or_path: str):
    """Load HF model, return (model, model_type)."""
    import transformers

    # Handle shorthand names
    if model_name_or_path in MODEL_IDS:
        model_name_or_path = MODEL_IDS[model_name_or_path]

    model_type = _model_type(model_name_or_path)

    logger.info("Loading model: %s", model_name_or_path)
    if model_type == "det":
        model = transformers.AutoModelForObjectDetection.from_pretrained(
            model_name_or_path, trust_remote_code=True
        )
    else:
        # Rec models: load the ForTextRecognition variant (includes head + softmax).
        # AutoModel loads the base model without the classification head.
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
        import importlib
        module = importlib.import_module(
            f"transformers.models.{config.model_type}.modeling_{config.model_type}"
        )
        # Find the ForTextRecognition class by scanning module attributes
        rec_cls = None
        for attr_name in dir(module):
            if attr_name.endswith("ForTextRecognition"):
                rec_cls = getattr(module, attr_name)
                break
        if rec_cls is not None:
            model = rec_cls.from_pretrained(model_name_or_path)
        elif config.architectures:
            model_cls = getattr(module, config.architectures[0])
            model = model_cls.from_pretrained(model_name_or_path)
        else:
            model = transformers.AutoModel.from_pretrained(
                model_name_or_path, trust_remote_code=True
            )
    return model, model_type


def _resolve_name(model_name_or_path: str) -> str:
    """Resolve a model identifier to a clean filename stem."""
    if model_name_or_path in MODEL_IDS:
        return model_name_or_path  # e.g. "server_det"

    # Extract last segment of HF ID or path
    name = Path(model_name_or_path).name
    # Strip common suffixes
    for suffix in ("_safetensors", "_pytorch", "_hf"):
        name = name.removesuffix(suffix)
    return name
