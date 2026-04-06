# pp-ocrv5-trt: TensorRT Inference for PP-OCRv5 (HuggingFace Transformers)

Standalone tool to export and run PP-OCRv5 models from HuggingFace Transformers
with TensorRT acceleration.

## Models

All 4 PP-OCRv5 models from `transformers>=5.0`:

| Model | HF Class | Input | Output |
|-------|----------|-------|--------|
| `pp_ocrv5_server_det` | `PPOCRV5ServerDetForObjectDetection` | `(N,3,H,W)` H/W dynamic, round to 32, max 960 | `(N,1,H,W)` seg map (sigmoid) |
| `pp_ocrv5_mobile_det` | `PPOCRV5MobileDetForObjectDetection` | same | same |
| `pp_ocrv5_server_rec` | `PPOCRV5ServerRecForTextRecognition` | `(N,3,48,W)` W dynamic 32-3200 | `(N,seq_len,18385)` logits (softmax) |
| `pp_ocrv5_mobile_rec` | `PPOCRV5MobileRecForTextRecognition` | same | same |

All use single `pixel_values` input. No custom ops — fully ONNX-exportable.

## Architecture

```
                ┌─────────────────────────────────────┐
                │          pp-ocrv5-trt CLI            │
                │   export / infer / benchmark / e2e  │
                └──────┬──────┬──────┬───────┬────────┘
                       │      │      │       │
               ┌───────▼──┐ ┌▼──────▼─┐ ┌───▼──────┐
               │  Export   │ │ Runtime  │ │ Pipeline │
               │  Module   │ │ Module   │ │ (e2e)    │
               └───────┬──┘ └┬────────┘ └┬─────────┘
                       │     │           │
           ┌───────────▼─────▼───────────▼──────────┐
           │          Pre/Post Processing            │
           │  (image resize, normalize, CTC decode)  │
           └─────────────────────────────────────────┘
```

## Repo Structure

```
pp-ocrv5-trt/
├── pyproject.toml              # uv/pip installable package
├── README.md
├── pp_ocrv5_trt/
│   ├── __init__.py
│   ├── export.py               # HF model → ONNX → TensorRT
│   ├── runtime.py              # Load TRT engine, run inference
│   ├── pipeline.py             # End-to-end: image → text/boxes
│   ├── processing.py           # Pre/post processing (shared)
│   └── benchmark.py            # Latency/throughput measurement
├── cli.py                      # CLI entry point
└── tests/
    ├── test_export.py
    ├── test_runtime.py
    └── test_pipeline.py
```

## Implementation Plan

### Phase 1: Export (ONNX + TensorRT)

**`pp_ocrv5_trt/export.py`**

#### Step 1: ONNX Export

Load HF model, trace with `torch.onnx.export()`:

```python
def export_onnx(model_name: str, output_path: str, opset: int = 17):
    model = AutoModel.from_pretrained(model_name).eval()
    
    # Detection: dynamic H,W
    # Recognition: fixed H=48, dynamic W
    if "det" in model_name:
        dummy = torch.randn(1, 3, 640, 640)
        dynamic_axes = {"pixel_values": {0: "batch", 2: "height", 3: "width"},
                        "output": {0: "batch", 2: "height", 3: "width"}}
    else:
        dummy = torch.randn(1, 3, 48, 320)
        dynamic_axes = {"pixel_values": {0: "batch", 3: "width"},
                        "output": {0: "batch", 1: "seq_len"}}
    
    torch.onnx.export(model, dummy, output_path,
                      input_names=["pixel_values"],
                      output_names=["output"],
                      dynamic_axes=dynamic_axes,
                      opset_version=opset)
```

#### Step 2: ONNX → TensorRT

Convert using `tensorrt` Python API (not trtexec, for programmatic control):

```python
def build_trt_engine(onnx_path: str, engine_path: str,
                     min_shape, opt_shape, max_shape,
                     precision: str = "fp16"):
    builder = trt.Builder(logger)
    network = builder.create_network(EXPLICIT_BATCH)
    parser = trt.OnnxParser(network, logger)
    parser.parse_from_file(onnx_path)
    
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    if precision == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
    
    profile = builder.create_optimization_profile()
    profile.set_shape("pixel_values", min_shape, opt_shape, max_shape)
    config.add_optimization_profile(profile)
    
    engine = builder.build_serialized_network(network, config)
    # save to engine_path
```

**Default shape profiles:**

| Model Type | Min | Opt | Max |
|-----------|-----|-----|-----|
| det | (1,3,320,320) | (1,3,640,640) | (4,3,960,960) |
| rec | (1,3,48,32) | (1,3,48,320) | (8,3,48,3200) |

### Phase 2: Runtime

**`pp_ocrv5_trt/runtime.py`**

```python
class TrtModel:
    def __init__(self, engine_path: str):
        # Load engine, create execution context
        # Allocate input/output GPU buffers
    
    def __call__(self, pixel_values: np.ndarray) -> np.ndarray:
        # Copy input to GPU, execute, copy output back
        # Returns raw model output (seg map or logits)
```

Uses `tensorrt` Python bindings + `cuda-python` for memory management.
No PyTorch dependency at inference time — pure numpy + TRT.

### Phase 3: Pre/Post Processing

**`pp_ocrv5_trt/processing.py`**

#### Detection Pre-processing
```python
def preprocess_det(image: np.ndarray, limit_side_len: int = 960) -> tuple:
    # 1. Resize preserving aspect ratio, round to 32
    # 2. Rescale: / 255.0
    # 3. Normalize: mean=[0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229]
    # 4. BGR→RGB channel swap
    # Returns: (preprocessed_tensor, original_size)
```

#### Detection Post-processing
```python
def postprocess_det(seg_map: np.ndarray, original_size: tuple,
                    threshold: float = 0.3,
                    box_threshold: float = 0.6) -> list[dict]:
    # 1. Binarize seg map at threshold
    # 2. Find contours (cv2.findContours)
    # 3. Fit bounding boxes (minAreaRect or boundingRect)
    # 4. Filter by box_threshold (mean score inside box)
    # 5. Scale boxes back to original image size
    # Returns: [{"boxes": np.array, "scores": np.array}]
```

#### Recognition Pre-processing
```python
def preprocess_rec(image: np.ndarray, target_height: int = 48,
                   max_width: int = 3200) -> np.ndarray:
    # 1. Resize to height=48, preserve aspect ratio
    # 2. Pad width to nearest multiple (or batch max width)
    # 3. Rescale + normalize (same as det)
    # Returns: preprocessed_tensor
```

#### Recognition Post-processing (CTC decode)
```python
def postprocess_rec(logits: np.ndarray,
                    character_list: list[str]) -> list[tuple[str, float]]:
    # 1. argmax along vocab axis → predicted indices
    # 2. Remove consecutive duplicates
    # 3. Remove blank token (index 0)
    # 4. Map indices to characters via character_list
    # 5. Compute confidence (mean of selected probs)
    # Returns: [(text, confidence), ...]
```

### Phase 4: End-to-End Pipeline

**`pp_ocrv5_trt/pipeline.py`**

```python
class OCRPipeline:
    def __init__(self, det_engine: str, rec_engine: str,
                 character_list_path: str,
                 variant: str = "server"):
        self.det = TrtModel(det_engine)
        self.rec = TrtModel(rec_engine)
        self.chars = load_character_list(character_list_path)
    
    def __call__(self, image: np.ndarray) -> list[dict]:
        # 1. Det: preprocess → infer → postprocess → boxes
        # 2. For each box: crop + perspective transform
        # 3. Rec: batch preprocess → infer → CTC decode
        # Returns: [{"box": [...], "text": "...", "confidence": 0.95}, ...]
```

### Phase 5: CLI

**`cli.py`**

```
# Export
pp-ocrv5-trt export --model pp_ocrv5_server_det --output ./engines/ --precision fp16

# Single-model inference
pp-ocrv5-trt infer --engine ./engines/server_det.trt --image test.jpg

# End-to-end OCR
pp-ocrv5-trt e2e --det-engine ./engines/server_det.trt \
                  --rec-engine ./engines/server_rec.trt \
                  --char-list ./chars.txt \
                  --image test.jpg

# Benchmark
pp-ocrv5-trt bench --engine ./engines/server_det.trt --warmup 10 --iterations 100
```

## Dependencies

```toml
[project]
requires-python = ">=3.10"
dependencies = [
    "numpy",
    "opencv-python",
    "tensorrt>=10.0",
    "cuda-python",
]

[project.optional-dependencies]
export = [
    "torch>=2.0",
    "transformers>=5.0",
    "onnx",
    "onnxruntime",
]
```

Core runtime has **no PyTorch/Transformers dependency** — only numpy + TRT + OpenCV.
Export step needs torch + transformers (one-time).

## Milestones

1. **Export works** — ONNX export for all 4 models, TRT engine builds successfully
2. **Runtime works** — Load TRT engine, feed dummy input, get correct output shape
3. **Accuracy match** — TRT output matches PyTorch output within tolerance (cosine sim > 0.999)
4. **Pipeline works** — End-to-end image → text with TRT, results match HF pipeline
5. **Benchmark** — Measure latency vs PyTorch, report speedup

## Character Dictionary

The 18385-char vocabulary is loaded at runtime. Source:
- From HF model's `processor.character_list` (if using HF tokenizer)
- Or from PaddleOCR's `ppocr/utils/dict/` (e.g., `chinese_sim_dict.txt` + special tokens)
- We'll bundle a default dict or download from HF Hub

## Notes

- Detection models output at original spatial resolution (not downsampled)
- Recognition models apply softmax internally — output is probabilities, not raw logits
- All models set `_can_compile_fullgraph = True` — clean for tracing
- No control flow branching — safe for ONNX static graph export
