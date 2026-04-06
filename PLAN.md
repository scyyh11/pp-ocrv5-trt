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

Must match PaddleOCR's exact pipeline (PaddleX DBPostProcess / CTCLabelDecode)
to produce identical results.

#### Detection Pre-processing
```python
def preprocess_det(image: np.ndarray, limit_side_len: int = 960) -> tuple:
    # 1. Resize preserving aspect ratio, round dims to nearest 32
    #    - limit_type="max": longest side ≤ limit_side_len
    #    - max_side_limit=4000
    # 2. BGR→RGB channel swap
    # 3. Rescale: / 255.0
    # 4. Normalize: mean=[0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229]
    #    NOTE: HF Transformers models use ImageNet stats.
    #    Must verify which normalization the exported weights expect.
    # 5. HWC→CHW transpose
    # Returns: (preprocessed_tensor, original_size, scale_factor)
```

#### Detection Post-processing (DBNet — must match PaddleOCR exactly)
```python
def postprocess_det(seg_map: np.ndarray, original_size: tuple,
                    scale_factor: tuple,
                    threshold: float = 0.3,
                    box_threshold: float = 0.6,
                    unclip_ratio: float = 2.0,
                    max_candidates: int = 1000) -> list[dict]:
    # 1. Binarize: binary_mask = seg_map[0] > threshold
    # 2. Find contours: cv2.findContours(RETR_LIST, CHAIN_APPROX_SIMPLE)
    #    - Limit to max_candidates (1000)
    # 3. For each contour:
    #    a. Approximate polygon: cv2.approxPolyDP(eps=0.002*arclen)
    #    b. Need ≥4 points
    #    c. Score: box_score_fast() — mean of seg_map values inside oriented rect
    #    d. Filter: score < box_threshold → discard
    # 4. Unclip: expand polygon using pyclipper (Clipper library)
    #    - distance = polygon_area * unclip_ratio / polygon_perimeter
    #    - pyclipper.PyclipperOffset().Execute(distance)
    # 5. Compute minAreaRect on expanded polygon
    # 6. Scale box coordinates back to original image size
    # Returns: [{"boxes": np.array(N,4,2), "scores": np.array(N,)}]
```

**Dependency note:** `pyclipper` is required for unclip. Lightweight C++ extension,
pip-installable.

#### Recognition Pre-processing
```python
def preprocess_rec(images: list[np.ndarray], target_height: int = 48,
                   max_width: int = 3200) -> np.ndarray:
    # 1. For each crop: resize to height=48, preserve aspect ratio
    # 2. Normalize: (img / 255.0 - 0.5) / 0.5  → range [-1, 1]
    #    NOTE: PaddleOCR uses [-1,1] normalization, NOT ImageNet stats.
    #    Must verify which the HF Transformers rec model expects.
    # 3. Pad width to batch max width (dynamic per-batch, not fixed)
    # 4. Stack into batch: (B, 3, 48, max_width_in_batch)
    # Returns: batched_tensor
```

#### Recognition Post-processing (CTC decode — matches PaddleOCR CTCLabelDecode)
```python
def postprocess_rec(logits: np.ndarray,
                    character_list: list[str]) -> list[tuple[str, float]]:
    # logits shape: (B, T, 18385) — softmax already applied by model
    # 1. preds_idx = argmax(logits, axis=-1)  → (B, T)
    # 2. preds_prob = max(logits, axis=-1)    → (B, T)
    # 3. For each sequence in batch:
    #    a. Remove consecutive duplicates: idx[t] != idx[t-1]
    #    b. Remove blank token: idx != 0
    #    c. Map remaining indices → chars via character_list
    #    d. Confidence = mean of probs at selected positions
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
    "pyclipper",
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

## Open Questions (must verify before implementing)

1. **Normalization mismatch**: HF Transformers image processors use ImageNet stats
   `(mean=[0.406,0.456,0.485], std=[0.225,0.224,0.229])`, but PaddleOCR rec uses
   `(x/255 - 0.5) / 0.5` → `[-1,1]`. The model weights were trained with
   PaddleOCR's normalization. Need to check which normalization the HF-ported
   weights actually expect — look at the HF image processor code.

2. **Det output resolution**: PaddleOCR's DBNet outputs at feature map resolution
   (typically H/4, W/4), then scales boxes back up. The HF model may output at
   full resolution. Verify output shape against input shape.

3. **Character dictionary source**: The 18385-char dict needs to ship with this
   tool or be downloaded from HF Hub. Check if it's embedded in the HF model
   config or needs separate distribution.

## Alignment with PaddleOCR Pipeline

This tool replicates PaddleOCR's inference pipeline:
- **Det**: DBNet post-processing with pyclipper unclip (same params)
- **Rec**: CTC greedy decode with blank removal (same algorithm)
- **Pipeline**: det → crop → batch rec → decode (same flow)
- **Params**: thresh=0.3, box_thresh=0.6, unclip_ratio=2.0 (same defaults)

Replacing only the inference backend (Paddle → TRT via ONNX).

## Notes

- Detection models output at original spatial resolution (not downsampled)
- Recognition models apply softmax internally — output is probabilities, not raw logits
- All models set `_can_compile_fullgraph = True` — clean for tracing
- No control flow branching — safe for ONNX static graph export
