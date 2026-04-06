# pp-ocrv5-trt

TensorRT export and inference for PP-OCRv5 models from [HuggingFace Transformers](https://huggingface.co/docs/transformers).

Export pretrained or finetuned PP-OCRv5 models to TensorRT engines for high-throughput, low-latency OCR inference — no PaddlePaddle dependency required.

## Performance

Tested on NVIDIA RTX 4070 Super, TensorRT 10.16, FP16, 200 iterations:

### Latency

| Model | Input Size | PyTorch (median) | TRT FP16 (median) | Speedup |
|-------|-----------|-----------------|-------------------|---------|
| server_det | 640x640 | 16.70 ms | 5.80 ms | **2.88x** |
| server_rec | 48x320 | 6.82 ms | 2.35 ms | **2.90x** |
| mobile_det | 640x640 | 10.10 ms | 3.43 ms | **2.95x** |
| mobile_rec | 48x320 | 12.18 ms | 2.97 ms | **4.10x** |

### Throughput

| Model | PyTorch (infer/s) | TRT FP16 (infer/s) | Gain |
|-------|------------------|-------------------|------|
| server_det | 57.5 | 166.7 | 2.9x |
| server_rec | 136.1 | 374.5 | 2.8x |
| mobile_det | 94.4 | 281.1 | 3.0x |
| mobile_rec | 67.6 | 298.4 | 4.4x |

### Accuracy (TRT FP16 vs PyTorch FP32)

| Model | Max abs diff | Cosine similarity |
|-------|-------------|-------------------|
| server_det | 0.000836 | 0.9998 |
| server_rec | 0.005075 | 0.9999 |
| mobile_det | 0.000000 | 0.9482 |
| mobile_rec | 0.017378 | 0.9999 |

TensorRT FP16 delivers **~3-4x speedup** over PyTorch with negligible accuracy loss.

## Supported Models

All 4 PP-OCRv5 models from `transformers>=5.0`:

| Shorthand | HuggingFace Model ID | Task |
|-----------|---------------------|------|
| `server_det` | [`PaddlePaddle/PP-OCRv5_server_det_safetensors`](https://huggingface.co/PaddlePaddle/PP-OCRv5_server_det_safetensors) | Text detection (server) |
| `mobile_det` | [`PaddlePaddle/PP-OCRv5_mobile_det_safetensors`](https://huggingface.co/PaddlePaddle/PP-OCRv5_mobile_det_safetensors) | Text detection (mobile) |
| `server_rec` | [`PaddlePaddle/PP-OCRv5_server_rec_safetensors`](https://huggingface.co/PaddlePaddle/PP-OCRv5_server_rec_safetensors) | Text recognition (server) |
| `mobile_rec` | [`PaddlePaddle/PP-OCRv5_mobile_rec_safetensors`](https://huggingface.co/PaddlePaddle/PP-OCRv5_mobile_rec_safetensors) | Text recognition (mobile) |

Works with pretrained weights or your own finetuned checkpoints.

## Installation

```bash
# Core runtime (inference only, no PyTorch needed)
pip install pp-ocrv5-trt[trt]

# With export support (needs PyTorch + Transformers)
pip install pp-ocrv5-trt[all]
```

**Requirements:**
- Python >= 3.10
- NVIDIA GPU with CUDA
- TensorRT >= 10.0 (for inference)
- PyTorch >= 2.0 + Transformers >= 5.0 (for export only)

## Quick Start

### Export a model

```bash
# Export pretrained model to TRT engine (FP16)
pp-ocrv5-trt export --model server_det --output ./engines/ --precision fp16

# Export a finetuned model
pp-ocrv5-trt export --model ./my-finetuned-det --output ./engines/

# Export with full HF model ID
pp-ocrv5-trt export --model PaddlePaddle/PP-OCRv5_server_rec_safetensors --output ./engines/
```

### Run inference

```bash
# Detection
pp-ocrv5-trt infer --engine ./engines/server_det.trt --image photo.jpg

# Recognition (requires character list)
pp-ocrv5-trt infer --engine ./engines/server_rec.trt --image crop.jpg --char-list chars.txt
```

### End-to-end OCR

```bash
pp-ocrv5-trt e2e \
  --det-engine ./engines/server_det.trt \
  --rec-engine ./engines/server_rec.trt \
  --char-list chars.txt \
  --image document.jpg
```

### Benchmark

```bash
pp-ocrv5-trt bench --engine ./engines/server_det.trt --iterations 100
```

### Python API

```python
from pp_ocrv5_trt.export import export
from pp_ocrv5_trt.runtime import TrtModel
from pp_ocrv5_trt.pipeline import OCRPipeline
from pp_ocrv5_trt.processing import preprocess_det, postprocess_det

# Export
engine_path = export("server_det", "./engines/", precision="fp16")

# Single model inference
model = TrtModel("./engines/server_det.trt")
output = model(preprocessed_input)  # numpy array in, numpy array out

# End-to-end pipeline
pipeline = OCRPipeline(
    det_engine="./engines/server_det.trt",
    rec_engine="./engines/server_rec.trt",
    character_list_path="chars.txt",
)
results = pipeline(cv2.imread("document.jpg"))
for r in results:
    print(r["text"], r["confidence"])
```

## How It Works

```
HF Hub / local checkpoint
        |
        v
  [torch.onnx.export]  ── ONNX model (with dynamic axes)
        |
        v
  [TensorRT builder]   ── TRT engine (.trt file, FP16/FP32)
        |
        v
  [TRT Runtime]        ── numpy in → GPU inference → numpy out
```

**Export** (one-time, requires PyTorch): loads the HF model, traces to ONNX with dynamic batch/spatial axes, builds a TRT engine with optimization profiles.

**Inference** (no PyTorch needed): loads the `.trt` engine, runs on GPU via TensorRT Python bindings + cuda-python. Only depends on numpy, opencv, and tensorrt.

**Processing** matches PaddleOCR's pipeline:
- Detection: DBNet post-processing with pyclipper polygon unclip (thresh=0.3, box_thresh=0.6, unclip_ratio=2.0)
- Recognition: CTC greedy decode with blank removal (18,385-char multilingual vocabulary)

## Project Structure

```
pp_ocrv5_trt/
    export.py       # HF model -> ONNX -> TRT engine
    runtime.py      # TRT engine inference (no torch)
    processing.py   # Det/rec pre/post processing
    pipeline.py     # End-to-end OCR pipeline
    cli.py          # Command-line interface
```

## License

Apache-2.0
