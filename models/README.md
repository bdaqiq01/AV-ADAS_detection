# Model conversion: PyTorch → ONNX → TensorRT (Jetson Orin Nano)

Two trained Ultralytics YOLO detectors live here:

| File            | Task   | Classes                        |
| --------------- | ------ | ------------------------------ |
| `speedlimit.pt` | detect | 14 (10 MPH … 75 MPH)           |
| `stop.pt`       | detect | 1 (stop)                       |

Deploy target: **Jetson Orin Nano** (JetPack 6.x, TensorRT 8.6+).

## Why two steps

TensorRT `.engine` files are tied to the exact GPU architecture, TensorRT
version, CUDA version, and JetPack version. An engine built on an x86
workstation will **not** load on the Orin. The portable format is ONNX,
so we do:

```
.pt  ──(on dev machine)──▶  .onnx  ──(on the Jetson)──▶  .engine
```

## Step 1 — Export .pt → .onnx (run on the dev machine)

From the repo root, with the project venv activated:

```bash
python converting_models/export_onnx.py
```

Settings used (in `export_onnx.py`):

- `imgsz=640` — matches YOLO11 training default. Change if you trained at a different size.
- `opset=12` — broad TensorRT compatibility.
- `simplify=True` — cleaner graph, better TRT conversion.
- `dynamic=False` — static 1×3×640×640 input for fastest, most stable engine.
- `half=False` — keep ONNX in FP32; quantize at engine build time.

Outputs (next to their `.pt` siblings):

- `converting_models/speedlimit.onnx` — input `images [1,3,640,640]`, output `output0 [1,18,8400]`
- `converting_models/stop.onnx`       — input `images [1,3,640,640]`, output `output0 [1,5,8400]`

YOLO head layout: `output0 = [batch, 4 + num_classes, num_anchors]`, i.e. 4 bbox
values followed by per-class scores for each of the 8400 anchor points.

## Step 2 — Build .engine on the Jetson

Copy the two `.onnx` files and `build_engine.sh` to the Orin Nano, then:

```bash
# optional but strongly recommended
sudo nvpmodel -m 0      # MAXN (max power)
sudo jetson_clocks       # lock clocks high

chmod +x build_engine.sh
./build_engine.sh                # FP16 (recommended)
# or:
PRECISION=fp32 ./build_engine.sh
```

This runs `trtexec` and produces `speedlimit.engine` and `stop.engine` next to
the ONNX files. Expect ~1–5 minutes per model the first time.

FP16 is the sweet spot on Orin: ~2× faster than FP32 with negligible accuracy
loss for YOLO detection. INT8 is faster still but requires a calibration
dataset — skip unless you need the extra throughput.

## Step 3 — Use the engines

Easiest: feed the `.engine` directly to Ultralytics on the Jetson:

```python
from ultralytics import YOLO
model = YOLO("speedlimit.engine")   # or stop.engine
results = model.predict("image.jpg", imgsz=640)
```

For lower-level C++/DeepStream integration, load the engine with the TensorRT
runtime and provide your own pre/post-processing (letterbox to 640×640,
normalize to 0–1, then NMS on `output0`).

## Troubleshooting

- `trtexec: command not found` — use full path `/usr/src/tensorrt/bin/trtexec`
  or install TensorRT via JetPack.
- Out of memory while building — lower `WORKSPACE_MIB` (default 4096) or close
  other GPU apps. The Orin Nano 8 GB is tight.
- Wrong input size at inference — rebuild ONNX with a matching `IMGSZ` in
  `export_onnx.py`; the engine is locked to that shape because we used
  `dynamic=False`.
- Accuracy drop after FP16 — try `PRECISION=fp32` to confirm it's a
  quantization effect; if so and you need speed, add INT8 calibration.

## Files in this folder

```
speedlimit.pt        speedlimit.onnx     (speedlimit.engine — built on Jetson)
stop.pt              stop.onnx           (stop.engine       — built on Jetson)
export_onnx.py       # .pt  -> .onnx   (run on dev machine)
build_engine.sh      # .onnx -> .engine (run on Jetson Orin Nano)
README.md
```
