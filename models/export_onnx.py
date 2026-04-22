"""
Export Ultralytics YOLO .pt weights to ONNX, tuned for TensorRT on Jetson Orin Nano.

Output settings:
    - opset=12              (broad TRT compatibility)
    - simplify=True         (onnx-simplifier pass)
    - dynamic=False         (static shapes => fastest/most stable TRT engine)
    - half=False            (keep ONNX in FP32; quantize to FP16 at engine build time)
    - imgsz=640             (standard YOLO input; change if you trained at a different size)

Run on the x86 dev machine. Copy the resulting .onnx files to the Jetson,
then build the TensorRT .engine there with build_engine.sh.
"""

from pathlib import Path

from ultralytics import YOLO

HERE = Path(__file__).resolve().parent

MODELS = [
    HERE / "speedlimit.pt",
    HERE / "stop.pt",
]

IMGSZ = 640
OPSET = 12


def export(weights: Path) -> Path:
    if not weights.exists():
        raise FileNotFoundError(weights)

    print(f"\n=== Exporting {weights.name} -> ONNX ===")
    model = YOLO(str(weights))
    print(f"  task   : {model.task}")
    print(f"  classes: {len(model.names)}")

    out = model.export(
        format="onnx",
        imgsz=IMGSZ,
        opset=OPSET,
        simplify=True,
        dynamic=False,
        half=False,
        device="cpu",
    )

    out_path = Path(out)
    print(f"  wrote  : {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")
    return out_path


def main() -> None:
    for w in MODELS:
        export(w)

    print("\nAll ONNX files are ready next to their .pt counterparts.")
    print("Next step: copy them to the Jetson Orin Nano and run build_engine.sh there.")


if __name__ == "__main__":
    main()
