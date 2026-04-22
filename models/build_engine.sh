#!/usr/bin/env bash
# Build TensorRT .engine files from .onnx on the Jetson Orin Nano.
#
# Run this ON THE JETSON (not on the x86 dev machine). TensorRT engines are
# hardware/version-specific and will not load on a different device.
#
# Prereqs on the Jetson (already present in JetPack 6.x):
#   - /usr/src/tensorrt/bin/trtexec
#   - CUDA + cuDNN + TensorRT from JetPack
#
# Usage:
#   chmod +x build_engine.sh
#   ./build_engine.sh            # builds FP16 engines for speedlimit + stop
#   PRECISION=fp32 ./build_engine.sh
#
# Tip: before running, max out the board for a faster build and for best
# inference later:
#   sudo nvpmodel -m 0          # MAXN power mode
#   sudo jetson_clocks           # lock clocks to max

set -euo pipefail

cd "$(dirname "$0")"

TRTEXEC="${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}"
PRECISION="${PRECISION:-fp16}"        # fp16 (recommended on Orin) or fp32
WORKSPACE_MIB="${WORKSPACE_MIB:-4096}" # Orin Nano 8GB: 4 GiB is safe

if [[ ! -x "$TRTEXEC" ]]; then
    echo "ERROR: trtexec not found at $TRTEXEC"
    echo "Set TRTEXEC=/path/to/trtexec or install TensorRT (JetPack)."
    exit 1
fi

case "$PRECISION" in
    fp16) FLAGS="--fp16" ;;
    fp32) FLAGS="" ;;
    *) echo "Unsupported PRECISION=$PRECISION (use fp16 or fp32)"; exit 1 ;;
esac

build() {
    local onnx="$1"
    local engine="${onnx%.onnx}.engine"

    if [[ ! -f "$onnx" ]]; then
        echo "Skipping $onnx (not found)"
        return
    fi

    echo ""
    echo "=== Building $engine  [$PRECISION] ==="
    "$TRTEXEC" \
        --onnx="$onnx" \
        --saveEngine="$engine" \
        --memPoolSize=workspace:${WORKSPACE_MIB} \
        $FLAGS

    echo "Built: $engine"
    ls -lh "$engine"
}

build "speedlimit.onnx"
build "stop.onnx"

echo ""
echo "Done. Use these .engine files with TensorRT / DeepStream / Ultralytics."
