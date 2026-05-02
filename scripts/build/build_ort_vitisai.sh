#!/usr/bin/env bash
# Build onnxruntime with VitisAI Execution Provider for Phoenix NPU on Linux.
# Target: Debian 12, Python 3.11, x86_64.
#
# Chain: XRT (amd/xdna-driver) → VAIP stack (xir + vart + vaip) → onnxruntime
# Estimated time on 5800X3D (j=16): ~1h 30m total
#
# Run inside the Dockerfile.build container. Output goes to /output/.

set -euo pipefail

JOBS="${JOBS:-16}"
INSTALL_PREFIX="/usr/local"
OUTPUT_DIR="/output"
BUILD_BASE="/tmp/npu-build"

mkdir -p "$OUTPUT_DIR" "$BUILD_BASE"

log() { echo; echo "════════════════════════════════════════════════════"; echo "  $*"; echo "════════════════════════════════════════════════════"; }

# ── Phase 1: XRT from amd/xdna-driver (~10 min) ─────────────────────────────
log "Phase 1/3 — XRT userspace shim (amd/xdna-driver)"

cd "$BUILD_BASE"
git clone --depth=1 https://github.com/amd/xdna-driver xdna-driver
cd xdna-driver

# The xdna-driver repo contains both the kernel module and the XRT userspace
# shim for XDNA. We only build the userspace part (no kernel module needed
# here — that's already loaded on the Proxmox host).
mkdir -p build && cd build
cmake ../src/shim \
    -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX" \
    -DCMAKE_BUILD_TYPE=Release \
    -DXRT_ENABLE_WERROR=OFF \
    -GNinja
ninja -j"$JOBS"
ninja install
ldconfig

log "XRT shim installed"

# ── Phase 2: VAIP runtime stack (~20 min) ────────────────────────────────────
log "Phase 2/3 — VAIP runtime (xir + vart + vaip)"

# XIR (Xilinx Intermediate Representation) — graph representation layer
cd "$BUILD_BASE"
git clone --depth=1 https://github.com/Xilinx/xir xir
cd xir && mkdir -p build && cd build
cmake .. \
    -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX" \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=ON \
    -GNinja
ninja -j"$JOBS" && ninja install
ldconfig

# VART (Vitis AI Runtime) — runner abstraction layer
cd "$BUILD_BASE"
git clone --depth=1 https://github.com/Xilinx/vart vart
cd vart && mkdir -p build && cd build
cmake .. \
    -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX" \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=ON \
    -DENABLE_CPU_RUNNER=ON \
    -GNinja
ninja -j"$JOBS" && ninja install
ldconfig

# VAIP (VitisAI Inference Processor) — the EP runtime core
cd "$BUILD_BASE"
git clone --depth=1 https://github.com/amd/vaip vaip
cd vaip && mkdir -p build && cd build
cmake .. \
    -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX" \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=ON \
    -DXRT_DIR="$INSTALL_PREFIX" \
    -GNinja
ninja -j"$JOBS" && ninja install
ldconfig

log "VAIP stack installed"

# ── Phase 3: onnxruntime + VitisAI EP wheel (~60 min) ────────────────────────
log "Phase 3/3 — onnxruntime wheel with VitisAI EP"

cd "$BUILD_BASE"
git clone --depth=1 --branch v1.20.1 \
    https://github.com/microsoft/onnxruntime onnxruntime
cd onnxruntime

# v1.20.1 matches the libonnxruntime.so.1.20.1 found in RyzenAI-SW v1.7.1 Linux
./build.sh \
    --config Release \
    --use_vitisai \
    --build_wheel \
    --parallel "$JOBS" \
    --skip_tests \
    --cmake_extra_defines \
        CMAKE_PREFIX_PATH="$INSTALL_PREFIX" \
        VAIP_INSTALL_DIR="$INSTALL_PREFIX" \
        XRT_DIR="$INSTALL_PREFIX"

# ── Collect artifacts ─────────────────────────────────────────────────────────
log "Collecting artifacts → $OUTPUT_DIR"

# Python wheel
find build/Linux/Release/dist -name "onnxruntime_vitisai-*.whl" \
    -exec cp {} "$OUTPUT_DIR/" \; 2>/dev/null || \
find build/Linux/Release/dist -name "onnxruntime-*.whl" \
    -exec cp {} "$OUTPUT_DIR/" \;

# VAIP shared libraries (needed at runtime via LD_LIBRARY_PATH)
mkdir -p "$OUTPUT_DIR/lib"
find "$INSTALL_PREFIX/lib" -name "libvaip*.so*" \
    -o -name "libvart*.so*" \
    -o -name "libxir*.so*" \
    -o -name "libonnxruntime_providers_vitisai*.so*" \
    -o -name "libonnxruntime_vitisai_ep*.so*" \
    | xargs -I{} cp -P {} "$OUTPUT_DIR/lib/" 2>/dev/null || true

# XRT runtime libs
find "$INSTALL_PREFIX/lib" -name "libxrt*.so*" \
    | xargs -I{} cp -P {} "$OUTPUT_DIR/lib/" 2>/dev/null || true

# ── Extract firmware from RyzenAI-SW v1.3.1 ──────────────────────────────────
log "Fetching Phoenix firmware from RyzenAI-SW v1.3.1"

mkdir -p "$OUTPUT_DIR/firmware"
cd /tmp
wget -q https://github.com/amd/RyzenAI-SW/archive/refs/tags/v1.3.1.tar.gz \
    -O ryzen-ai-1.3.1.tgz
tar -xzf ryzen-ai-1.3.1.tgz \
    "RyzenAI-SW-1.3.1/example/transformers/xclbin/phx/1x4.xclbin" \
    "RyzenAI-SW-1.3.1/tutorial/yolov8/yolov8_python/vaip_config.json"
cp RyzenAI-SW-1.3.1/example/transformers/xclbin/phx/1x4.xclbin \
    "$OUTPUT_DIR/firmware/1x4.xclbin"
cp RyzenAI-SW-1.3.1/tutorial/yolov8/yolov8_python/vaip_config.json \
    "$OUTPUT_DIR/firmware/vaip_config.json"
rm -rf RyzenAI-SW-1.3.1 ryzen-ai-1.3.1.tgz

# ── Summary ───────────────────────────────────────────────────────────────────
log "Build complete"
echo "Artifacts in $OUTPUT_DIR:"
ls -lh "$OUTPUT_DIR/"
ls -lh "$OUTPUT_DIR/lib/" | head -20
ls -lh "$OUTPUT_DIR/firmware/"
echo
echo "Next step: run scripts/build/deploy_to_lxc.sh"
