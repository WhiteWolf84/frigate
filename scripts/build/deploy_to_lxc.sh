#!/usr/bin/env bash
# Deploy onnxruntime-vitisai artifacts to the Frigate community-script LXC.
# Run this on the Proxmox HOST after the Docker build completes.
#
# Usage:
#   LXC_ID=100 bash scripts/build/deploy_to_lxc.sh
#   # or with explicit output dir:
#   LXC_ID=100 OUTPUT_DIR=scripts/build/output bash scripts/build/deploy_to_lxc.sh

set -euo pipefail

LXC_ID="${LXC_ID:?Set LXC_ID to your Frigate container ID, e.g. LXC_ID=100}"
OUTPUT_DIR="${OUTPUT_DIR:-$(dirname "$0")/output}"
FRIGATE_DIR="/opt/frigate"
VAIP_LIB_DIR="/opt/vaip/lib"
FIRMWARE_DIR="/opt/vaip/etc"
XCLBIN_DIR="/opt/xilinx/xrt/amdxdna"

log() { echo; echo "▶ $*"; }

# ── Verify artifacts exist ────────────────────────────────────────────────────
log "Checking artifacts in $OUTPUT_DIR"
[[ -d "$OUTPUT_DIR" ]] || { echo "ERROR: $OUTPUT_DIR not found. Run the Docker build first."; exit 1; }

WHL=$(find "$OUTPUT_DIR" -name "onnxruntime*.whl" | head -1)
[[ -n "$WHL" ]] || { echo "ERROR: No onnxruntime wheel found in $OUTPUT_DIR"; exit 1; }

[[ -f "$OUTPUT_DIR/firmware/1x4.xclbin" ]]    || { echo "ERROR: 1x4.xclbin missing"; exit 1; }
[[ -f "$OUTPUT_DIR/firmware/vaip_config.json" ]] || { echo "ERROR: vaip_config.json missing"; exit 1; }

echo "  Wheel:    $WHL"
echo "  Libs:     $(ls "$OUTPUT_DIR/lib/" 2>/dev/null | wc -l) .so files"
echo "  Firmware: 1x4.xclbin + vaip_config.json"

# ── Copy artifacts into LXC via pct push ─────────────────────────────────────
log "Pushing wheel to LXC $LXC_ID"
pct push "$LXC_ID" "$WHL" "/tmp/$(basename "$WHL")"

log "Pushing VAIP libraries to LXC $LXC_ID"
pct exec "$LXC_ID" -- mkdir -p "$VAIP_LIB_DIR"
for lib in "$OUTPUT_DIR"/lib/*.so*; do
    [[ -L "$lib" ]] && continue   # skip symlinks; we'll fix them inside
    pct push "$LXC_ID" "$lib" "$VAIP_LIB_DIR/$(basename "$lib")"
done

log "Pushing firmware to LXC $LXC_ID"
pct exec "$LXC_ID" -- mkdir -p "$FIRMWARE_DIR" "$XCLBIN_DIR"
pct push "$LXC_ID" "$OUTPUT_DIR/firmware/1x4.xclbin" \
    "$XCLBIN_DIR/1x4.xclbin"
pct push "$LXC_ID" "$OUTPUT_DIR/firmware/vaip_config.json" \
    "$FIRMWARE_DIR/vaip_config.json"

# ── Plugin source ─────────────────────────────────────────────────────────────
log "Deploying amd_npu plugin"
PLUGIN_SRC="$(dirname "$0")/../../frigate/detectors/plugins/amd_npu.py"
[[ -f "$PLUGIN_SRC" ]] && \
    pct push "$LXC_ID" "$PLUGIN_SRC" "$FRIGATE_DIR/frigate/detectors/plugins/amd_npu.py"

# ── Inside LXC: install wheel and configure environment ───────────────────────
log "Installing inside LXC $LXC_ID"
pct exec "$LXC_ID" -- bash -c "
set -e
echo '==> Uninstalling stock onnxruntime'
pip3 uninstall -y onnxruntime onnxruntime-gpu 2>/dev/null || true

echo '==> Installing onnxruntime-vitisai wheel'
pip3 install --break-system-packages /tmp/$(basename "$WHL")

echo '==> Running ldconfig for VAIP libs'
echo '$VAIP_LIB_DIR' >> /etc/ld.so.conf.d/vaip.conf
ldconfig

echo '==> Verifying VitisAI EP is visible'
python3 -c \"
import onnxruntime
providers = onnxruntime.get_available_providers()
print('Available providers:', providers)
if 'VitisAIExecutionProvider' in providers:
    print('✓ VitisAIExecutionProvider FOUND')
else:
    print('✗ VitisAIExecutionProvider NOT found — check LD_LIBRARY_PATH')
\"
"

# ── Add LD_LIBRARY_PATH to Frigate's environment file ─────────────────────────
log "Adding VAIP lib path to Frigate env"
pct exec "$LXC_ID" -- bash -c "
grep -q VAIP_LIB /etc/frigate.env 2>/dev/null || \
    echo 'LD_LIBRARY_PATH=$VAIP_LIB_DIR:\${LD_LIBRARY_PATH:-}' >> /etc/frigate.env
"

log "Done. Restart Frigate and check logs:"
echo "  pct exec $LXC_ID -- systemctl restart frigate"
echo "  pct exec $LXC_ID -- journalctl -u frigate -f | grep -i 'amd npu\\|vitisai'"
