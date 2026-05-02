#!/usr/bin/env bash
# Deploy the amd_npu detector plugin into an existing community-script
# Frigate LXC installation. Idempotent — safe to re-run.
#
# Run from INSIDE the Frigate LXC, as root:
#   curl -fsSL https://raw.githubusercontent.com/WhiteWolf84/frigate/feature/amd-npu-detector/scripts/deploy_amd_npu_to_lxc.sh | bash
# or, after copying the file into the LXC:
#   bash deploy_amd_npu_to_lxc.sh

set -euo pipefail

FRIGATE_DIR="${FRIGATE_DIR:-/opt/frigate}"
FORK_RAW="https://raw.githubusercontent.com/WhiteWolf84/frigate/feature/amd-npu-detector"

if [[ ! -d "$FRIGATE_DIR" ]]; then
    echo "ERROR: $FRIGATE_DIR not found. Is Frigate installed via community-scripts?" >&2
    exit 1
fi

echo "==> Verifying NPU device access"
if [[ ! -e /dev/accel/accel0 ]]; then
    echo "ERROR: /dev/accel/accel0 not found. Add LXC passthrough first." >&2
    exit 2
fi
ls -l /dev/accel/accel0

echo "==> Backing up existing plugin (if any)"
if [[ -f "$FRIGATE_DIR/frigate/detectors/plugins/amd_npu.py" ]]; then
    cp "$FRIGATE_DIR/frigate/detectors/plugins/amd_npu.py" \
       "$FRIGATE_DIR/frigate/detectors/plugins/amd_npu.py.bak.$(date +%s)"
fi

echo "==> Downloading amd_npu plugin from fork"
curl -fsSL "$FORK_RAW/frigate/detectors/plugins/amd_npu.py" \
    -o "$FRIGATE_DIR/frigate/detectors/plugins/amd_npu.py"

echo "==> Verifying plugin syntax"
python3 -m py_compile "$FRIGATE_DIR/frigate/detectors/plugins/amd_npu.py"

echo "==> Plugin deployed."
echo
echo "Next steps:"
echo "  1. Add to /config/config.yml:"
echo "       detectors:"
echo "         npu0:"
echo "           type: amd_npu"
echo "  2. Restart Frigate:    systemctl restart frigate"
echo "  3. Watch logs:         journalctl -u frigate -f | grep -i 'amd npu\\|vitisai'"
echo
echo "Expected: a clear RuntimeError about onnxruntime-vitisai or"
echo "VitisAIExecutionProvider missing. That confirms the plugin loads"
echo "and registers correctly, even before VAIP runtime is built."
