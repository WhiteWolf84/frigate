# syntax=docker/dockerfile:1.4

# AMD XDNA NPU image variant. Targets Ryzen AI / Phoenix-class APUs (8945HS etc.)
# via onnxruntime with VitisAI Execution Provider.
#
# Requires a host kernel with amdxdna driver (mainline >= 6.10) and
# /dev/accel/accel0 passed through to the container.
#
# Firmware source: RyzenAI-SW v1.3.1 (Phoenix / ryzen12 support).
#   xclbin:      example/transformers/xclbin/phx/1x4.xclbin
#   vaip_config: tutorial/yolov8/yolov8_python/vaip_config.json
#
# ARGs (override via --set or buildx bake args):
#   RYZEN_AI_VERSION   SDK tag, default "1.3.1" (maps to GitHub tag "v1.3.1")
#   RYZEN_AI_SDK_URL   full URL to the SDK archive

ARG DEBIAN_FRONTEND=noninteractive
ARG RYZEN_AI_VERSION=1.3.1
ARG RYZEN_AI_SDK_URL=https://github.com/amd/RyzenAI-SW/archive/refs/tags/v1.3.1.tar.gz

#######################################################################
# Stage: download and unpack the RyzenAI-SW SDK; extract firmware blobs
FROM wget AS ryzen-ai-sdk

ARG RYZEN_AI_SDK_URL

RUN mkdir -p /opt/ryzen-ai-sdk && \
    wget -q "${RYZEN_AI_SDK_URL}" -O /tmp/ryzen_ai_sdk.tgz && \
    tar -xzf /tmp/ryzen_ai_sdk.tgz -C /opt/ryzen-ai-sdk --strip-components=1 && \
    rm /tmp/ryzen_ai_sdk.tgz

# Phoenix (phx / ryzen12) xclbin is at the known path in v1.3.1.
# vaip_config.json is the one from the YOLOv8 Python tutorial (ONNX inference).
RUN mkdir -p /opt/ryzen-ai-artifacts/firmware && \
    cp /opt/ryzen-ai-sdk/example/transformers/xclbin/phx/1x4.xclbin \
       /opt/ryzen-ai-artifacts/firmware/1x4.xclbin && \
    cp /opt/ryzen-ai-sdk/tutorial/yolov8/yolov8_python/vaip_config.json \
       /opt/ryzen-ai-artifacts/firmware/vaip_config.json && \
    test -f /opt/ryzen-ai-artifacts/firmware/1x4.xclbin || \
        { echo "ERROR: 1x4.xclbin not found in SDK archive"; exit 1; } && \
    test -f /opt/ryzen-ai-artifacts/firmware/vaip_config.json || \
        { echo "ERROR: vaip_config.json not found in SDK archive"; exit 1; }

#######################################################################
FROM deps AS amd-npu-deps

# Runtime libs needed by XRT userspace and VAIP shared libraries
RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends \
        libdrm2 libudev1 ocl-icd-libopencl1 libgomp1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /opt/frigate
COPY --from=rootfs / /

RUN wget -q https://bootstrap.pypa.io/get-pip.py -O get-pip.py && \
    python3 get-pip.py "pip" --break-system-packages
RUN python3 -m pip config set global.break-system-packages true

# onnxruntime-vitisai is distributed via AMD's installation tooling (conda/pip),
# not as a standalone wheel in the GitHub source archive.
# Install the latest onnxruntime-vitisai from PyPI (AMD publishes it there
# starting from Ryzen AI SDK 1.x). If the PyPI package is unavailable, replace
# this line with the path to a locally-vendored wheel (see README.md).
COPY docker/amd_npu/requirements-wheels-amd-npu.txt /requirements-amd-npu.txt
RUN pip3 uninstall -y onnxruntime onnxruntime-gpu || true && \
    pip3 install -r /requirements-amd-npu.txt

# Firmware: Phoenix 4-column overlay bitstream + VAIP op config
COPY --from=ryzen-ai-sdk /opt/ryzen-ai-artifacts/firmware/1x4.xclbin \
     /opt/xilinx/xrt/amdxdna/1x4.xclbin
COPY --from=ryzen-ai-sdk /opt/ryzen-ai-artifacts/firmware/vaip_config.json \
     /opt/vaip/etc/vaip_config.json

# Default env vars for Phoenix; the amd_npu detector overrides these at runtime
# from its config, but having them set here lets the container be tested without
# the detector config (e.g., with the bare onnxruntime Python API).
ENV XLNX_VART_FIRMWARE=/opt/xilinx/xrt/amdxdna/1x4.xclbin
ENV XLNX_TARGET_NAME=AMD_AIE2_Nx4_Overlay
ENV NUM_OF_DPU_RUNNERS=1
ENV LD_LIBRARY_PATH=/opt/xilinx/xrt/lib:$LD_LIBRARY_PATH

#######################################################################
FROM amd-npu-deps AS amd_npu
