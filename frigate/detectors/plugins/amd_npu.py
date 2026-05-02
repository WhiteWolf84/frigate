"""AMD XDNA NPU detector for Frigate (Ryzen AI / Phoenix-class APUs).

Targets the AMD XDNA NPU exposed via /dev/accel/accel0 on Linux >= 6.10.
Inference runs through onnxruntime with VitisAIExecutionProvider; ops not
compiled to the NPU fall back to CPUExecutionProvider (mandatory tail).

Model requirements (per VAIP):
  - Static input shapes (no dynamic batch / sequence dims).
  - INT8-quantized ONNX (QDQ or Vitis AI Quantizer output).
  - Standard NCHW or NHWC layout matching ModelConfig.input_tensor.

Runtime configuration uses environment variables as AMD intends:
  XLNX_VART_FIRMWARE  → path to the xclbin overlay bitstream
  XLNX_TARGET_NAME    → overlay name (AMD_AIE2_Nx4_Overlay for Phoenix)
  NUM_OF_DPU_RUNNERS  → parallel DPU threads (default 1)
These are set from the detector config at __init__ time so Frigate controls
them without requiring the user to set them in the host environment.
"""

import logging
import os

import numpy as np
from pydantic import Field
from typing_extensions import Literal

from frigate.detectors.detection_api import DetectionApi
from frigate.detectors.detection_runners import ONNXModelRunner
from frigate.detectors.detector_config import BaseDetectorConfig, ModelTypeEnum
from frigate.util.model import (
    post_process_dfine,
    post_process_rfdetr,
    post_process_yolo,
    post_process_yolox,
)

logger = logging.getLogger(__name__)

DETECTOR_KEY = "amd_npu"

DEFAULT_XCLBIN = "/opt/xilinx/xrt/amdxdna/1x4.xclbin"
DEFAULT_VAIP_CONFIG = "/opt/vaip/etc/vaip_config.json"
# Phoenix (7040/8040) uses AMD_AIE2_Nx4_Overlay.
# Hawk Point (Ryzen AI 300 / ryzen14) uses AMD_AIE2P_Nx4_Overlay.
DEFAULT_TARGET = "AMD_AIE2_Nx4_Overlay"


class AmdNpuDetectorConfig(BaseDetectorConfig):
    type: Literal[DETECTOR_KEY]
    xclbin: str = Field(default=DEFAULT_XCLBIN, title="NPU bitstream (xclbin) path")
    config_file: str = Field(default=DEFAULT_VAIP_CONFIG, title="vaip_config.json path")
    target: str = Field(
        default=DEFAULT_TARGET,
        title="VAIP overlay target (AMD_AIE2_Nx4_Overlay for Phoenix, "
        "AMD_AIE2P_Nx4_Overlay for Hawk Point / Strix)",
    )
    num_dpu_runners: int = Field(
        default=1, ge=1, le=4, title="Number of parallel DPU runner threads"
    )


class AmdNpuDetector(DetectionApi):
    type_key = DETECTOR_KEY

    def __init__(self, detector_config: AmdNpuDetectorConfig):
        super().__init__(detector_config)

        try:
            import onnxruntime as ort
        except ImportError as e:
            raise RuntimeError(
                "amd_npu detector requires onnxruntime-vitisai. "
                "Install the AMD Ryzen AI ONNX Runtime build inside this "
                "container/venv. See docker/amd_npu/Dockerfile."
            ) from e

        available = ort.get_available_providers()
        if "VitisAIExecutionProvider" not in available:
            raise RuntimeError(
                f"VitisAIExecutionProvider not available in onnxruntime; "
                f"providers seen: {available}. Verify onnxruntime-vitisai "
                f"is installed and /dev/accel/accel0 is accessible."
            )

        # VAIP reads firmware and target from env vars, not from provider_options.
        os.environ["XLNX_VART_FIRMWARE"] = detector_config.xclbin
        os.environ["XLNX_TARGET_NAME"] = detector_config.target
        os.environ["NUM_OF_DPU_RUNNERS"] = str(detector_config.num_dpu_runners)

        path = detector_config.model.path
        provider_options = [{"config_file": detector_config.config_file}, {}]
        providers = ["VitisAIExecutionProvider", "CPUExecutionProvider"]

        logger.info(
            f"AMD NPU: loading {path} "
            f"[xclbin={detector_config.xclbin}, target={detector_config.target}]"
        )
        session = ort.InferenceSession(
            path,
            providers=providers,
            provider_options=provider_options,
        )
        chosen = session.get_providers()
        logger.info(f"AMD NPU: ORT providers actually engaged: {chosen}")
        if "VitisAIExecutionProvider" not in chosen:
            logger.warning(
                "AMD NPU: VitisAI did NOT activate; falling back entirely to CPU. "
                "Inspect VAIP log; verify xclbin/config_file paths and that the "
                "model is INT8-quantized with static shapes."
            )

        self.runner = ONNXModelRunner(
            session, model_type=detector_config.model.model_type
        )
        self.npu_model_type = detector_config.model.model_type

        if self.npu_model_type == ModelTypeEnum.yolox:
            self.calculate_grids_strides()

    def detect_raw(self, tensor_input: np.ndarray):
        if self.npu_model_type == ModelTypeEnum.dfine:
            tensor_output = self.runner.run(
                {
                    "images": tensor_input,
                    "orig_target_sizes": np.array(
                        [[self.height, self.width]], dtype=np.int64
                    ),
                }
            )
            return post_process_dfine(tensor_output, self.width, self.height)

        model_input_name = self.runner.get_input_names()[0]
        tensor_output = self.runner.run({model_input_name: tensor_input})

        if self.npu_model_type == ModelTypeEnum.rfdetr:
            return post_process_rfdetr(tensor_output)
        elif self.npu_model_type == ModelTypeEnum.yolonas:
            predictions = tensor_output[0]
            detections = np.zeros((20, 6), np.float32)
            for i, prediction in enumerate(predictions):
                if i == 20:
                    break
                (_, x_min, y_min, x_max, y_max, confidence, class_id) = prediction
                if class_id < 0:
                    break
                detections[i] = [
                    class_id,
                    confidence,
                    y_min / self.height,
                    x_min / self.width,
                    y_max / self.height,
                    x_max / self.width,
                ]
            return detections
        elif self.npu_model_type == ModelTypeEnum.yologeneric:
            return post_process_yolo(tensor_output, self.width, self.height)
        elif self.npu_model_type == ModelTypeEnum.yolox:
            return post_process_yolox(
                tensor_output[0],
                self.width,
                self.height,
                self.grids,
                self.expanded_strides,
            )
        else:
            raise Exception(
                f"{self.npu_model_type} is currently not supported for amd_npu. "
                "Supported: dfine, rfdetr, yolonas, yolo-generic, yolox."
            )
