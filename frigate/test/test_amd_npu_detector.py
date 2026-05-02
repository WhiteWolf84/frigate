import logging
import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

# Stub load_labels before ModelConfig.__init__ tries to read /labelmap.txt.
import frigate.detectors.detector_config as _det_cfg

_det_cfg.load_labels = lambda *a, **kw: {}

from frigate.detectors.detector_config import (  # noqa: E402
    InputDTypeEnum,
    InputTensorEnum,
    ModelConfig,
    ModelTypeEnum,
    PixelFormatEnum,
)


class _FakeRunner:
    """Stand-in for ONNXModelRunner so tests don't pull in frigate.embeddings."""

    def __init__(self, session, model_type=None):
        self.ort = session
        self.model_type = model_type

    def get_input_names(self):
        return [i.name for i in self.ort.get_inputs()]

    def run(self, feed):
        return self.ort.run(None, feed)


def _patch_runner():
    return patch("frigate.detectors.plugins.amd_npu.ONNXModelRunner", _FakeRunner)


def _make_config(
    model_type: ModelTypeEnum = ModelTypeEnum.yologeneric,
    model_path: str = "/tmp/frigate-amd-npu-test.onnx",
):
    from frigate.detectors.plugins.amd_npu import AmdNpuDetectorConfig

    return AmdNpuDetectorConfig(
        type="amd_npu",
        model=ModelConfig(
            path=model_path,
            labelmap_path=None,
            width=320,
            height=320,
            model_type=model_type,
            input_tensor=InputTensorEnum.nchw,
            input_pixel_format=PixelFormatEnum.rgb,
            input_dtype=InputDTypeEnum.float,
        ),
        xclbin="/tmp/test.xclbin",
        config_file="/tmp/vaip_config.json",
        target="AMD_AIE2_Nx4_Overlay",
    )


def _fake_ort_module(
    available_providers=("VitisAIExecutionProvider", "CPUExecutionProvider"),
    chosen_providers=("VitisAIExecutionProvider", "CPUExecutionProvider"),
    run_return=None,
    input_names=("images",),
):
    """Build a Mock onnxruntime module whose InferenceSession yields a controllable session."""
    ort = MagicMock(name="onnxruntime")
    ort.get_available_providers.return_value = list(available_providers)

    session = MagicMock(name="InferenceSession")
    session.get_providers.return_value = list(chosen_providers)
    session.get_inputs.return_value = [MagicMock(name=n) for n in input_names]
    for inp, n in zip(session.get_inputs.return_value, input_names):
        inp.name = n
    session.run.return_value = run_return or [np.zeros((1, 1), np.float32)]
    ort.InferenceSession.return_value = session
    return ort, session


class TestAmdNpuDetector(unittest.TestCase):
    def test_session_built_with_vitisai_then_cpu(self):
        ort, _ = _fake_ort_module()
        with patch.dict(sys.modules, {"onnxruntime": ort}), _patch_runner():
            from frigate.detectors.plugins.amd_npu import AmdNpuDetector

            cfg = _make_config()
            AmdNpuDetector(cfg)

        ort.InferenceSession.assert_called_once()
        kwargs = ort.InferenceSession.call_args.kwargs
        self.assertEqual(
            kwargs["providers"],
            ["VitisAIExecutionProvider", "CPUExecutionProvider"],
        )
        # VAIP reads xclbin/target from env vars; provider_options only needs config_file.
        opts = kwargs["provider_options"][0]
        self.assertEqual(opts["config_file"], cfg.config_file)
        self.assertNotIn("xclbin", opts)
        self.assertNotIn("cacheDir", opts)
        self.assertEqual(kwargs["provider_options"][1], {})
        # Verify env vars were set correctly.
        import os

        self.assertEqual(os.environ.get("XLNX_VART_FIRMWARE"), cfg.xclbin)
        self.assertEqual(os.environ.get("XLNX_TARGET_NAME"), cfg.target)
        self.assertEqual(os.environ.get("NUM_OF_DPU_RUNNERS"), str(cfg.num_dpu_runners))

    def test_logs_chosen_providers(self):
        ort, _ = _fake_ort_module()
        with patch.dict(sys.modules, {"onnxruntime": ort}), _patch_runner():
            from frigate.detectors.plugins.amd_npu import AmdNpuDetector

            with self.assertLogs("frigate.detectors.plugins.amd_npu", "INFO") as cm:
                AmdNpuDetector(_make_config())

        joined = "\n".join(cm.output)
        self.assertIn("providers actually engaged", joined)
        self.assertIn("VitisAIExecutionProvider", joined)

    def test_warns_when_vitisai_did_not_activate(self):
        ort, _ = _fake_ort_module(chosen_providers=("CPUExecutionProvider",))
        with patch.dict(sys.modules, {"onnxruntime": ort}), _patch_runner():
            from frigate.detectors.plugins.amd_npu import AmdNpuDetector

            with self.assertLogs("frigate.detectors.plugins.amd_npu", "WARNING") as cm:
                AmdNpuDetector(_make_config())

        self.assertTrue(
            any("did NOT activate" in line for line in cm.output),
            f"expected fallback WARNING; got {cm.output!r}",
        )

    def test_raises_when_vitisai_unavailable(self):
        ort, _ = _fake_ort_module(available_providers=("CPUExecutionProvider",))
        with patch.dict(sys.modules, {"onnxruntime": ort}):
            from frigate.detectors.plugins.amd_npu import AmdNpuDetector

            with self.assertRaises(RuntimeError) as ctx:
                AmdNpuDetector(_make_config())

        self.assertIn("VitisAIExecutionProvider not available", str(ctx.exception))

    def test_raises_when_onnxruntime_missing(self):
        # Simulate onnxruntime entirely unavailable by removing it from sys.modules
        # and forcing the lazy import in __init__ to ImportError.
        real_import = __import__

        def _blocked_import(name, *a, **kw):
            if name == "onnxruntime":
                raise ImportError("onnxruntime not installed")
            return real_import(name, *a, **kw)

        from frigate.detectors.plugins.amd_npu import AmdNpuDetector

        with patch("builtins.__import__", side_effect=_blocked_import):
            with self.assertRaises(RuntimeError) as ctx:
                AmdNpuDetector(_make_config())

        self.assertIn("onnxruntime-vitisai", str(ctx.exception))

    def test_detect_raw_shape_per_model_type(self):
        cases = [
            (
                ModelTypeEnum.yologeneric,
                "frigate.detectors.plugins.amd_npu.post_process_yolo",
            ),
            (
                ModelTypeEnum.yolox,
                "frigate.detectors.plugins.amd_npu.post_process_yolox",
            ),
            (
                ModelTypeEnum.rfdetr,
                "frigate.detectors.plugins.amd_npu.post_process_rfdetr",
            ),
            (
                ModelTypeEnum.dfine,
                "frigate.detectors.plugins.amd_npu.post_process_dfine",
            ),
        ]
        for model_type, post_path in cases:
            with self.subTest(model_type=model_type):
                ort, session = _fake_ort_module(
                    run_return=[np.zeros((1, 84, 100), np.float32)]
                )
                expected = np.zeros((20, 6), np.float32)
                with patch.dict(sys.modules, {"onnxruntime": ort}), _patch_runner():
                    from frigate.detectors.plugins.amd_npu import AmdNpuDetector

                    cfg = _make_config(model_type=model_type)
                    with patch(post_path, return_value=expected) as mock_post:
                        det = AmdNpuDetector(cfg)
                        out = det.detect_raw(np.zeros((1, 3, 320, 320), np.float32))

                self.assertEqual(out.shape, (20, 6))
                self.assertEqual(out.dtype, np.float32)
                mock_post.assert_called_once()

    def test_yolonas_results_sorted_by_confidence(self):
        # yolonas has its own inline branch in detect_raw; verify confidence ordering.
        # Each prediction tuple: (_, x_min, y_min, x_max, y_max, confidence, class_id)
        preds = np.array(
            [
                [0, 10, 20, 110, 120, 0.91, 2],
                [0, 30, 40, 130, 140, 0.55, 1],
                [0, 0, 0, 0, 0, 0.0, -1],  # sentinel: stops iteration
            ],
            dtype=np.float32,
        )
        ort, _ = _fake_ort_module(run_return=[preds])
        with patch.dict(sys.modules, {"onnxruntime": ort}), _patch_runner():
            from frigate.detectors.plugins.amd_npu import AmdNpuDetector

            det = AmdNpuDetector(_make_config(model_type=ModelTypeEnum.yolonas))
            out = det.detect_raw(np.zeros((1, 3, 320, 320), np.float32))

        self.assertEqual(out.shape, (20, 6))
        self.assertAlmostEqual(out[0][1], 0.91, places=4)
        self.assertAlmostEqual(out[1][1], 0.55, places=4)
        # Sentinel row stops processing -> remaining rows are zero-filled.
        self.assertEqual(out[2][1], 0.0)

    def test_dfine_dispatch_passes_orig_target_sizes(self):
        ort, session = _fake_ort_module()
        with patch.dict(sys.modules, {"onnxruntime": ort}), _patch_runner():
            from frigate.detectors.plugins.amd_npu import AmdNpuDetector

            cfg = _make_config(model_type=ModelTypeEnum.dfine)
            with patch(
                "frigate.detectors.plugins.amd_npu.post_process_dfine",
                return_value=np.zeros((20, 6), np.float32),
            ):
                det = AmdNpuDetector(cfg)
                det.detect_raw(np.zeros((1, 3, 320, 320), np.float32))

        # Find the run() call that came from detect_raw (the only one for dfine).
        run_calls = session.run.call_args_list
        self.assertGreaterEqual(len(run_calls), 1)
        # ONNXModelRunner.run forwards (None, input_dict) to session.run.
        feed = run_calls[0].args[1]
        self.assertIn("images", feed)
        self.assertIn("orig_target_sizes", feed)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
