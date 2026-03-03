from pathlib import Path

from src.config import load_app_config, resolve_path
from src.core.capture import ScreenCapturer


def run_preflight_checks():
    issues = []
    warnings = []

    cfg = load_app_config()
    detector_model_path = resolve_path(cfg.detector_model_path)
    classifier_model_path = resolve_path(cfg.classifier_model_path)

    if not Path(detector_model_path).exists():
        warnings.append(f"Detector model not found: {detector_model_path}")
    if not Path(classifier_model_path).exists():
        warnings.append(f"Classifier model not found: {classifier_model_path}")

    if cfg.fps <= 0:
        issues.append("MAHJONG_PIPELINE_FPS must be > 0")

    try:
        capturer = ScreenCapturer(monitor_index=cfg.monitor_index, hand_roi_height=cfg.hand_roi_height)
        if hasattr(capturer.sct, "close"):
            capturer.sct.close()
    except Exception as exc:
        issues.append(f"Monitor validation failed: {exc}")

    return issues, warnings
