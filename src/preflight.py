import os
from pathlib import Path

from src.core.capture import ScreenCapturer


def _resolve_model_path(env_var, default_rel_path):
    configured = os.getenv(env_var)
    if configured:
        return str(Path(configured).expanduser().resolve())
    return str((Path.cwd() / default_rel_path).resolve())


def run_preflight_checks():
    issues = []
    warnings = []

    detector_model_path = _resolve_model_path("MAHJONG_DETECTOR_MODEL", "models/tile_detector.pt")
    classifier_model_path = _resolve_model_path("MAHJONG_CLASSIFIER_MODEL", "models/tile_classifier.pt")

    if not Path(detector_model_path).exists():
        warnings.append(f"Detector model not found: {detector_model_path}")
    if not Path(classifier_model_path).exists():
        warnings.append(f"Classifier model not found: {classifier_model_path}")

    try:
        fps = int(os.getenv("MAHJONG_PIPELINE_FPS", "10"))
        if fps <= 0:
            issues.append("MAHJONG_PIPELINE_FPS must be > 0")
    except ValueError:
        issues.append("MAHJONG_PIPELINE_FPS must be an integer")

    try:
        monitor_index = int(os.getenv("MAHJONG_MONITOR_INDEX", "1"))
    except ValueError:
        issues.append("MAHJONG_MONITOR_INDEX must be an integer")
        monitor_index = 1

    try:
        capturer = ScreenCapturer(monitor_index=monitor_index)
        if hasattr(capturer.sct, "close"):
            capturer.sct.close()
    except Exception as exc:
        issues.append(f"Monitor validation failed: {exc}")

    return issues, warnings
