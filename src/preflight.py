import importlib
import importlib.util
from pathlib import Path

from src.config import load_app_config, resolve_path
from src.core.capture import ScreenCapturer


REQUIRED_DEPENDENCIES = [
    "numpy",
    "cv2",
    "mss",
    "PyQt5",
    "torch",
    "torchvision",
    "ultralytics",
]


def _default_dependency_probe(module_name):
    if importlib.util.find_spec(module_name) is None:
        return False, None

    try:
        module = importlib.import_module(module_name)
        version = getattr(module, "__version__", "unknown")
    except Exception:
        version = "unknown"

    return True, str(version)


def _default_capture_validator(cfg):
    capturer = ScreenCapturer(monitor_index=cfg.monitor_index, hand_roi_height=cfg.hand_roi_height)
    if hasattr(capturer.sct, "close"):
        capturer.sct.close()


def run_preflight_report(config=None, dependency_probe=None, capture_validator=None):
    cfg = load_app_config() if config is None else config
    dependency_probe = _default_dependency_probe if dependency_probe is None else dependency_probe
    capture_validator = _default_capture_validator if capture_validator is None else capture_validator

    issues = []
    warnings = []

    detector_model_path = resolve_path(cfg.detector_model_path)
    classifier_model_path = resolve_path(cfg.classifier_model_path)

    if not Path(detector_model_path).exists():
        warnings.append(f"Detector model not found: {detector_model_path}")
    if not Path(classifier_model_path).exists():
        warnings.append(f"Classifier model not found: {classifier_model_path}")

    if cfg.fps <= 0:
        issues.append("MAHJONG_PIPELINE_FPS must be > 0")

    missing_dependencies = []
    dependency_versions = {}
    for module_name in REQUIRED_DEPENDENCIES:
        available, version = dependency_probe(module_name)
        if not available:
            missing_dependencies.append(module_name)
            continue
        dependency_versions[module_name] = version or "unknown"

    if missing_dependencies:
        issues.append(f"Missing required dependencies: {', '.join(missing_dependencies)}")

    try:
        capture_validator(cfg)
    except Exception as exc:
        issues.append(f"Monitor validation failed: {exc}")

    return {
        "issues": issues,
        "warnings": warnings,
        "dependencies": {
            "required": list(REQUIRED_DEPENDENCIES),
            "missing": missing_dependencies,
            "versions": dependency_versions,
        },
        "config": {
            "fps": cfg.fps,
            "monitor_index": cfg.monitor_index,
            "hand_roi_height": cfg.hand_roi_height,
            "detector_model_path": detector_model_path,
            "classifier_model_path": classifier_model_path,
        },
    }


def run_preflight_checks():
    report = run_preflight_report()
    return report["issues"], report["warnings"]
