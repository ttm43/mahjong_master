import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class AppConfig:
    fps: int = 10
    monitor_index: int = 1
    hand_roi_height: int = 200
    detector_model_path: str = "models/tile_detector.pt"
    classifier_model_path: str = "models/tile_classifier.pt"
    log_level: str = "INFO"
    log_format: str = "text"
    log_file: str | None = None
    log_rotate_bytes: int = 1048576
    log_backup_count: int = 3


def _parse_scalar(value):
    v = value.strip()
    if v.startswith(("'", '"')) and v.endswith(("'", '"')) and len(v) >= 2:
        return v[1:-1]

    if v.lower() in {"true", "false"}:
        return v.lower() == "true"

    if v.lstrip("-").isdigit():
        return int(v)

    if v.lower() in {"none", "null"}:
        return None

    return v


def _load_simple_yaml(path):
    if not path.exists():
        return {}

    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if ":" not in s:
            continue
        key, raw_value = s.split(":", 1)
        out[key.strip()] = _parse_scalar(raw_value)
    return out


def load_app_config(config_path=None, env=None):
    env = os.environ if env is None else env
    config_path = Path("configs/app.yaml") if config_path is None else Path(config_path)

    cfg = AppConfig()
    file_cfg = _load_simple_yaml(config_path)

    # file values
    if "fps" in file_cfg:
        cfg.fps = int(file_cfg["fps"])
    if "monitor_index" in file_cfg:
        cfg.monitor_index = int(file_cfg["monitor_index"])
    if "hand_roi_height" in file_cfg:
        cfg.hand_roi_height = int(file_cfg["hand_roi_height"])
    if "detector_model_path" in file_cfg:
        cfg.detector_model_path = str(file_cfg["detector_model_path"])
    if "detector_model" in file_cfg:
        cfg.detector_model_path = str(file_cfg["detector_model"])
    if "classifier_model_path" in file_cfg:
        cfg.classifier_model_path = str(file_cfg["classifier_model_path"])
    if "classifier_model" in file_cfg:
        cfg.classifier_model_path = str(file_cfg["classifier_model"])
    if "log_level" in file_cfg:
        cfg.log_level = str(file_cfg["log_level"]).upper()
    if "log_format" in file_cfg:
        cfg.log_format = str(file_cfg["log_format"]).lower()
    if "log_file" in file_cfg:
        cfg.log_file = str(file_cfg["log_file"]) if file_cfg["log_file"] is not None else None
    if "log_rotate_bytes" in file_cfg:
        cfg.log_rotate_bytes = int(file_cfg["log_rotate_bytes"])
    if "log_backup_count" in file_cfg:
        cfg.log_backup_count = int(file_cfg["log_backup_count"])

    # env overrides
    if "MAHJONG_PIPELINE_FPS" in env:
        cfg.fps = int(env["MAHJONG_PIPELINE_FPS"])
    if "MAHJONG_MONITOR_INDEX" in env:
        cfg.monitor_index = int(env["MAHJONG_MONITOR_INDEX"])
    if "MAHJONG_HAND_ROI_HEIGHT" in env:
        cfg.hand_roi_height = int(env["MAHJONG_HAND_ROI_HEIGHT"])
    if "MAHJONG_DETECTOR_MODEL" in env:
        cfg.detector_model_path = str(env["MAHJONG_DETECTOR_MODEL"])
    if "MAHJONG_CLASSIFIER_MODEL" in env:
        cfg.classifier_model_path = str(env["MAHJONG_CLASSIFIER_MODEL"])
    if "MAHJONG_LOG_LEVEL" in env:
        cfg.log_level = str(env["MAHJONG_LOG_LEVEL"]).upper()
    if "MAHJONG_LOG_FORMAT" in env:
        cfg.log_format = str(env["MAHJONG_LOG_FORMAT"]).lower()
    if "MAHJONG_LOG_FILE" in env:
        cfg.log_file = str(env["MAHJONG_LOG_FILE"])
    if "MAHJONG_LOG_ROTATE_BYTES" in env:
        cfg.log_rotate_bytes = int(env["MAHJONG_LOG_ROTATE_BYTES"])
    if "MAHJONG_LOG_BACKUP_COUNT" in env:
        cfg.log_backup_count = int(env["MAHJONG_LOG_BACKUP_COUNT"])

    return cfg


def resolve_path(path_str):
    return str(Path(path_str).expanduser().resolve())
