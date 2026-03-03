from pathlib import Path

from src.config import load_app_config


def test_load_app_config_defaults_from_missing_file(tmp_path):
    cfg = load_app_config(config_path=tmp_path / "missing.yaml", env={})

    assert cfg.fps == 10
    assert cfg.monitor_index == 1
    assert cfg.hand_roi_height == 200
    assert cfg.detector_model_path == "models/tile_detector.pt"
    assert cfg.classifier_model_path == "models/tile_classifier.pt"
    assert cfg.log_level == "INFO"
    assert cfg.log_format == "text"
    assert cfg.log_file is None
    assert cfg.log_rotate_bytes == 1048576
    assert cfg.log_backup_count == 3


def test_load_app_config_file_values_and_env_override(tmp_path):
    cfg_file = tmp_path / "app.yaml"
    cfg_file.write_text(
        "fps: 12\n"
        "monitor_index: 2\n"
        "hand_roi_height: 180\n"
        "detector_model_path: custom/detector.pt\n"
        "classifier_model_path: custom/classifier.pt\n"
        "log_level: DEBUG\n"
        "log_format: json\n"
        "log_rotate_bytes: 4096\n"
        "log_backup_count: 5\n",
        encoding="utf-8",
    )

    env = {
        "MAHJONG_PIPELINE_FPS": "20",
        "MAHJONG_LOG_FILE": str(tmp_path / "logs" / "app.log"),
        "MAHJONG_LOG_BACKUP_COUNT": "7",
    }

    cfg = load_app_config(config_path=cfg_file, env=env)

    assert cfg.fps == 20
    assert cfg.monitor_index == 2
    assert cfg.hand_roi_height == 180
    assert cfg.detector_model_path == "custom/detector.pt"
    assert cfg.classifier_model_path == "custom/classifier.pt"
    assert cfg.log_level == "DEBUG"
    assert cfg.log_format == "json"
    assert cfg.log_file == str(tmp_path / "logs" / "app.log")
    assert cfg.log_rotate_bytes == 4096
    assert cfg.log_backup_count == 7
