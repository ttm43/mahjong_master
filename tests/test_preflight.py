from src.config import AppConfig
from src.preflight import run_preflight_report


def test_preflight_report_marks_missing_dependencies_as_issue(tmp_path):
    cfg = AppConfig(
        detector_model_path=str(tmp_path / "detector.pt"),
        classifier_model_path=str(tmp_path / "classifier.pt"),
        fps=10,
        monitor_index=1,
        hand_roi_height=200,
    )

    report = run_preflight_report(
        config=cfg,
        dependency_probe=lambda name: (False, None),
        capture_validator=lambda _: None,
    )

    assert report["issues"]
    assert "Missing required dependencies" in report["issues"][0]
    assert report["dependencies"]["missing"]


def test_preflight_report_returns_warning_for_missing_models_only(tmp_path):
    cfg = AppConfig(
        detector_model_path=str(tmp_path / "detector.pt"),
        classifier_model_path=str(tmp_path / "classifier.pt"),
        fps=10,
        monitor_index=1,
        hand_roi_height=200,
    )

    report = run_preflight_report(
        config=cfg,
        dependency_probe=lambda name: (True, "x.y"),
        capture_validator=lambda _: None,
    )

    assert report["issues"] == []
    assert len(report["warnings"]) == 2
    assert report["dependencies"]["missing"] == []


def test_preflight_report_uses_config_path(tmp_path):
    cfg_file = tmp_path / "app.yaml"
    cfg_file.write_text("fps: 0\n", encoding="utf-8")

    report = run_preflight_report(
        config_path=cfg_file,
        dependency_probe=lambda name: (True, "x.y"),
        capture_validator=lambda _: None,
    )

    assert "MAHJONG_PIPELINE_FPS must be > 0" in report["issues"]
