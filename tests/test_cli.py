import pytest

from src import cli


def test_cli_run_dispatches_to_main(monkeypatch):
    calls = {"count": 0}

    def _fake_main():
        calls["count"] += 1

    monkeypatch.setattr("src.cli.run_app", _fake_main)

    exit_code = cli.main(["run"])

    assert exit_code == 0
    assert calls["count"] == 1


def test_cli_collect_data_returns_placeholder_error_text():
    exit_code = cli.main(["collect-data"])
    assert exit_code == 2


def test_cli_train_models_returns_placeholder_error_text():
    exit_code = cli.main(["train-models"])
    assert exit_code == 2


def test_cli_collect_data_init_scaffold_creates_layout(tmp_path):
    exit_code = cli.main(["collect-data", "--init-scaffold", "--workspace", str(tmp_path)])

    assert exit_code == 0
    assert (tmp_path / "data" / "raw_screenshots").is_dir()
    assert (tmp_path / "data" / "sessions" / ".gitkeep").is_file()
    assert (tmp_path / "data" / "labels" / "yolo").is_dir()
    assert (tmp_path / "data" / "labels" / "classifier").is_dir()
    assert (tmp_path / "data" / "labels" / "README.md").is_file()
    assert (tmp_path / "data" / "README.md").is_file()
    assert "intentionally left blank" in (tmp_path / "data" / "README.md").read_text(encoding="utf-8")


def test_cli_train_models_init_scaffold_creates_layout(tmp_path):
    exit_code = cli.main(["train-models", "--init-scaffold", "--workspace", str(tmp_path)])

    assert exit_code == 0
    assert (tmp_path / "configs" / "detector.example.yaml").is_file()
    assert (tmp_path / "configs" / "classifier.example.yaml").is_file()
    assert (tmp_path / "scripts" / "train_detector_placeholder.py").is_file()
    assert (tmp_path / "scripts" / "train_classifier_placeholder.py").is_file()


def test_cli_collect_data_check_reports_missing_then_ready(tmp_path):
    missing_exit = cli.main(["collect-data", "--check", "--workspace", str(tmp_path)])
    assert missing_exit == 3

    init_exit = cli.main(["collect-data", "--init-scaffold", "--workspace", str(tmp_path)])
    assert init_exit == 0

    ready_exit = cli.main(["collect-data", "--check", "--workspace", str(tmp_path)])
    assert ready_exit == 0


def test_cli_train_models_check_reports_missing_then_ready(tmp_path):
    missing_exit = cli.main(["train-models", "--check", "--workspace", str(tmp_path)])
    assert missing_exit == 3

    init_exit = cli.main(["train-models", "--init-scaffold", "--workspace", str(tmp_path)])
    assert init_exit == 0

    ready_exit = cli.main(["train-models", "--check", "--workspace", str(tmp_path)])
    assert ready_exit == 0


def test_cli_collect_data_dry_run_does_not_create_files(tmp_path):
    exit_code = cli.main(["collect-data", "--init-scaffold", "--dry-run", "--workspace", str(tmp_path)])

    assert exit_code == 0
    assert not (tmp_path / "data").exists()


def test_cli_train_models_dry_run_does_not_create_files(tmp_path):
    exit_code = cli.main(["train-models", "--init-scaffold", "--dry-run", "--workspace", str(tmp_path)])

    assert exit_code == 0
    assert not (tmp_path / "configs").exists()
    assert not (tmp_path / "scripts").exists()


def test_cli_collect_data_rejects_conflicting_flags(tmp_path, capsys):
    exit_code = cli.main(["collect-data", "--init-scaffold", "--check", "--workspace", str(tmp_path)])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "cannot be used together" in captured.out


def test_cli_train_models_rejects_conflicting_flags(tmp_path, capsys):
    exit_code = cli.main(["train-models", "--init-scaffold", "--check", "--workspace", str(tmp_path)])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "cannot be used together" in captured.out


def test_cli_help_mentions_exit_codes(capsys):
    exit_code = cli.main([])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "Exit codes" in captured.out


def test_cli_preflight_success(monkeypatch, capsys):
    monkeypatch.setattr("src.cli.run_preflight_checks", lambda: ([], ["w1"]))

    exit_code = cli.main(["preflight"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Preflight completed with warnings" in captured.out


def test_cli_preflight_failure(monkeypatch, capsys):
    monkeypatch.setattr("src.cli.run_preflight_checks", lambda: (["e1"], ["w1"]))

    exit_code = cli.main(["preflight"])
    captured = capsys.readouterr()

    assert exit_code == 3
    assert "Preflight failed" in captured.out
