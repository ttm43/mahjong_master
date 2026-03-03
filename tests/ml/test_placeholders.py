import pytest

from src.ml.placeholders import (
    TILE_CLASSES,
    check_training_scaffold,
    init_training_scaffold,
    run_data_collection,
    run_model_training,
)


def test_data_collection_placeholder_raises_clear_error():
    with pytest.raises(NotImplementedError, match="Data collection is intentionally left blank"):
        run_data_collection()


def test_model_training_placeholder_raises_clear_error():
    with pytest.raises(NotImplementedError, match="Model training is intentionally left blank"):
        run_model_training()


def test_init_training_scaffold_creates_tile_classes_catalog(tmp_path):
    init_training_scaffold(tmp_path)

    classes_file = tmp_path / "configs" / "tile_classes.txt"
    assert classes_file.is_file()

    labels = [line.strip() for line in classes_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(labels) == 34
    assert labels[0] == "1m"
    assert labels[-1] == "C"


def test_init_training_scaffold_does_not_overwrite_existing_config(tmp_path):
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    detector_cfg = cfg_dir / "detector.example.yaml"
    detector_cfg.write_text("custom: true\n", encoding="utf-8")

    init_training_scaffold(tmp_path)

    assert detector_cfg.read_text(encoding="utf-8") == "custom: true\n"


def test_check_training_scaffold_detects_tile_class_catalog_drift(tmp_path):
    init_training_scaffold(tmp_path)
    classes_file = tmp_path / "configs" / "tile_classes.txt"
    classes_file.write_text("broken\n", encoding="utf-8")

    missing = check_training_scaffold(tmp_path)

    assert str(classes_file) in missing


def test_tile_classes_match_classifier_contract():
    assert len(TILE_CLASSES) == 34
    assert TILE_CLASSES[0] == "1m"
    assert TILE_CLASSES[-1] == "C"
