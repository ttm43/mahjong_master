from pathlib import Path


TILE_CLASSES = [
    "1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m",
    "1p", "2p", "3p", "4p", "5p", "6p", "7p", "8p", "9p",
    "1s", "2s", "3s", "4s", "5s", "6s", "7s", "8s", "9s",
    "E", "S", "W", "N", "P", "F", "C",
]


def run_data_collection():
    raise NotImplementedError(
        "Data collection is intentionally left blank. Add your screenshot capture/labeling pipeline later."
    )


def run_model_training():
    raise NotImplementedError(
        "Model training is intentionally left blank. Add detector/classifier training workflows later."
    )


def _data_entries(root):
    return [
        ("dir", root / "data" / "raw_screenshots", None),
        ("dir", root / "data" / "sessions", None),
        ("dir", root / "data" / "labels" / "yolo", None),
        ("dir", root / "data" / "labels" / "classifier", None),
        (
            "file",
            root / "data" / "README.md",
            "# Data Workspace\n\n"
            "This section is intentionally left blank.\n\n"
            "- Put raw screenshots in `data/raw_screenshots/`\n"
            "- Put YOLO labels in `data/labels/yolo/`\n"
            "- Put classifier labels in `data/labels/classifier/`\n",
        ),
        (
            "file",
            root / "data" / "labels" / "README.md",
            "# Labeling Notes\n\n"
            "This section is intentionally left blank.\n"
            "Add your annotation rules and QA checklist here.\n",
        ),
        ("file", root / "data" / "sessions" / ".gitkeep", ""),
    ]


def _training_entries(root):
    return [
        ("dir", root / "configs", None),
        ("dir", root / "scripts", None),
        (
            "file",
            root / "configs" / "detector.example.yaml",
            "model: yolov8n\n"
            "data: data/detector_dataset.yaml\n"
            "epochs: 100\n"
            "imgsz: 640\n",
        ),
        (
            "file",
            root / "configs" / "classifier.example.yaml",
            "model: mobilenet_v3_small\n"
            "data_dir: data/classifier_dataset\n"
            "epochs: 50\n"
            "input_size: 96\n",
        ),
        ("file", root / "configs" / "tile_classes.txt", "\n".join(TILE_CLASSES) + "\n"),
        (
            "file",
            root / "scripts" / "train_detector_placeholder.py",
            "raise NotImplementedError(\"Detector training is intentionally left blank. Add real training code later.\")\n",
        ),
        (
            "file",
            root / "scripts" / "train_classifier_placeholder.py",
            "raise NotImplementedError(\"Classifier training is intentionally left blank. Add real training code later.\")\n",
        ),
    ]


def _apply_entries(entries, dry_run):
    changed = []
    for kind, path, content in entries:
        if kind == "dir":
            if path.exists():
                continue
            changed.append(path)
            if not dry_run:
                path.mkdir(parents=True, exist_ok=True)
            continue

        if path.exists():
            continue
        changed.append(path)
        if not dry_run:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")

    return [str(p) for p in changed]


def _missing_entries(entries):
    return [str(path) for _, path, _ in entries if not path.exists()]


def init_data_scaffold(workspace, dry_run=False):
    return _apply_entries(_data_entries(Path(workspace)), dry_run=dry_run)


def init_training_scaffold(workspace, dry_run=False):
    return _apply_entries(_training_entries(Path(workspace)), dry_run=dry_run)


def check_data_scaffold(workspace):
    return _missing_entries(_data_entries(Path(workspace)))


def check_training_scaffold(workspace):
    root = Path(workspace)
    missing = _missing_entries(_training_entries(root))

    classes_file = root / "configs" / "tile_classes.txt"
    if classes_file.exists():
        current = [line.strip() for line in classes_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        if current != TILE_CLASSES:
            missing.append(str(classes_file))

    return missing
