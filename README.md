# Mahjong Master

Real-time Mahjong vision assistant pipeline scaffold.

## Current Scope

- Screen capture + ROI cropping
- Tile detection/classification runtime wiring
- Overlay rendering
- CLI scaffolding for data/training workflows
- Preflight checks

Data collection and model training logic are intentionally left as placeholders.

## Requirements

- Python 3.10+
- Windows desktop environment (for overlay/capture)

Install dependencies:

```bash
pip install -r requirements.txt
```

Install package (editable) with dev tooling:

```bash
pip install -e .[dev]
```

## Run

```bash
python -m src.cli run
```

After package install, you can also run:

```bash
mahjong-master run
```

Run with explicit config file:

```bash
python -m src.cli run --config configs/app.yaml
```

## Preflight

Validate monitor/config/model prerequisites before runtime:

```bash
python -m src.cli preflight
```

Machine-readable report:

```bash
python -m src.cli preflight --json
```

Use explicit config file for preflight:

```bash
python -m src.cli preflight --config configs/app.yaml
```

Strict mode (warnings also fail with exit code `3`):

```bash
python -m src.cli preflight --strict
```

The JSON report includes `issues`, `warnings`, dependency versions/missing modules, and effective runtime config.

## Scaffold Commands

Initialize data workspace layout:

```bash
python -m src.cli collect-data --init-scaffold --workspace .
```

Initialize training workspace layout:

```bash
python -m src.cli train-models --init-scaffold --workspace .
```

Dry-run (no files created):

```bash
python -m src.cli collect-data --init-scaffold --dry-run --workspace .
python -m src.cli train-models --init-scaffold --dry-run --workspace .
```

Check scaffold completeness:

```bash
python -m src.cli collect-data --check --workspace .
python -m src.cli train-models --check --workspace .
```

## Environment Variables

- `MAHJONG_DETECTOR_MODEL` (default: `models/tile_detector.pt`)
- `MAHJONG_CLASSIFIER_MODEL` (default: `models/tile_classifier.pt`)
- `MAHJONG_PIPELINE_FPS` (default: `10`)
- `MAHJONG_MONITOR_INDEX` (default: `1`)
- `MAHJONG_HAND_ROI_HEIGHT` (default: `200`)
- `MAHJONG_LOG_LEVEL` (default: `INFO`)
- `MAHJONG_LOG_FORMAT` (`text` or `json`, default: `text`)
- `MAHJONG_LOG_FILE` (optional, example: `logs/app.log`)
- `MAHJONG_LOG_ROTATE_BYTES` (default: `1048576`)
- `MAHJONG_LOG_BACKUP_COUNT` (default: `3`)

## Optional Config File

You can place runtime config in `configs/app.yaml` (simple `key: value` format):

```yaml
fps: 10
monitor_index: 1
hand_roi_height: 200
detector_model_path: models/tile_detector.pt
classifier_model_path: models/tile_classifier.pt
log_level: INFO
log_format: text
log_file: logs/app.log
log_rotate_bytes: 1048576
log_backup_count: 3
```

Environment variables override config file values.

## CLI Exit Codes

- `0`: success
- `1`: usage/help shown
- `2`: invalid flag combination or placeholder command
- `3`: check/preflight failed

## Notes

- Overlay displays startup/runtime status text, including missing model warnings.
- If model files are missing, runtime continues in placeholder mode.

## Quality Checks

Local checks:

```bash
ruff check src tests
mypy src
pytest -q
```

CI runs the same checks on `push`/`pull_request` via [.github/workflows/ci.yml](c:/Users/46025/work/mahjong_master/.github/workflows/ci.yml).
