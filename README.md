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

## Run

```bash
python -m src.cli run
```

## Preflight

Validate monitor/config/model prerequisites before runtime:

```bash
python -m src.cli preflight
```

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

## CLI Exit Codes

- `0`: success
- `1`: usage/help shown
- `2`: invalid flag combination or placeholder command
- `3`: check/preflight failed

## Notes

- Overlay displays startup/runtime status text, including missing model warnings.
- If model files are missing, runtime continues in placeholder mode.
