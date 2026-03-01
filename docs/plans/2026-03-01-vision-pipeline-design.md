# Vision-based Mahjong Assistant Design Document
**Date:** 2026-03-01
**Topic:** Core Vision & Inference Pipeline

## 1. Overview
The goal is to build a real-time vision-based assistant for Mahjong Soul (Japanese Riichi Mahjong) that analyzes the screen and provides decision support using the Mortal AI engine via the `mjai` protocol.

## 2. Architecture: YOLO + CNN Hybrid
To solve the problem of varied meld (副露) sizes and overlapping tiles without requiring massive full-screen bounding box annotations:

*   **YOLO Detector (Tile Locator):** A `YOLOv8n` model trained on just **ONE class** (`0: tile`). Its only job is to return the bounding boxes $[x1, y1, x2, y2]$ of every mahjong tile it sees in the defined Screen Region of Interest (ROI).
*   **CNN Classifier (Tile Recognizer):** A lightweight image classifier (e.g., `MobileNetV3-small`) trained on 34 classes (the 34 mahjong tile types). It takes the cropped tile image patches from the YOLO bounding boxes and predicts the tile's rank/suit.

## 3. State Management: Stateless Snapshotting
To handle visual occlusions (e.g., flashy animations like Ron, Riichi, or Kokushi Musou cut-ins) and prevent the internal state machine from desyncing with the real game:

*   **Stateless Inference:** The system will evaluate the screen as a completely new puzzle every frame (or every N frames). It will not rely solely on "diffing" or event tracking.
*   **Confidence Gating:** If the screen is heavily occluded by an animation, the YOLO detector will fail to find the expected number of tiles, or the CNN will return low confidence scores.
*   **Silent Drop:** When confidence is below the threshold or the parsed JSON structure is invalid for a mahjong phase, the system will drop the frame and wait. Once the UI settles, the next valid snapshot will instantly rebuild the complete 100% accurate game state.

## 4. Implementation Phasing
*   **Phase 1: Hand Tiles Only.** Restrict the ROI to the bottom of the screen. Train the minimalist YOLO tile detector and the 34-class CNN to perfectly read the player's 13/14 hand tiles.
*   **Phase 2: Discard Piles.** Expand ROI to the center grid. Parse the discards by sorting the identified tiles functionally by coordinates.
*   **Phase 3: Melds (副露).** Expand ROI to the edges. Use coordination clustering (e.g., finding horizontal/vertical groups and face-down tiles) to infer Chii/Pon/Kan operations.

## 5. User Interface
A separate, floating desktop window (`PyQt`/`PySide` based) that displays:
*   Pre-game configuration (e.g., 3-player vs 4-player toggle).
*   Real-time output of the Mortal AI recommendations mapped continuously from the parsed fast snapshots.
