# Code Complexity Audit

## Executive Summary
This report analyzes the code complexity of the MangaAutoColor Pro codebase. The analysis focuses on cyclomatic complexity, cognitive load, line counts, coupling, and cohesion.
**Date:** 2026-02-07

## 1. Top Files by Lines of Code (LOC)
| File | Lines | Description |
|------|-------|-------------|
| `core/generation/pipeline.py` | ~1550 | **Critical Hotspot**. Contains `TileAwareGenerator` with mixing logic for tiles, prompts, and model inference. |
| `core/database/chapter_db.py` | ~709 | Large class. Mixes persistence (Parquet), search (FAISS), and business logic. |
| `core/detection/yolo_detector.py` | ~550 | Moderate. Logic for detection and heuristic grouping is complex. |
| `core/pipeline.py` | ~490 | Orchestrator. Natural high coupling but acceptable LOC. |
| `core/pass1_analyzer.py` | ~424 | Analysis logic. Reasonable size but increasing complexity. |

---

## 2. Complexity Analysis & Findings

### A. Cyclomatic & Cognitive Complexity

#### 1. `TileAwareGenerator` (`core/generation/pipeline.py`) - **Score: 9/10**
- **Problem**: This class is effectively a "God Object" for the generation phase.
- **Complex Methods**:
    - `_generate_tile`: Handles tiling, Canny extraction, prompt building, inference, and blending preparation in one place.
    - `_build_prompt`: Contains nested logic for style priority (`options` vs `style_presets` vs `color_references`) and text formatting.
    - `_blend_tiles`: Nested loops for pixel-level blending.
- **Cognitive Load**: High. Requires understanding IP-Adapter, ControlNet, VAE, and prompt engineering simultaneously.

#### 2. `ChapterDatabase` (`core/database/chapter_db.py`) - **Score: 7/10**
- **Problem**: Low Cohesion. It manages file I/O (Parquet), vector search (FAISS), and data consolidation logic.
- **Complex Methods**:
    - `consolidate_characters`: Implements clustering logic inside the data access layer.
    - `save_character_embedding`: Handles file saving, index updating, and dataframe appending.

#### 3. `YOLODetector` (`core/detection/yolo_detector.py`) - **Score: 6/10**
- **Problem**: Heuristic complexity.
- **Complex Methods**:
    - `group_body_face_pairs`: Uses nested loops and spatial heuristics to match faces to bodies. Hard to debug.

### B. Coupling Metrics

- **`MangaColorizationPipeline`**: High **Efferent Coupling** (depends on many modules). This is expected for an orchestrator but makes testing difficult without extensive mocking.
- **`Pass1Analyzer`**: High Coupling. Directly instantiates `YOLODetector`, `HybridIdentityEncoder`, etc., instead of receiving them via dependency injection.
- **`TileAwareGenerator`**: Tightly coupled to `diffusers` implementation details and `config.settings`.

### C. Cohesion Analysis

- **`core/utils/image_ops.py`**: **High Cohesion**. Focused solely on image processing geometry and math. (Good job on refactoring!)
- **`core/chapter_db.py`**: **Low Cohesion**. Mixes *Storage* (Parquet/Disk) with *Search* (FAISS) and *Logic* (Merges).
- **`core/generation/pipeline.py`**: **Low Cohesion**. Mixes *Prompt Engineering* (Strings) with *Image Generation* (Tensors).

---

## 3. Refactoring Recommendations

### Priority 1: Extract Prompt Logic (Importance: 9/10)
`TileAwareGenerator` has ~200 lines dedicated to building prompts.
**Remediation**: Create `core/generation/prompt_builder.py`.
```python
class MangaPromptBuilder:
    def build(self, page_data, options, active_palettes) -> str:
        # Logic from _build_prompt
```

### Priority 2: Split ChapterDatabase (Importance: 7/10)
Separate vector search from metadata storage.
**Remediation**: Extract `VectorIndex` or `CharacterSearchEngine`.
```python
class CharacterVectorIndex:
    def add(self, id, embedding): ...
    def search(self, embedding): ...
    def save(self): ...
```
`ChapterDatabase` should wrap this, not implement it.

### Priority 3: Decouple Generation from Tiling (Importance: 6/10)
`TileAwareGenerator` handles both the "What to generate" (Tile logic) and "How to generate" (SDXL logic).
**Remediation**: Extract `TileScheduler` to calculate grids and overlaps.

### Priority 4: Dependency Injection for Analyzer (Importance: 5/10)
Pass detectors/encoders into `Pass1Analyzer` instead of creating them inside property getters. This improves testability.

## 4. Conclusion
The codebase is functional but `TileAwareGenerator` is approaching unmaintainable categorization. Extracting `PromptBuilder` is the highest ROI refactor. `ChapterDatabase` is robust but monolithic; splitting it would prepare the system for scaling (e.g., using a real DB instead of Parquet later).
