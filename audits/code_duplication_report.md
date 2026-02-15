# Code Duplication & Integrity Audit Report

**Date:** 2026-02-07
**Status:** Critical Issues Found

## Executive Summary
A deep audit of the codebase revealed a failed refactoring/migration effort. While "duplicate" files were identified, the logic within them was **not** fully migrated to the new architecture. This has left the active pipeline (`core/pipeline.py`) calling missing methods (`consolidate_characters`, `detect_narrative_arcs`) and lacking key features (Color References).

## 🚨 Critical Findings (Broken Logic)

### 1. Missing Method: `detect_narrative_arcs`
**Severity: 10/10 (Crash)**
- **Issue:** `core/pipeline.py` calls `db.detect_narrative_arcs()` (line 228).
- **Finding:** This method **does not exist** in `core/database/chapter_db.py` OR in any analyzed file (`pass1_analyzer.py.bak`).
- **Consequence:** The pipeline will crash unconditionally at the end of Pass 1.

### 2. Missing Method: `consolidate_characters`
**Severity: 10/10 (Crash)**
- **Issue:** `core/pipeline.py` calls `db.consolidate_characters()` (line 227).
- **Finding:** `core/database/chapter_db.py` **does not implement** this method.
- **Source:** The logic exists in `core/chapter_processing/pass1_analyzer.py.bak` (lines 560-630) but was not moved to `ChapterDatabase`.
- **Consequence:** The pipeline will crash with `AttributeError`.

### 3. Missing Feature: Color Reference Processing
**Severity: 9/10 (Feature Loss)**
- **Issue:** The `process_color_references` feature (extracting palettes from colored images) is completely missing from the active `MangaColorizationPipeline`.
- **Source:** The logic exists in `core/chapter_processing/pass1_analyzer.py.bak` (lines 155-235) but is not called by `core/pipeline.py`.
- **Consequence:** Providing color references will have no effect; they will be ignored.

## ⚠️ Code Duplication

### 4. Duplicate `Pass2Generator`
**Severity: 6/10 (Confusion)**
- **File A:** `core/pass2_generator.py` (Active, Updated with Fixes)
- **File B:** `core/chapter_processing/pass2_generator.py` (Obsolete, Missing Fixes)
- **Duplication:** ~95%
- **Status:** File B is a stale functionality duplicate that poses a risk only if imported by mistake.

### 5. `Pass1Analyzer` Fragmentation
**Severity: 8/10 (Architecture Split)**
- **File A:** `core/pass1_analyzer.py` (Active, Single-Page)
- **File B:** `core/chapter_processing/pass1_analyzer.py.bak` (Obsolete, Batch/Chapter Orchestrator)
- **Analysis:** File A is a simplified version of File B. The "Orchestration" logic (iterating pages, managing DB lifecycle) was moved to `core/pipeline.py`, but the *helper methods* for that orchestration (consolidation, narrative arcs) were left behind in File B.

## 🛠️ Remediation Plan

### Immediate Fixes (Required for Stability)

1.  **Migrate `consolidate_characters`**: Move logic from `pass1_analyzer.py.bak` to `core/database/chapter_db.py`.
2.  **Migrate `process_color_references`**: Move logic from `pass1_analyzer.py.bak` to `core/pipeline.py` (or a helper class).
3.  **Implement `detect_narrative_arcs`**: Create logic for this missing method (likely using `scene_type` from pages).
4.  **Delete Obsolete Files**: Remove `core/chapter_processing/` entirely after verifying migration.

### Refactoring Suggestions (DRY)

1.  **Shared Image Utilities**:
    - Both `yolo_detector.py` and `pass1_analyzer.py` implement bbox context expansion logic.
    - **Suggestion**: Create `core/utils/image_ops.py` for `inflate_bbox`, `create_context_crop`, `extract_canny`.

2.  **Unified Logger**:
    - `pass2_generator.py` has embedded logging setup.
    - `pipeline.py` has simple print statements and callbacks.
    - **Suggestion**: Centralize logging in `core/logging/`.

## DRY Matrix

| Component | Duplication | Suggestion | Effort |
|-----------|-------------|------------|--------|
| Bbox Math | `inflate_bbox` in 3 files | Extract to `core.utils.geometry` | Low |
| Image IO | `Image.open(...).convert('RGB')` everywhere | Extract to `core.utils.io` | Low |
| Config | `STYLE_PRESETS` accessed raw | Create Config Class Wrapper | Medium |

