# SOLID Principles Audit Report

**Date:** 2026-02-07
**Project:** MangaAutoColor Pro
**Scope:** Core Architecture (`core/`)

## Executive Summary
**Overall Score: 6/10**

The application demonstrates good separation of concerns at the macro level (Pipeline architecture), but individual components exhibit tight coupling (DIP violations) and resistance to extension (OCP violations).

## Detailed Findings

### 1. Single Responsibility Principle (SRP)
> *A module should have one, and only one, reason to change.*

| Module | Status | Score | Findings |
| :--- | :--- | :--- | :--- |
| **TileAwareGenerator** | ⚠️ Violation | 4/10 | **Too many responsibilities**: <br>1. Model Management (Loading/Offloading)<br>2. Tiling Execution strategies<br>3. Image Blending (Low-level numpy ops)<br>4. Text Compositing (OCR-like patch)<br>5. Image Upscaling/Preprocessing |
| **MangaPromptBuilder** | ✅ Pass | 9/10 | Focuses solely on converting data (page/options) into a string prompt. |
| **Pipeline (Orchestrator)** | ✅ Pass | 8/10 | Acts strictly as a workflow coordinator. |

**Remediation:**
Extract image processing logic from `TileAwareGenerator` into specialized processors.

```python
# PROPOSED: Extract Text Compositor
class TextCompositor:
    def compose(self, base_image, original_image, detections):
        # ... logic ...
```

---

### 2. Open/Closed Principle (OCP)
> *Software entities should be open for extension, but closed for modification.*

| Module | Status | Score | Findings |
| :--- | :--- | :--- | :--- |
| **MangaPromptBuilder** | ❌ Violation | 3/10 | **Hardcoded Dictionaries**: `scene_descriptions` is defined inside `build_prompt`. Adding a new scene type requires modifying the code.<br>**Hardcoded Logic**: `_lab_to_color_name` is a massive `if/else` chain. |
| **SceneDetector** | ❌ Violation | 3/10 | Heuristics (brightness/contrast thresholds) are hardcoded. Cannot add new detection strategies without modifying the class. |
| **Pipeline** | ⚠️ Warning | 6/10 | The sequence of passes is hardcoded in `process_chapter`. Adding a "Pass 3" (e.g., Upscaling) requires modifying the main method. |

**Remediation:**
Use Configuration/Strategy pattern for scene types and color mappings.

```python
# PROPOSED: Configurable Scene Types
SCENE_DESCRIPTORS = {
    SceneType.PRESENT: "present day scene...",
    SceneType.FLASHBACK: "flashback scene...",
     # New types can be injected via config
}
```

---

### 3. Liskov Substitution Principle (LSP)
> *Objects of a superclass shall be replaceable with objects of its subclasses.*

| Module | Status | Score | Findings |
| :--- | :--- | :--- | :--- |
| **Generators** | ⚠️ Warning | 5/10 | `TileAwareGenerator` and `Pass2Generator` do not share a common base class or interface, making them non-interchangeable. |
| **Services** | ✅ Pass | 8/10 | `CharacterService` and `NarrativeService` are standalone domain services. |

**Remediation:**
Define an abstract `ImageGenerator` interface.

```python
class ImageGenerator(ABC):
    @abstractmethod
    def generate(self, context: dict) -> Image:
        pass
```

---

### 4. Interface Segregation Principle (ISP)
> *Clients should not be forced to depend upon interfaces that they do not use.*

| Module | Status | Score | Findings |
| :--- | :--- | :--- | :--- |
| **Logger** | ✅ Pass | 8/10 | The logger usage is minimal and consistent (`start_step`, `log_error`). |
| **Options** | ⚠️ Warning | 6/10 | `TileAwareGenerator` takes a generic `options` dictionary or object, which is ambiguous. Explicit configuration objects would be better. |

---

### 5. Dependency Inversion Principle (DIP)
> *Depend upon abstractions, [not] concretions.*

| Module | Status | Score | Findings |
| :--- | :--- | :--- | :--- |
| **TileAwareGenerator** | ❌ Critical | 2/10 | **Hard Dependencies**: Instantiates `MangaPromptBuilder` and `TilingManager` directly in `__init__`.<br>Depends directly on `StableDiffusionXLControlNetPipeline` (concrete class) instead of an abstract `ModelLoader`. |
| **Pipeline** | ❌ Critical | 3/10 | Instantiates `TileAwareGenerator`, `ChapterDatabase`, `YOLODetector` directly. No Dependency Injection container. |

**Remediation:**
Inject dependencies via constructor.

```python
# CURRENT (Violation)
class TileAwareGenerator:
    def __init__(self):
        self._prompt_builder = MangaPromptBuilder() # Tied to concrete class

# PROPOSED (Fix)
class TileAwareGenerator:
    def __init__(self, prompt_builder: PromptBuilderInterface):
        self._prompt_builder = prompt_builder
```

## Critical Fixes Roadmap

1.  **[DIP] Inject Dependencies in `TileAwareGenerator`**:
    *   Pass `MangaPromptBuilder` and `TilingManager` in constructor.
    *   Allows mocking for tests and swapping implementations.

2.  **[SRP] Extract `TextCompositor`**:
    *   Move the complex text restoration logic out of the generator.

3.  **[OCP] data-drive `MangaPromptBuilder`**:
    *   Move scene descriptors to `core/constants.py` or a config file.
