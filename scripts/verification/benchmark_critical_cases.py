"""Benchmark contínuo AVQV para casos críticos internos.

Uso:
  python scripts/verification/benchmark_critical_cases.py \
      --input-dir output/critical_cases \
      --report-path output/critical_cases/avqv_report.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

from PIL import Image

from core.generation.quality_gate import analyze_avqv_metrics


def iter_images(root: Path):
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
        yield from root.rglob(ext)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--report-path", required=True)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    report_path = Path(args.report_path)

    rows = []
    for img_path in sorted(iter_images(input_dir)):
        try:
            img = Image.open(img_path)
            metrics = analyze_avqv_metrics(img)
            rows.append({"file": str(img_path), **metrics})
        except Exception as exc:
            rows.append({"file": str(img_path), "error": str(exc)})

    valid = [r for r in rows if "error" not in r]
    summary = {
        "count": len(rows),
        "valid_count": len(valid),
        "mean_saturation": mean([r["saturation_mean"] for r in valid]) if valid else None,
        "mean_extreme_pixels": mean([r["extreme_pixels_ratio"] for r in valid]) if valid else None,
        "mean_color_std": mean([r["color_std"] for r in valid]) if valid else None,
    }

    report = {"summary": summary, "rows": rows}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    print(f"[benchmark] imagens: {summary['count']} válidas: {summary['valid_count']}")
    print(f"[benchmark] relatório: {report_path}")


if __name__ == "__main__":
    main()
