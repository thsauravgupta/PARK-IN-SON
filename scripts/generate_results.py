# -*- coding: utf-8 -*-
"""
Regenerate presentation figures + RESULTS.md from the saved metrics JSON
without retraining. Run after `python src/main.py`:

    python scripts/generate_results.py
"""
import sys
import logging
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

import yaml
from src.evaluation.results_report import generate_report

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")

if __name__ == "__main__":
    with open(project_root / "config.yaml") as f:
        config = yaml.safe_load(f)
    generate_report(
        project_root / config["paths"]["results"],
        project_root / config["paths"]["figures"],
    )
    print(f"Done. See {project_root / config['paths']['results'] / 'RESULTS.md'}")
