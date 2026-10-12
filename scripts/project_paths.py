"""Repository-local paths and explicit gates for unresolved research decisions."""
import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = Path(os.environ.get("PFAS_RESULTS_DIR", str(PROJECT_ROOT / "results/recomputed"))).resolve()
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def require_review(*issues):
    status = json.loads((PROJECT_ROOT / "docs/review_status.json").read_text(encoding="utf-8"))
    unresolved = [key for key in issues if not status[key]["resolved"]]
    if unresolved:
        details = "\n".join(f"- {key}: {status[key]['reason']}" for key in unresolved)
        raise RuntimeError("This script is withheld from the publication workflow pending author review.\n"
                           + details + "\nSee docs/RELEASE_CHECKLIST_CN.md. No analysis was started.")
