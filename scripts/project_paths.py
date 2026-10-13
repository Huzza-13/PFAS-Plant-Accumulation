"""Repository-local paths and explicit gates for unresolved research decisions."""
import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = Path(os.environ.get("PFAS_RESULTS_DIR", str(PROJECT_ROOT / "results/recomputed"))).resolve()
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Stable IDs used in the saved grouped-CV outputs. Verified row-by-row against
# the previous strict-matched input; the public workbook has cleaned references.
FIELD_STUDY_IDS = {
    "Liu et al., 2019": "S157",
    "Liu et al., 2022": "S218",
    "Wen et al., 2014": "S399",
}


def read_field_dataset(path):
    """Read the author's already-filtered Field model sheet, not raw candidates."""
    import pandas as pd

    frame = pd.read_excel(path, sheet_name="Field model")
    studies = frame["Reference"].astype(str).str.strip().map(FIELD_STUDY_IDS)
    if studies.isna().any():
        unknown = frame.loc[studies.isna(), "Reference"].astype(str).unique().tolist()
        raise ValueError(f"Unmapped field studies: {unknown}; review the study-ID mapping.")
    frame["Study_ID"] = studies
    # This restores the previous inclusion label for this author-supplied subset;
    # it is not a new independent assessment of observation quality.
    frame["Data quality tier"] = "A_strict_matched"
    return frame


def require_review(*issues):
    status = json.loads((PROJECT_ROOT / "docs/review_status.json").read_text(encoding="utf-8"))
    unresolved = [key for key in issues if not status[key]["resolved"]]
    if unresolved:
        details = "\n".join(f"- {key}: {status[key]['reason']}" for key in unresolved)
        raise RuntimeError("This script is withheld from the publication workflow pending author review.\n"
                           + details + "\nSee docs/RELEASE_CHECKLIST_CN.md. No analysis was started.")
