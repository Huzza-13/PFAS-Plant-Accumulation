"""Run one selected analysis; never silently retrain all models."""
import argparse
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def snapshot_results():
    source = ROOT / "reference_results"
    destination = ROOT / "results/snapshot_replay"
    # Seed only immutable inputs, not tables that figure scripts regenerate.
    inputs = {
        "Feature_block_model_results_final15.csv",
        "Nested_XGBoost_OOF_predictions.csv",
        "Final_XGBoost_pipeline.joblib",
        "Final_XGBoost_raw_SHAP_values.csv",
        "Final_XGBoost_transformed_X.csv",
        "Final_XGBoost_transformed_feature_names.csv",
        "Final_XGBoost_SHAP_metadata.csv",
        "Overall_aggregated_SHAP_importance.csv",
        "Field_Nested_ML_results/summary.json",
        "Field_Nested_ML_results/Field_Nested_OOF_predictions.csv",
    }
    for path in sorted(source.rglob("*")):
        if not path.is_file() or path.relative_to(source).as_posix() not in inputs:
            continue
        target = destination / path.relative_to(source)
        if target.exists():
            digest = lambda p: hashlib.sha256(p.read_bytes()).digest()
            if digest(path) != digest(target):
                raise RuntimeError(f"Snapshot input was changed: {target}. Use a separate result directory.")
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    return destination


def main():
    scripts = {p.name.split("_", 1)[0]: p for p in (ROOT / "scripts").glob("*.py")
               if p.name[0].isdigit()}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("script", nargs="?", choices=sorted(scripts))
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--snapshot", action="store_true", help="Replay figures from existing saved outputs, not new training")
    args = parser.parse_args()
    if args.list:
        for key, path in sorted(scripts.items()):
            print(f"{key:>5}  {path.name}")
        return
    if not args.script:
        parser.error("Choose a script ID or --list")
    if args.snapshot and args.script in {"02", "03", "05", "08", "13", "15", "16"}:
        parser.error("Training/refitting cannot run in snapshot replay mode")
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"  # Windows consoles otherwise may fail on scientific symbols.
    env["PYTHONIOENCODING"] = "utf-8"
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("PFAS_SHAP_CHAIN_FIG_SHOW", "0")
    if args.snapshot:
        env["PFAS_RESULTS_DIR"] = str(snapshot_results())
    result = subprocess.run([sys.executable, str(scripts[args.script])], cwd=ROOT, env=env)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
