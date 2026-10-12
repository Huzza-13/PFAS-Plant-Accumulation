"""Non-training integrity checks; not a substitute for end-to-end reproduction."""
import ast
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))
from pfas_chain_lengths import carbon_chain_lengths


def main():
    checks = []
    for path in sorted((ROOT / "scripts").glob("*.py")):
        source = path.read_text(encoding="utf-8-sig")
        tree = ast.parse(source, filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                value = node.value.replace("\\", "/")
                assert not value.startswith(("D:/python/pythonProject1", "C:/Users/78698")), path.name
    checks.append("Research/helper scripts parse; no executable original private path literals")
    # Execute only imports and estimator/grid definitions, never fit/search/data reads.
    numbered = sorted(p.name for p in (ROOT / "scripts").glob("*.py") if p.name[0].isdigit())
    assert len(numbered) == 18
    assert [name.split("_", 1)[0] for name in numbered] == [f"{i:02}" for i in range(1, 19)]
    excluded = {"Field data concentration_rice_HI.py", "Field data concentration_wheat.py", "world map concentration.py"}
    assert not any((ROOT / "scripts" / name).exists() for name in excluded)
    checks.append("Exactly 18 numbered scripts, 01-18; three author-excluded scripts absent")
    field_tree = ast.parse((ROOT / "scripts/13_Field_nested_modeling.py").read_text(encoding="utf-8"))
    namespace = {}
    for node in field_tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)) or (
            isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in {"cat_cols", "num_cols", "preprocessor", "models"}
        ):
            exec(compile(ast.Module(body=[node], type_ignores=[]), "<field definitions only>", "exec"), namespace)
    for name, (model, grid) in namespace["models"].items():
        pipe = namespace["Pipeline"]([("prep", namespace["preprocessor"]), ("model", model)])
        assert set(grid).issubset(pipe.get_params(deep=True)), name
        pipe.set_params(**{key: values[0] for key, values in grid.items()})
    gb, grid = namespace["models"]["GB"]
    assert type(gb).__name__ == "GradientBoostingRegressor"
    assert grid["model__n_estimators"] == [200, 300]
    assert "model__max_iter" not in grid
    searches = [n for n in ast.walk(field_tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "GridSearchCV"]
    assert any(any(k.arg == "scoring" and isinstance(k.value, ast.Constant) and k.value.value == "r2" for k in n.keywords) for n in searches)
    checks.append("Field four-model parameter names valid; ordinary GB n_estimators grid and R2/GridSearchCV retained; no fitting")
    lab = pd.read_excel(ROOT / "data/Machine-learning original dataset.xlsx")
    assert len(lab) == 825
    chains = carbon_chain_lengths(lab["PFASs Name"])
    assert chains[lab["PFASs Name"].str.strip() == "8:2 diPAP"].eq(20).all()
    assert carbon_chain_lengths(pd.Series(["PFOA", "PFOS", "GenX", "ADONA", "F-53B", "6:2 FTS", "PFOSA", "N-EtFOSA", "N-EtFOSAA", "8:2 diPAP"])).tolist() == [8,8,6,7,8,8,8,10,12,20]
    try:
        carbon_chain_lengths(pd.Series(["unknown compound"]))
    except ValueError:
        pass
    else:
        raise AssertionError("Unmapped PFAS must fail")
    checks.append("Laboratory rows and explicit chain mapping, including unmapped failure")
    field = pd.read_excel(ROOT / "data/Machine-learning original dataset_with_field_dataset.xlsx", sheet_name="Field dataset")
    strict = field[field["Data quality tier"] == "A_strict_matched"]
    assert len(strict) == 862
    assert (strict["Study_ID"].astype(str) + "_" + strict["PFASs Name"].astype(str)).nunique() == 35
    assert strict["Study_ID"].nunique() == 3
    checks.append("Field strict-matched rows, studies, Study-PFAS group count")
    reference = ROOT / "reference_results"
    shap = pd.read_csv(reference / "Final_XGBoost_raw_SHAP_values.csv")
    x = pd.read_csv(reference / "Final_XGBoost_transformed_X.csv")
    meta = pd.read_csv(reference / "Final_XGBoost_SHAP_metadata.csv")
    assert shap.shape == x.shape and shap.columns.equals(x.columns)
    assert len(shap) == len(meta) == 825
    assert np.isfinite(shap.to_numpy()).all()
    checks.append("Saved laboratory SHAP/X/metadata dimensions and numeric finiteness")
    summary = json.loads((reference / "Field_Nested_ML_results/summary.json").read_text(encoding="utf-8"))
    oof = pd.read_csv(reference / "Field_Nested_ML_results/Field_Nested_OOF_predictions.csv")
    y, pred = oof["Observed"].to_numpy(), oof["Predicted"].to_numpy()
    expected = next(r for r in summary["Results"] if r["Model"] == "XGBoost")
    rmse = float(np.sqrt(np.mean((y-pred)**2)))
    r2 = float(1 - np.sum((y-pred)**2) / np.sum((y-y.mean())**2))
    assert len(y) == 862
    np.testing.assert_allclose([r2, rmse], [expected["OOF_R2"], expected["OOF_RMSE"]], atol=1e-6, rtol=0)
    checks.append("Saved field OOF metrics reproduce the accompanying stored summary")
    with (ROOT / "docs/source_manifest.csv").open(encoding="utf-8-sig", newline="") as stream:
        records = list(csv.DictReader(stream))
    for record in records:
        path = ROOT / record["package_path"]
        if record["package_path"].startswith("scripts/"):
            continue  # Paths and documented fixes intentionally differ from source.
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["source_sha256"], path
    checks.append("Copied input/reference/resource files retain source checksums")
    pending = json.loads((ROOT / "docs/review_status.json").read_text(encoding="utf-8"))
    unresolved = [key for key, item in pending.items() if not item["resolved"]]
    report = {"checks_passed": checks, "unresolved_research_decisions": unresolved,
              "complete_training_reproduction": "NOT RUN", "publication_release": "NOT CLEARED" if unresolved else "Requires full-run and rights review"}
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
