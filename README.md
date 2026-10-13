# PFAS accumulation in terrestrial plants

Supporting code and research snapshots for **Patterns and Predictors of Global PFAS Accumulation in Terrestrial Plants and Implications for Dietary Exposure**.

Target repository: [Huzza-13/PFAS-Plant-Accumulation](https://github.com/Huzza-13/PFAS-Plant-Accumulation).

**Status: organized research-support draft, not a fully validated publication release.** The author requested 18 machine-learning scripts numbered 01-18, excluding the rice/vegetable, wheat and world-map scripts. No complete model retraining was performed. Original source files and prior packages remain unchanged. The former 16-script directory was removed before uploading the new workflow; previous versions remain in Git history.

See [Chinese upload list](GITHUB_UPLOAD_LIST_CN.md), [old/new filename mapping](docs/script_numbering.csv), and [release checklist](docs/RELEASE_CHECKLIST_CN.md).

## Contents

| Location | Contents |
|---|---|
| `scripts/` | 18 research scripts plus two path/chain helpers |
| `data/` | Author-designated `Machine-learning dataset.xlsx`, three unchanged historical/supporting workbooks, and existing map resources |
| `reference_results/` | Existing model, SHAP arrays, parameters and result tables; not freshly regenerated |
| `docs/` | Source hashes, numbering map, environment, changes and test records |
| `results/` | New/replayed outputs; excluded from Git and ZIP |

Other supporting files were retained as requested. Historical dietary/map snapshots have unresolved assumptions and are not reproduced by this candidate, which excludes their scripts.

## Setup

Use Python 3.12.3. From the repository root:

```sh
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python check_package.py
python run.py --list
```

Requirements describe the inspected environment, not every historical training run. The saved laboratory joblib reports scikit-learn 1.9.0, while the inspected environment has 1.5.1; loading also produced an XGBoost serialization warning. A prior tissue-SHAP numerical check passed, but the original supported environment remains unconfirmed. See [model-persistence guidance](https://scikit-learn.org/stable/model_persistence.html). Load only trusted joblib files.

## Numbered workflow

Each command runs one analysis, not the entire pipeline. Numbering groups related tasks; independent steps do not require every lower-numbered script.

| New ID | Original ID | Purpose / current manuscript location |
|---|---|---|
| 01 | 03 | Molecular correlations/classes; Fig. S4 |
| 02 | 11 | Laboratory algorithm screening; Fig. S5 / Table S3 |
| 03 | 02 | Predictor-block models |
| 04 | 13 | Predictor-block display; Fig. S7 |
| 05 | 05 | Laboratory nested grouped CV |
| 06 | 06 | Laboratory OOF display; Fig. 3a |
| 07 | 14 | Laboratory fold diagnostics; Fig. S6 |
| 08 | 07 | Final full-data laboratory model for interpretation |
| 09 | 08 | Laboratory overall SHAP and importance |
| 10 | 008 | Laboratory combined SHAP; Fig. 3e |
| 11 | 17 | Chain/class molecular SHAP; Fig. S10 |
| 12 | 19 | One global model, tissue-stratified SHAP; Fig. S8/S9 |
| 13 | 001 | Field four-algorithm nested development |
| 14 | 002 | Field OOF display; Fig. 3b |
| 15 | 003 | Field SHAP/importance; pending parameter review |
| 16 | 005 | Field SHAP beeswarm; Fig. 3f; pending parameter review |
| 17 | 004 | Laboratory/field importance comparison; needs exact field table |
| 18 | 006 | Field validation displays; Fig. S11 |

Examples:

```sh
python run.py 01
python run.py 06 --snapshot
python run.py 07 --snapshot
python run.py 10 --snapshot
python run.py 11 --snapshot
python run.py 12 --snapshot
python run.py 14 --snapshot
python run.py 18 --snapshot
```

Snapshot mode seeds saved inputs in a separate replay directory and refuses training/refitting scripts (02, 03, 05, 08, 13, 15, 16). Replay does not validate historical training. Direct execution is also supported, e.g. `python scripts/06_Laboratory_OOF_performance.py`. Repository-local paths do not depend on the current working directory; `PFAS_RESULTS_DIR` may select another output location. Figure output basenames are preserved, but their parent directories are relocated to avoid writing into the original project.

## Decisions and limits

- The canonical model input is `data/Machine-learning dataset.xlsx`, supplied by the author on 2026-10-13 and copied without modification. Its `Lab model` sheet is identical to the prior laboratory input; its `Field model` sheet is identical to the previous 862-row strict subset except for cleaned reference labels. `read_field_dataset` restores the original stable study IDs from those references, preserving row order and validation groups. See `docs/author_workbook_validation.json`. Historical workbooks are retained for provenance, not used as alternative canonical inputs.

- Laboratory: 825 observations, 13 raw predictors. Field: 862 strict-matched observations, 11 raw predictors, 35 Study-PFAS combinations from three studies.
- Study-PFAS grouping holds each study-compound combination within one fold; it is not whole-study holdout.
- Script 13 uses ordinary `GradientBoostingRegressor` in the GB comparison branch, as confirmed by the author. Unsupported `max_iter` was repaired to `n_estimators=[200,300]`. Other candidates, seeds, R2 scoring and GridSearchCV are retained. Parameter checks did not fit models; historical GB metrics were not regenerated. The existing best field model remains XGBoost, not GB.
- Scripts 15/16 remain gated until fixed XGBoost parameters are reconciled with the final field model. Script 17 needs exact laboratory importance from 09 and exact field importance from reviewed 15. The latter table is absent from saved snapshots; no rounded-constant fallback is supplied.
- Chain length is an interpretation variable, not a final-model predictor. Scripts 01/11 use conventional Cn for PFCA/PFSA (PFOA=8, PFOS=8), and total molecular carbon number for other listed PFAS (8:2 diPAP=20), not perfluorinated-carbon count. Unknown compounds fail. Original Excel values remain unchanged; Fig. S4 needs review after correction.
- SHAP is descriptive attribution, not causal evidence or OOF validation.
- OriginPro analyses, external ChemDes generation and the excluded dietary/map analyses are not reproduced. No full-paper reproduction claim is made.

## Sharing

No code/data licence was invented. Confirm ownership, redistribution rights, source attribution and the training environment before a final publication release; see `RIGHTS.md`. After validation, use a fixed release and actual accessible identifier. A request to upload is not proof of successful upload.
