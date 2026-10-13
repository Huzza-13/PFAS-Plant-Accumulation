# Changes made during package preparation

## Author-designated workbook supplied 2026-10-13

- Added the unchanged Machine-learning dataset.xlsx as the canonical input. Laboratory rows/values/order are identical to the earlier input. Field rows and all non-reference values/order are identical to the previous strict subset; reference labels are cleaned.
- Updated input filenames and field-sheet reading only. The shared reader restores the same S157/S218/S399 labels from verified reference mappings. No predictors, tuning settings, saved metrics, model objects or figure output basenames were changed. No models were fitted.
- See author_workbook_validation.json; historical input workbooks remain available for provenance.

## Upload candidate: 18 scripts, renumbered 01-18

- Author excluded the rice/vegetable, wheat and world-map scripts. They are absent from scripts/ and run.py. Other supporting data/results remain as reference snapshots, not proof of dietary/map reproduction.
- Scripts are numbered by analysis order. See script_numbering.csv. Runtime entry points, parameter checks and provenance paths were updated. Research-script contents, model settings and figure output basenames are unchanged from the reviewed copies.
- Initial preparation on 2026-10-11 had failed GitHub connections and no remote writes. On 2026-10-12, owner browser login enabled deletion of the old 16-script directory (36b5660), followed by upload of 18 numbered research scripts plus two helpers (391a0e9). Old history and a local full-history bundle are retained; supporting materials follow in subsequent commits.
- The upload-candidate run.py sets UTF-8 for its child process after Windows GBK failed while printing R2's superscript symbol. This is an output-encoding repair, not a research-data/model change.

## Previous preparation history (numbers below refer to original filenames)

- Retained the 21 selected research scripts; original numbering, figure basenames, filters, seeds, model predictor sets and plotting layouts were preserved.
- Flattened script locations into `scripts/`; all input workbook paths point to `data/`, output paths to an independent result directory. Supplied the existing map resource components instead of a private Cartopy cache path.
- Added one path helper, one explicit PFAS carbon-number helper, a single-script runner and a small non-training integrity check.
- Fixed 05's missing Path import and inconsistent output_dir variable name.
- Replaced 03's Excel chain source with explicit mapping; 17 imports the same map. The original workbook is unchanged. 03 has been rerun only for descriptor correlations/display; updated Fig. S4 should be checked against the manuscript.
- Replaced 004's manually specified SHAP importance numbers with exact table reads. 003 now exports its computed importance table. The field table still requires a reviewed final model and is not invented from historical plotted values.
- Replaced 006's fixed performance list with its existing result JSON; expected OOF metrics now come from that same JSON. Restricted its search to the selected result directory.
- Replaced 07's old fixed performance printouts with a read of the separate nested OOF performance table.
- Used an available font fallback for 03 without changing the figure's data.
- Added explicit startup gates to field training, field SHAP refitting, map and wheat scripts while their research decisions remain unresolved. The original questionable algorithms/assumptions are preserved behind the gates, not silently corrected.
- Subsequent author decision: field development follows the existing code/results and ordinary GradientBoostingRegressor. Retained R²/GridSearchCV and repaired its unsupported max_iter grid key to n_estimators=[200,300]. Cleared only the field-training method gate, not the field-SHAP/map/wheat gates. The saved best field model remains XGBoost. No training was executed to regenerate GB metrics.
- Copied saved outputs as reference evidence, keeping new analysis/replay outputs separate and excluded from the upload archive.
- Recorded current environment metadata and the observed saved-model version mismatch. No dependencies were installed or upgraded.

No original code/data/manuscript files were edited. No nested training, hyperparameter tuning, model refitting, licence selection, GitHub upload or publication release was performed.
