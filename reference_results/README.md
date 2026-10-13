# Existing saved outputs

These files are copied research outputs, not newly computed or newly validated results. `docs/source_manifest.csv` records their original locations relative to the working code directory and SHA-256 checksums.

The canonical laboratory snapshot was selected from the original `no_chain_PFASclass` directory, rather than mixing it with the root directory's different final SHAP/model files. Field snapshots come from `Field_Nested_ML_results`; existing supplementary validation tables were also copied.

Only machine-learning snapshots associated with the 18 included scripts are retained. Dietary/rice/vegetable/wheat historical outputs were removed because their analyses are outside this repository's scope.

The precise full field importance table needed by current script 17 (formerly 004) was not present. It must be generated from the reviewed final field model, not reconstructed from a plotted image or rounded constants.

The supplied model, OOF predictions, transformed inputs, SHAP arrays, parameter tables and figure-source tables are retained for inspection and snapshot replay. Their presence does not establish that complete training has been reproduced. See the release checklist for unresolved machine-learning issues.

The `.joblib` model is a trusted local research artifact. Load only trusted files: pickle/joblib formats may execute code. Training code and inputs are supplied as well; the model file is not a substitute for them.
