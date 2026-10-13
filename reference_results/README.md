# Existing saved outputs

These files are copied research outputs, not newly computed or newly validated results. `docs/source_manifest.csv` records their original locations relative to the working code directory and SHA-256 checksums.

The canonical laboratory snapshot was selected from the original `no_chain_PFASclass` directory, rather than mixing it with the root directory's different final SHAP/model files. Field snapshots come from `Field_Nested_ML_results`; existing supplementary validation tables were also copied.

`Fig5a_rice_vegetable_updated`, `Fig5a_rice_vegetable_RFD_corrected`, and `Wheat_figure_outputs` are retained for comparison only. Their names do not establish methodological correctness. See the release checklist.

The precise full field importance table needed by current script 17 (formerly 004) was not present. It must be generated from the reviewed final field model, not reconstructed from a plotted image or rounded constants.

The author excluded the rice/vegetable, wheat and map scripts. Their old result tables remain as supporting snapshots, not outputs reproduced by the 18 included scripts. The previously flagged map screening benchmark and wheat scaling assumptions remain unresolved, not fixed by exclusion.

The `.joblib` model is a trusted local research artifact. Load only trusted files: pickle/joblib formats may execute code. Training code and inputs are supplied as well; the model file is not a substitute for them.
