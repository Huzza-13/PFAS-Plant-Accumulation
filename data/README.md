# Input data and provenance

## Canonical machine-learning input

`Machine-learning dataset.xlsx` is the author-designated model database supplied on 2026-10-13. It is copied byte-for-byte, with SHA-256 `06a95abad0a817d41c1629849588b3990e4cddd3900a50f9a7496a705b57faf8`.

- `Lab model`: 825 rows and 23 columns, identical to the previous laboratory data in both values and order. Laboratory scripts read this first sheet.
- `Field model`: 862 rows and 20 columns. All 19 non-reference columns are identical, row-by-row, to the previous strict-matched subset. The `Reference` labels were cleaned in the author's workbook.
- The public field sheet already represents the selected subset. The shared reader restores `Study_ID` and its previous inclusion label for compatibility; it does not independently grade observation quality or add observations.

| Public reference | Stable saved-model study ID | Rows |
|---|---|---:|
| Liu et al., 2019 | S157 | 363 |
| Liu et al., 2022 | S218 | 357 |
| Wen et al., 2014 | S399 | 142 |

The recovered IDs and row order match the previous input exactly; the same 35 Study-PFAS combinations are preserved. Unmapped references raise an error rather than being assigned a guessed ID. Verification is recorded in `docs/author_workbook_validation.json`. No training was performed.

## Historical and other supporting inputs

The following three workbooks remain unchanged historical/supporting inputs. Original hashes are recorded in `docs/source_manifest.csv`; model scripts now use the canonical workbook above.

| Workbook | Contents and use |
|---|---|
| `Machine-learning original dataset.xlsx` | The actual parent-project input referenced by laboratory scripts: one `Sheet1` with 825 observations and 23 columns |
| `Machine-learning original dataset_with_field_dataset.xlsx` | `Raw data`: 825 rows; `Field dataset`: 876 rows before the strict-matched filter, 862 after filtering |
| `可食部位 2026.09.xlsx` | Crop concentration sheets, dietary parameters, global/country and China-year tables used by dietary/figure scripts |

The first workbook retains its legacy `Carbon Chain Length` values, including 8:2 diPAP=8. Current scripts 01 and 11 (formerly 03 and 17) use the explicit interpretation-only mapping instead. Do not use the unchanged Excel column as a verified chain-length lookup.

## Field and laboratory inputs

The laboratory response is `Log PFASs concentration`; predictors retained in the final model are `Plant tissue group classify`, `Morphotypes`, `Soil pH`, `SOM`, `Exposure Concentration`, `Growth Duration`, `Growth Temperature`, `Chi3v`, `MinPartialCharge`, `TPSA`, `ALogP`, `GATS3c`, and `SpMin8_Bhi`.

Field scripts read the canonical `Field model` sheet through `read_field_dataset`; the old raw `Field dataset` was used only for the equivalence audit. The final field predictor set comprises `Plant tissue group`, `Morphotype`, `Soil pH`, `SOM`, `Measured soil concentration of the corresponding PFAS`, and the same six molecular descriptors. Group identifiers combine restored `Study_ID` and `PFASs Name`.

Missing values remain as represented in the original workbooks and are handled by each analysis script. Category codes and physical units must be interpreted using the manuscript and original source-study records; this preparation does not infer undocumented code meanings or units from column labels alone.

## Dietary inputs

Dietary sheets include Rice, Wheat, Leafy vegetables, Fruit vegetables, root shoot vegetables, Concentrations, Exposure_parameters, world and china, and China-year. The workbook is a working research file and contains unnamed/intermediate columns; no columns have been silently removed or renamed.

The dietary workbook is retained as supporting data in accordance with the author's request to retain other files. Rice/vegetable, wheat and map scripts are excluded. This candidate does not reproduce dietary analyses.

## Maps

`maps/ne_50m_admin_0_countries.*` contains five existing Natural Earth components copied from the original Cartopy cache. They are retained as supporting resources but are not used by the 18 included scripts. Record and verify their upstream version, source and redistribution conditions before public release.

## Rights and bibliographic traceability

The authors must confirm rights to share the compiled data and retain source-study references/DOIs. Data are not licensed by the mere presence of this folder. No full-text third-party papers are bundled.
