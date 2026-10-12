"""Explicit interpretation-only carbon numbers; not model predictors.

PFCAs/PFSAs use conventional Cn naming, including the carboxyl carbon for
PFCAs (PFOA=8, PFOS=8). Other entries use total molecular carbon number,
consistent with the author's C Chain convention. These are NOT counts of
C-F bonds or perfluorinated carbons. F-53B here denotes the C8/6:2 component,
not a carbon number applicable to every component of a commercial mixture.
"""
PFAS_CHAIN_LENGTH_MAP = {
    "PFBA": 4, "PFPeA": 5, "PFHxA": 6, "PFHpA": 7, "PFOA": 8,
    "PFNA": 9, "PFDA": 10, "PFUnDA": 11, "PFDoDA": 12,
    "PFTriDA": 13, "PFTeDA": 14,
    "PFBS": 4, "PFHxS": 6, "PFOS": 8,
    "GenX": 6, "HFPO-DA": 6, "HFPO-DA (GenX)": 6, "GenX (HFPO-DA)": 6,
    "ADONA": 7, "F-53B": 8, "6:2 Cl-PFESA": 8,
    "6:2 FTS": 8, "6:2FTS": 8, "PFOSA": 8, "FOSA": 8,
    "N-EtFOSA": 10, "N-EtFOSAA": 12, "8:2 diPAP": 20,
}


def carbon_chain_lengths(names):
    clean = names.astype(str).str.strip()
    values = clean.map(PFAS_CHAIN_LENGTH_MAP)
    if values.isna().any():
        unknown = sorted(clean[values.isna()].unique())
        raise ValueError(f"Unmapped PFAS carbon-chain lengths: {unknown}. Update the explicit mapping first.")
    return values.astype(int)
