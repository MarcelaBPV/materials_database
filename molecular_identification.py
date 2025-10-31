# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

GROUPS_RANGES = {
    "O–H (álcool/fenol)": (3200, 3600),
    "N–H (amina/amida)": (3300, 3500),
    "C–H (alifático/aromático)": (2800, 3100),
    "C=O (carbonila/éster/amida)": (1650, 1750),
    "C=C (alqueno/aromático)": (1500, 1650),
    "C≡C / C≡N": (2100, 2250),
    "C–O / C–N": (1000, 1300),
    "Aromático (anel benzênico)": (1550, 1620),
    "CH2 / CH3 deformação": (1350, 1470),
    "Fingerprint region": (400, 1000),
}

def identify_molecular_groups(peaks_df: pd.DataFrame, tolerance: float = 15.0) -> pd.DataFrame:
    results = []
    for _, row in peaks_df.iterrows():
        pos = row["pos_cm1"]
        for group, (lo, hi) in GROUPS_RANGES.items():
            if lo - tolerance <= pos <= hi + tolerance:
                center = (lo + hi) / 2
                conf = max(0, 100 - abs(pos - center) / (hi - lo) * 100)
                results.append({
                    "Pico (cm⁻¹)": round(pos, 1),
                    "Grupo funcional provável": group,
                    "Confiança (%)": round(conf, 1),
                })
    return pd.DataFrame(results)
