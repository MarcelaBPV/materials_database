#Identificação automática de grupos funcionais com base nos picos Raman.
"""

import pandas as pd
import numpy as np

# Tabela de referência (faixas em cm⁻¹) — valores médios de literatura
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
    "Fingerprint region (complexa)": (400, 1000),
}

def identify_molecular_groups(peaks_df: pd.DataFrame, tolerance: float = 15.0) -> pd.DataFrame:
    """
    Recebe DataFrame de picos ('pos_cm1') e retorna possíveis grupos funcionais.
    tolerance: desvio aceitável em cm⁻¹
    """
    if peaks_df.empty:
        return pd.DataFrame(columns=["Pico (cm⁻¹)", "Grupo funcional provável", "Confiança (%)"])

    results = []
    for _, row in peaks_df.iterrows():
        pos = row["pos_cm1"]
        candidates = []
        for group, (lo, hi) in GROUPS_RANGES.items():
            # Se o pico cair dentro ou próximo da faixa
            if lo - tolerance <= pos <= hi + tolerance:
                # quanto mais perto do centro, maior confiança
                center = (lo + hi) / 2
                confidence = max(0, 100 - abs(pos - center) / (hi - lo) * 100)
                candidates.append((group, confidence))
        if candidates:
            # escolhe o grupo mais provável
            group_best, conf_best = max(candidates, key=lambda x: x[1])
            results.append({
                "Pico (cm⁻¹)": round(pos, 1),
                "Grupo funcional provável": group_best,
                "Confiança (%)": round(conf_best, 1),
            })
    df_result = pd.DataFrame(results).sort_values("Pico (cm⁻¹)").reset_index(drop=True)
    return df_result
