# -*- coding: utf-8 -*-
"""
Identificação molecular para amostra de sangue + prata + papel
"""

import pandas as pd

# ======================== Tabela de referência Raman ========================
BLOOD_RAMAN_REFERENCE = [
    {"freq_cm1": 713,  "atributo": "γ₁₁ - vibração fora do plano do anel pirrólico", "componente": "Heme (hemoglobina)"},
    {"freq_cm1": 750,  "atributo": "Vibração característica do triptofano", "componente": "Aminoácido (proteína)"},
    {"freq_cm1": 968,  "atributo": "δ(C-O-H) - dobramento de carboidratos", "componente": "Carboidratos"},
    {"freq_cm1": 1004, "atributo": "νₛ(C-C) - estiramento simétrico da fenilalanina", "componente": "Aminoácido (proteína)"},
    {"freq_cm1": 1122, "atributo": "Estiramento C-CT de carboidratos", "componente": "Carboidratos"},
    {"freq_cm1": 1252, "atributo": "Amida III", "componente": "Proteínas"},
    {"freq_cm1": 1342, "atributo": "Deformação CH₂ de lipoproteínas", "componente": "Lipoproteínas"},
    {"freq_cm1": 1370, "atributo": "ν₄ - vibração do anel pirrólico semi-simétrico", "componente": "Heme (hemoglobina)"},
    {"freq_cm1": 1454, "atributo": "Vibrações de colágeno e fosfolipídios", "componente": "Colágeno/Fosfolipídios"},
    {"freq_cm1": 1575, "atributo": "δ(C=C) - dobramento da fenilalanina", "componente": "Aminoácido (proteína)"},
    {"freq_cm1": 1598, "atributo": "ν(C=C) - estiramento da hemoglobina", "componente": "Hemoglobina"},
]

BLOOD_REF_DF = pd.DataFrame(BLOOD_RAMAN_REFERENCE)

# ======================== Função de identificação ========================
def identify_blood_components(peaks_df: pd.DataFrame, tolerance: float = 10.0) -> pd.DataFrame:
    """
    Associa picos Raman de amostra de sangue à tabela de referência.
    
    Parâmetros
    ----------
    peaks_df : pd.DataFrame
        DataFrame com coluna 'pos_cm1' dos picos detectados
    tolerance : float
        Tolerância em cm⁻¹ para correspondência
    
    Retorna
    -------
    pd.DataFrame
        DataFrame com colunas:
        - Pico (cm⁻¹)
        - Frequência Referência (cm⁻¹)
        - Atribuição Molecular
        - Componente Químico
        - Confiança (%)
    """
    results = []
    for _, peak in peaks_df.iterrows():
        pos = peak["pos_cm1"]
        for _, ref in BLOOD_REF_DF.iterrows():
            ref_freq = ref["freq_cm1"]
            if abs(pos - ref_freq) <= tolerance:
                conf = max(0, 100 - abs(pos - ref_freq) / tolerance * 100)
                results.append({
                    "Pico (cm⁻¹)": round(pos, 1),
                    "Frequência Referência (cm⁻¹)": ref_freq,
                    "Atribuição Molecular": ref["atributo"],
                    "Componente Químico": ref["componente"],
                    "Confiança (%)": round(conf, 1)
                })
    return pd.DataFrame(results)
