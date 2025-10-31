# -*- coding: utf-8 -*-
"""
raman_processing.py
-------------------
Pipeline completo de processamento Raman compatível com o app principal (app.py).

Usa a biblioteca ramanchada2 (MIT) para pré-processamento, suavização, remoção de baseline,
detecção de picos e comparação espectral via similaridade cosseno.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importa funções da biblioteca oficial do MIT
from ramanchada2.misc import spectrum_similarity
from ramanchada2 import spectrum
Spectrum = spectrum.Spectrum
cosine_similarity = spectrum_similarity.cosine_similarity


# ===============================
# 1) Normalização de DataFrame
# ===============================
def load_raman_dataframe(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Garante que o DataFrame Raman tenha as colunas padronizadas:
    'wavenumber_cm1' e 'intensity_a'.
    """
    df = df_raw.copy()

    # Renomeia colunas comuns
    colmap = {
        "Wavenumber": "wavenumber_cm1",
        "wavenumber": "wavenumber_cm1",
        "Raman Shift": "wavenumber_cm1",
        "Raman_Shift": "wavenumber_cm1",
        "Intensity": "intensity_a",
        "intensity": "intensity_a",
        "Counts": "intensity_a",
        "y": "intensity_a",
        "x": "wavenumber_cm1",
    }
    df = df.rename(columns=colmap)

    required_cols = {"wavenumber_cm1", "intensity_a"}
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"Arquivo Raman inválido. Esperado colunas: {required_cols}, encontrado: {df.columns.tolist()}"
        )

    df = df.dropna(subset=["wavenumber_cm1", "intensity_a"])
    df = df.sort_values("wavenumber_cm1").reset_index(drop=True)
    return df[["wavenumber_cm1", "intensity_a"]]


# ===============================
# 2) Pré-processamento
# ===============================
def preprocess_spectrum(df_spec: pd.DataFrame, smooth: bool = True, baseline: bool = True) -> Spectrum:
    """
    Converte o DataFrame em um objeto Spectrum (ramanchada2)
    e aplica etapas opcionais de suavização e remoção de baseline.
    """
    spec = Spectrum(x=df_spec["wavenumber_cm1"].values, y=df_spec["intensity_a"].values)

    if baseline:
        spec = spec.baseline_subtract()
    if smooth:
        spec = spec.smooth(smoothness=5)

    # Normaliza entre 0 e 1
    y = spec.y - np.min(spec.y)
    if np.max(y) > 0:
        y = y / np.max(y)
    spec.y = y

    return spec


# ===============================
# 3) Pipeline completo
# ===============================
def process_raman_pipeline(
    df_spec: pd.DataFrame,
    smooth: bool = True,
    baseline: bool = True,
    peak_prominence: float = None,
):
    """
    Executa o pipeline Raman completo:
      1. Pré-processamento
      2. Detecção de picos
      3. Geração de gráfico matplotlib

    Retorna:
      - Spectrum processado
      - DataFrame com picos detectados
      - Figura matplotlib pronta para exibir
    """
    spec = preprocess_spectrum(df_spec, smooth=smooth, baseline=baseline)

    # Detecção de picos
    try:
        if peak_prominence is not None and peak_prominence > 0:
            peaks = spec.find_peaks(prominence=peak_prominence)
        else:
            peaks = spec.find_peaks()  # heurística automática
    except Exception:
        # fallback em caso de erro interno da ramanchada2
        peaks = spec.find_peaks()

    peaks_df = pd.DataFrame({
        "pos_cm1": peaks.positions,
        "intensity": peaks.intensities,
    }).sort_values("pos_cm1").reset_index(drop=True)

    # Plot
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(spec.x, spec.y, color="steelblue", lw=1.2, label="Espectro Raman")
    if not peaks_df.empty:
        ax.scatter(peaks_df["pos_cm1"], peaks_df["intensity"], color="red", s=20, label="Picos detectados")
    ax.set_xlabel("Número de onda (cm⁻¹)")
    ax.set_ylabel("Intensidade (u.a.)")
    ax.legend()
    ax.set_title("Processamento Raman — ramanchada2")
    fig.tight_layout()

    return spec, peaks_df, fig


# ===============================
# 4) Comparação entre espectros
# ===============================
def compare_spectra(spec_a: Spectrum, spec_b: Spectrum) -> float:
    """
    Calcula a similaridade entre dois espectros Raman.
    Retorna um valor entre 0 e 1 (quanto maior, mais similares).
    """
    try:
        sim = cosine_similarity(spec_a, spec_b)
        return float(sim)
    except Exception:
        # fallback: interpola para mesmo grid e calcula similaridade manual
        x_common = np.linspace(
            max(spec_a.x.min(), spec_b.x.min()),
            min(spec_a.x.max(), spec_b.x.max()),
            1000,
        )
        y_a = np.interp(x_common, spec_a.x, spec_a.y)
        y_b = np.interp(x_common, spec_b.x, spec_b.y)
        sim = np.dot(y_a, y_b) / (np.linalg.norm(y_a) * np.linalg.norm(y_b))
        return float(sim)
