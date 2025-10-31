# -*- coding: utf-8 -*-
"""
Processamento Raman (MIT) usando ramanchada2.
Compatível com o Streamlit Cloud.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Imports atualizados conforme estrutura recente do pacote MIT
from ramanchada2 import spectrum
from ramanchada2.misc import spectrum_similarity

Spectrum = spectrum.Spectrum
cosine_similarity = spectrum_similarity.cosine_similarity


# -------------------------------
# 1) Carregamento e normalização
# -------------------------------
def load_raman_dataframe(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()

    colmap = {
        "Wavenumber": "wavenumber_cm1",
        "wavenumber": "wavenumber_cm1",
        "Raman Shift": "wavenumber_cm1",
        "Intensity": "intensity_a",
        "intensity": "intensity_a",
        "Counts": "intensity_a",
    }
    df = df.rename(columns=colmap)

    if not {"wavenumber_cm1", "intensity_a"}.issubset(df.columns):
        raise ValueError("Arquivo precisa conter colunas: 'wavenumber_cm1' e 'intensity_a'.")

    df = df.dropna(subset=["wavenumber_cm1", "intensity_a"])
    df = df.sort_values("wavenumber_cm1").reset_index(drop=True)
    return df[["wavenumber_cm1", "intensity_a"]]


# -------------------------------
# 2) Pré-processamento
# -------------------------------
def preprocess_spectrum(df_spec: pd.DataFrame, smooth=True, baseline=True) -> Spectrum:
    spec = Spectrum(x=df_spec["wavenumber_cm1"].values, y=df_spec["intensity_a"].values)

    if baseline:
        spec = spec.baseline_subtract()
    if smooth:
        spec = spec.smooth(smoothness=5)

    # Normaliza entre 0–1
    y = spec.y - np.min(spec.y)
    spec.y = y / np.max(y)
    return spec


# -------------------------------
# 3) Pipeline completo
# -------------------------------
def process_raman_pipeline(df_spec, smooth=True, baseline=True, peak_prominence=None):
    spec = preprocess_spectrum(df_spec, smooth=smooth, baseline=baseline)

    if peak_prominence is not None and peak_prominence > 0:
        peaks = spec.find_peaks(prominence=peak_prominence)
    else:
        peaks = spec.find_peaks()

    peaks_df = pd.DataFrame({
        "pos_cm1": peaks.positions,
        "intensity": peaks.intensities,
    }).sort_values("pos_cm1").reset_index(drop=True)

    # Gera figura
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(spec.x, spec.y, color="steelblue", lw=1.2, label="Espectro Raman")
    if not peaks_df.empty:
        ax.scatter(peaks_df["pos_cm1"], peaks_df["intensity"], color="red", s=20, label="Picos detectados")
    ax.set_xlabel("Número de onda (cm⁻¹)")
    ax.set_ylabel("Intensidade (u.a.)")
    ax.legend()
    ax.set_title("Processamento Raman (ramanchada2 MIT)")
    fig.tight_layout()

    return spec, peaks_df, fig


# -------------------------------
# 4) Similaridade espectral
# -------------------------------
def compare_spectra(spec_a: Spectrum, spec_b: Spectrum) -> float:
    try:
        return float(cosine_similarity(spec_a, spec_b))
    except Exception:
        x_common = np.linspace(
            max(spec_a.x.min(), spec_b.x.min()),
            min(spec_a.x.max(), spec_b.x.max()),
            1000,
        )
        y_a = np.interp(x_common, spec_a.x, spec_a.y)
        y_b = np.interp(x_common, spec_b.x, spec_b.y)
        sim = np.dot(y_a, y_b) / (np.linalg.norm(y_a) * np.linalg.norm(y_b))
        return float(sim)
