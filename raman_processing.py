# -*- coding: utf-8 -*-
"""
Módulo de processamento Raman compatível com app.py
Usa ramanchada2 (MIT) para pré-processamento, detecção de picos e comparação espectral.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Biblioteca oficial MIT
from ramanchada2.spectrum import Spectrum
from ramanchada2.misc.spectrum_similarity import cosine_similarity

# -------------------------------
# Funções principais
# -------------------------------

def load_raman_dataframe(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza DataFrame vindo do Supabase ou CSV.
    Garante colunas 'wavenumber_cm1' e 'intensity_a'.
    """
    df = df_raw.copy()

    # Tentativas de nomes comuns
    colmap = {
        "Wavenumber": "wavenumber_cm1",
        "wavenumber": "wavenumber_cm1",
        "Raman Shift": "wavenumber_cm1",
        "Intensity": "intensity_a",
        "intensity": "intensity_a",
        "Counts": "intensity_a",
    }
    df = df.rename(columns=colmap)

    # Apenas as colunas necessárias
    cols = [c for c in ["wavenumber_cm1", "intensity_a"] if c in df.columns]
    if len(cols) < 2:
        raise ValueError("O espectro precisa conter colunas 'wavenumber_cm1' e 'intensity_a'.")

    # Remove NaNs e ordena
    df = df.dropna(subset=cols)
    df = df.sort_values("wavenumber_cm1").reset_index(drop=True)
    return df[["wavenumber_cm1", "intensity_a"]]


def preprocess_spectrum(df_spec: pd.DataFrame, smooth: bool = True, baseline: bool = True) -> Spectrum:
    """
    Converte o DataFrame em um objeto Spectrum da ramanchada2,
    aplicando suavização e/ou remoção de baseline.
    """
    spec = Spectrum(x=df_spec["wavenumber_cm1"].values, y=df_spec["intensity_a"].values)

    if baseline:
        spec = spec.baseline_subtract()
    if smooth:
        spec = spec.smooth(smoothness=5)

    # Normaliza entre 0–1 para comparação
    y = spec.y - np.min(spec.y)
    y = y / np.max(y)
    spec.y = y

    return spec


def process_raman_pipeline(
    df_spec: pd.DataFrame,
    smooth: bool = True,
    baseline: bool = True,
    peak_prominence: float = None,
):
    """
    Pipeline completo:
      1. Pré-processa espectro
      2. Detecta picos
      3. Retorna (espectro processado, DataFrame de picos, figura matplotlib)
    """
    spec = preprocess_spectrum(df_spec, smooth=smooth, baseline=baseline)

    # Detecção de picos
    if peak_prominence is not None and peak_prominence > 0:
        peaks = spec.find_peaks(prominence=peak_prominence)
    else:
        peaks = spec.find_peaks()  # usa heurística automática

    # Converte em DataFrame
    peaks_df = pd.DataFrame({
        "pos_cm1": peaks.positions,
        "intensity": peaks.intensities,
    }).sort_values("pos_cm1").reset_index(drop=True)

    # Gera figura
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(spec.x, spec.y, color="steelblue", lw=1.2, label="Espectro")
    if not peaks_df.empty:
        ax.scatter(peaks_df["pos_cm1"], peaks_df["intensity"], color="red", s=20, label="Picos detectados")
    ax.set_xlabel("Número de onda (cm⁻¹)")
    ax.set_ylabel("Intensidade (u.a.)")
    ax.legend()
    ax.set_title("Processamento Raman com ramanchada2")
    fig.tight_layout()

    return spec, peaks_df, fig


def compare_spectra(spec_a: Spectrum, spec_b: Spectrum) -> float:
    """
    Compara dois espectros via similaridade cosseno.
    Retorna valor entre 0 e 1 (quanto mais próximo de 1, mais similares).
    """
    try:
        sim = cosine_similarity(spec_a, spec_b)
        return float(sim)
    except Exception:
        # fallback caso espectros tenham grids diferentes
        x_common = np.linspace(
            max(spec_a.x.min(), spec_b.x.min()),
            min(spec_a.x.max(), spec_b.x.max()),
            1000,
        )
        y_a = np.interp(x_common, spec_a.x, spec_a.y)
        y_b = np.interp(x_common, spec_b.x, spec_b.y)
        sim = np.dot(y_a, y_b) / (np.linalg.norm(y_a) * np.linalg.norm(y_b))
        return float(sim)
