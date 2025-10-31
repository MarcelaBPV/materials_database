# -*- coding: utf-8 -*-
"""
Processamento Raman (MIT - ramanchada2)
Compatível com todas as versões conhecidas (antigas e novas)
e com ambientes limitados como o Streamlit Cloud.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# ======================================================
# 🔧 Compatibilidade com diferentes versões do ramanchada2
# ======================================================
try:
    # Versões novas (>= 0.8.0)
    from ramanchada2 import spectrum
    from ramanchada2.similarity.spectrum_similarity import cosine_similarity
except Exception:
    try:
        # Versões antigas (Cloud ou builds desatualizados)
        from ramanchada2 import spectrum
        from ramanchada2.misc.spectrum_similarity import cosine_similarity
    except Exception as e:
        st.error("❌ Não foi possível importar o pacote `ramanchada2`.\n"
                 "Verifique se ele está corretamente instalado no ambiente.")
        st.info("💡 Dica: adicione esta linha ao seu requirements.txt:\n"
                "git+https://github.com/h2020charisma/ramanchada2.git@main#egg=ramanchada2")
        raise e

# Garante compatibilidade universal
Spectrum = spectrum.Spectrum


# ======================================================
# 1️⃣ Carregamento e normalização de dados Raman
# ======================================================
def load_raman_dataframe(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Normaliza nomes de colunas e prepara o DataFrame para processamento."""
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
        raise ValueError("❌ O arquivo precisa conter colunas: 'wavenumber_cm1' e 'intensity_a'.")

    df = df.dropna(subset=["wavenumber_cm1", "intensity_a"])
    df = df.sort_values("wavenumber_cm1").reset_index(drop=True)
    return df[["wavenumber_cm1", "intensity_a"]]


# ======================================================
# 2️⃣ Pré-processamento espectral
# ======================================================
def preprocess_spectrum(df_spec: pd.DataFrame, smooth=True, baseline=True) -> Spectrum:
    """Aplica suavização, remoção de baseline e normalização 0–1."""
    spec = Spectrum(x=df_spec["wavenumber_cm1"].values, y=df_spec["intensity_a"].values)

    if baseline:
        spec = spec.baseline_subtract()
    if smooth:
        spec = spec.smooth(smoothness=5)

    y = spec.y - np.min(spec.y)
    spec.y = y / np.max(y)
    return spec


# ======================================================
# 3️⃣ Pipeline completo de processamento
# ======================================================
def process_raman_pipeline(df_spec, smooth=True, baseline=True, peak_prominence=None):
    """
    Executa o pipeline completo:
    - Pré-processa o espectro
    - Detecta picos
    - Gera gráfico e DataFrame de picos
    """
    spec = preprocess_spectrum(df_spec, smooth=smooth, baseline=baseline)

    if peak_prominence is not None and peak_prominence > 0:
        peaks = spec.find_peaks(prominence=peak_prominence)
    else:
        peaks = spec.find_peaks()

    peaks_df = pd.DataFrame({
        "pos_cm1": peaks.positions,
        "intensity": peaks.intensities,
    }).sort_values("pos_cm1").reset_index(drop=True)

    # Gráfico Raman
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(spec.x, spec.y, color="steelblue", lw=1.3, label="Espectro Raman")
    if not peaks_df.empty:
        ax.scatter(peaks_df["pos_cm1"], peaks_df["intensity"], color="crimson", s=22, label="Picos detectados")

    ax.set_xlabel("Número de onda (cm⁻¹)")
    ax.set_ylabel("Intensidade (u.a.)")
    ax.legend()
    ax.set_title("Processamento Raman (ramanchada2 MIT)")
    fig.tight_layout()

    return spec, peaks_df, fig


# ======================================================
# 4️⃣ Similaridade espectral
# ======================================================
def compare_spectra(spec_a: Spectrum, spec_b: Spectrum) -> float:
    """
    Calcula a similaridade espectral entre dois espectros Raman.
    Retorna valor entre 0 (diferente) e 1 (idêntico).
    """
    try:
        return float(cosine_similarity(spec_a, spec_b))
    except Exception:
        # Caso a versão do pacote não tenha a função direta
        x_common = np.linspace(
            max(spec_a.x.min(), spec_b.x.min()),
            min(spec_a.x.max(), spec_b.x.max()),
            1000,
        )
        y_a = np.interp(x_common, spec_a.x, spec_a.y)
        y_b = np.interp(x_common, spec_b.x, spec_b.y)
        sim = np.dot(y_a, y_b) / (np.linalg.norm(y_a) * np.linalg.norm(y_b))
        return float(sim)
