# -*- coding: utf-8 -*-
"""
Processamento Raman SEM ramanchada2
Compatível com Streamlit Cloud
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks
from sklearn.metrics.pairwise import cosine_similarity


# ======================================================
# 1️⃣ Carregamento e normalização de dados Raman
# ======================================================
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
        raise ValueError("Arquivo precisa ter colunas: wavenumber_cm1, intensity_a")

    df = df.dropna(subset=["wavenumber_cm1", "intensity_a"])
    df = df.sort_values("wavenumber_cm1").reset_index(drop=True)
    return df


# ======================================================
# 2️⃣ Funções auxiliares
# ======================================================
def baseline_correction(y, lam=1e5, p=0.001):
    """
    baseline estilo ALS (Asymmetric Least Squares)
    """
    from scipy import sparse
    from scipy.sparse.linalg import spsolve

    L = len(y)
    D = sparse.diags([1,-2,1],[0,-1,-2], shape=(L,L-2))
    w = np.ones(L)
    for _ in range(10):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.T)
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1-p) * (y < z)
    return y - z


# ======================================================
# 3️⃣ Pipeline completo
# ======================================================
def process_raman_pipeline(df, smooth=True, baseline=True):
    x = df["wavenumber_cm1"].values
    y = df["intensity_a"].values.astype(float)

    # baseline
    if baseline:
        y = baseline_correction(y)

    # smooth
    if smooth:
        y = savgol_filter(y, window_length=15, polyorder=3)

    # normalize
    y = (y - np.min(y)) / (np.max(y) - np.min(y))

    # peaks
    peaks, props = find_peaks(y, prominence=0.05)
    peaks_df = pd.DataFrame({
        "pos_cm1": x[peaks],
        "intensity": y[peaks],
        "prominence": props["prominences"]
    })

    # plot
    fig, ax = plt.subplots(figsize=(7,4))
    ax.plot(x, y, label="Espectro Processado")
    ax.scatter(x[peaks], y[peaks], color="red")
    ax.set_title("Processamento Raman (SEM ramanchada2)")
    ax.set_xlabel("Número de onda (cm⁻¹)")
    ax.set_ylabel("Intensidade (normalizada)")
    ax.grid(True)

    return (x, y), peaks_df, fig


# ======================================================
# 4️⃣ Similaridade espectral
# ======================================================
def compare_spectra(spec1, spec2):
    x1, y1 = spec1
    x2, y2 = spec2
    
    x_common = np.linspace(max(x1.min(), x2.min()), min(x1.max(), x2.max()), 1000)
    y1_interp = np.interp(x_common, x1, y1)
    y2_interp = np.interp(x_common, x2, y2)
    
    cos = cosine_similarity(y1_interp.reshape(1, -1), y2_interp.reshape(1, -1))[0][0]
    return float(cos)
