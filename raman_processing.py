"""
raman_processing.py
Pipeline modular para processamento de espectros Raman.
- Tenta usar a biblioteca do MIT (ramanchada2).
- Se não estiver disponível, entra em fallback com SciPy/Numpy.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------- Tentativa de usar ramanchada2 ----------------------
RC2_OK = True
try:
    from ramanchada2 import rc2
except Exception:
    RC2_OK = False

# ---------------------- Fallback (SciPy/Numpy) -----------------------------
if not RC2_OK:
    from scipy.signal import savgol_filter, find_peaks

# ====================== 1. Leitura / Conversão ============================

def load_raman_dataframe(df: pd.DataFrame):
    """
    Se RC2_OK: converte para rc2.spectrum.
    Caso contrário: retorna uma tupla (x, y) numpy.
    """
    if not {"wavenumber_cm1", "intensity_a"}.issubset(df.columns):
        raise ValueError("DataFrame precisa ter colunas: wavenumber_cm1 e intensity_a")

    x = np.asarray(df["wavenumber_cm1"].values, dtype=float)
    y = np.asarray(df["intensity_a"].values, dtype=float)

    if RC2_OK:
        return rc2.spectrum.from_array(x, y)
    else:
        return (x, y)

# ====================== 2. Pré-processamento ===============================

def preprocess_spectrum(spectrum, smooth_window=9):
    """
    Se RC2_OK: baseline_subtract + smooth + normalize('area')
    Fallback: baseline polinomial + Savitzky-Golay + normalização pela área.
    """
    if RC2_OK:
        return (
            spectrum.copy()
            .baseline_subtract()
            .smooth(window_length=smooth_window)
            .normalize(mode="area")
        )

    # ---- Fallback ----
    x, y = spectrum
    # baseline polinomial de grau 3 (simples/rápido)
    coef = np.polyfit(x, y, deg=3)
    baseline = np.poly1d(coef)(x)
    y_corr = y - baseline
    # suavização Savitzky–Golay (janela ímpar)
    window = smooth_window if smooth_window % 2 == 1 else smooth_window + 1
    y_smooth = savgol_filter(y_corr, window_length=window, polyorder=3, mode="interp")
    # normalização pela área
    area = np.trapz(np.abs(y_smooth), x)
    y_norm = y_smooth / (area if area != 0 else 1.0)
    return (x, y_norm)

# ====================== 3. Detecção de picos ===============================

def detect_peaks(processed_spectrum, prominence=0.05):
    """
    Retorna DataFrame com colunas: wavenumber_cm1, intensity_a.
    """
    if RC2_OK:
        peaks = processed_spectrum.find_peaks(prominence=prominence)
        peak_positions = [p.x for p in peaks]
        peak_intensities = [p.y for p in peaks]
        return pd.DataFrame({"wavenumber_cm1": peak_positions, "intensity_a": peak_intensities})

    # ---- Fallback (SciPy) ----
    x, y = processed_spectrum
    # limiar relativo simples
    prom = float(prominence) * (np.max(y) - np.min(y))
    idx, _ = find_peaks(y, prominence=max(prom, 1e-9))
    return pd.DataFrame({"wavenumber_cm1": x[idx], "intensity_a": y[idx]})

# ====================== 4. Similaridade ====================================

def compare_spectra(spec1, spec2):
    """
    Similaridade do cosseno entre dois espectros pré-processados.
    """
    if RC2_OK:
        y1 = spec1.y
        y2 = spec2.y
    else:
        _, y1 = spec1
        _, y2 = spec2

    y1 = y1 / (np.linalg.norm(y1) + 1e-12)
    y2 = y2 / (np.linalg.norm(y2) + 1e-12)
    return float(np.dot(y1, y2))

# ====================== 5. Visualização ====================================

def plot_spectrum_with_peaks(processed_spectrum, peaks_df):
    fig, ax = plt.subplots()
    if RC2_OK:
        ax.plot(processed_spectrum.x, processed_spectrum.y, label="Espectro (tratado)")
    else:
        x, y = processed_spectrum
        ax.plot(x, y, label="Espectro (tratado)")

    ax.plot(peaks_df["wavenumber_cm1"], peaks_df["intensity_a"], "o", label="Picos")
    ax.set_xlabel("Número de onda (cm⁻¹)")
    ax.set_ylabel("Intensidade (a.u.)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig

# ====================== 6. Pipeline principal ===============================

def process_raman_pipeline(df: pd.DataFrame):
    spec = load_raman_dataframe(df)
    processed = preprocess_spectrum(spec)
    peaks = detect_peaks(processed)
    fig = plot_spectrum_with_peaks(processed, peaks)
    return processed, peaks, fig
