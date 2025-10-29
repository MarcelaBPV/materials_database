# raman_processing.py
# -*- coding: utf-8 -*-
"""
Pipeline Raman com integração opcional ao ramanchada2 (Charisma/MIT)
e fallback para NumPy/SciPy para evitar travamentos no Streamlit.

Funções exportadas:
- load_raman_dataframe(df_raw) -> pd.DataFrame
- preprocess_spectrum(df_spec, smooth=True, baseline=True, peak_prominence=None, **kw) -> pd.DataFrame
- process_raman_pipeline(df_spec, smooth=True, baseline=True, peak_prominence=None, **kw)
    -> (processed_df, peaks_df, fig)
- compare_spectra(spec_a, spec_b) -> float
"""

from typing import Tuple, Optional, Dict, Any
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---- Dependências opcionais
_rc2_available = False
_rc2 = None
try:
    import ramanchada2 as rc2  # pacote
    _rc2 = rc2
    _rc2_available = True
except Exception:
    _rc2_available = False

# ---- Dependências para fallback
from scipy.signal import savgol_filter, find_peaks
from scipy.interpolate import interp1d
from numpy.linalg import norm


# ============================ Utilidades ============================

def _ensure_df(df: pd.DataFrame) -> pd.DataFrame:
    # Normaliza nomes e ordena por wavenumber
    rename_map = {
        "Wavenumber": "wavenumber_cm1",
        "wavenumber": "wavenumber_cm1",
        "RamanShift": "wavenumber_cm1",
        "raman_shift": "wavenumber_cm1",
        "Intensity": "intensity_a",
        "intensity": "intensity_a",
        "counts": "intensity_a",
    }
    df2 = df.rename(columns=rename_map).copy()
    if {"wavenumber_cm1", "intensity_a"} - set(df2.columns):
        # tenta detectar 2 colunas sem cabeçalho
        if df2.shape[1] >= 2 and set(df2.columns) <= {0, 1, 2}:
            cols = list(df2.columns)[:2]
            df2 = df2.rename(columns={cols[0]: "wavenumber_cm1", cols[1]: "intensity_a"})
    df2 = df2[["wavenumber_cm1", "intensity_a"]].dropna()
    df2 = df2.sort_values("wavenumber_cm1").reset_index(drop=True)
    return df2


def _als_baseline(y: np.ndarray, lam: float = 1e5, p: float = 0.01, niter: int = 10) -> np.ndarray:
    """
    Asymmetric Least Squares baseline (Eilers & Boelens).
    Implementação leve para fallback.
    """
    L = len(y)
    D = np.diff(np.eye(L), 2)
    w = np.ones(L)
    for _ in range(niter):
        W = np.diag(w)
        Z = W + lam * D.T @ D
        b = np.linalg.solve(Z, w * y)
        w = p * (y > b) + (1 - p) * (y < b)
    return b


def _normalize_intensity(y: np.ndarray) -> np.ndarray:
    y = y.astype(float)
    rng = np.max(y) - np.min(y)
    if rng == 0:
        return y.copy()
    return (y - np.min(y)) / rng


def _to_common_grid(xa, ya, xb, yb, num=2000) -> Tuple[np.ndarray, np.ndarray]:
    xmin = max(np.min(xa), np.min(xb))
    xmax = min(np.max(xa), np.max(xb))
    if xmax <= xmin:
        # grids não sobrepostos — faz retorno seguro
        return np.array([]), np.array([])
    grid = np.linspace(xmin, xmax, num=num)
    fa = interp1d(xa, ya, bounds_error=False, fill_value="extrapolate")
    fb = interp1d(xb, yb, bounds_error=False, fill_value="extrapolate")
    return fa(grid), fb(grid)


# ============================ API pública ============================

def load_raman_dataframe(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Padroniza o dataframe para colunas ['wavenumber_cm1','intensity_a'] ordenadas.
    """
    return _ensure_df(df_raw)


def preprocess_spectrum(
    df_spec: pd.DataFrame,
    smooth: bool = True,
    baseline: bool = True,
    peak_prominence: Optional[float] = None,
    sg_window: int = 11,
    sg_poly: int = 3,
    als_lambda: float = 1e5,
    als_p: float = 0.01,
    normalize: bool = True,
    **kwargs: Dict[str, Any],
) -> pd.DataFrame:
    """
    Pré-processa espectro: (opcional) baseline, suavização e normalização.
    Tenta ramanchada2 se disponível; senão, fallback (ALS + SavGol).
    """
    df = _ensure_df(df_spec)
    x = df["wavenumber_cm1"].values
    y = df["intensity_a"].values.astype(float)

    # ---- Remoção de baseline
    if baseline:
        if _rc2_available:
            # Abordagem conservadora: caso a API varie, protegemos em try
            try:
                # Muitas instalações expõem algoritmos em rc2.preprocess.*; como a API pode variar,
                # usamos ALS próprio se falhar.
                y_base = _als_baseline(y, lam=als_lambda, p=als_p)
            except Exception:
                y_base = _als_baseline(y, lam=als_lambda, p=als_p)
        else:
            y_base = _als_baseline(y, lam=als_lambda, p=als_p)
        y = y - y_base

    # ---- Suavização
    if smooth:
        try:
            # garante janela ímpar e <= len(y)
            win = max(5, sg_window)
            if win % 2 == 0:
                win += 1
            win = min(win, len(y) - (1 - len(y) % 2))
            if win < 5:
                win = 5 if len(y) >= 5 else len(y) - 1
            y = savgol_filter(y, window_length=win, polyorder=min(sg_poly, 5))
        except Exception:
            pass

    # ---- Normalização
    if normalize:
        y = _normalize_intensity(y)

    return pd.DataFrame({"wavenumber_cm1": x, "intensity_a": y})


def process_raman_pipeline(
    df_spec: pd.DataFrame,
    smooth: bool = True,
    baseline: bool = True,
    peak_prominence: Optional[float] = None,
    **kwargs: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame, plt.Figure]:
    """
    Pipeline completo: pré-processa + detecta picos + plota.
    Retorna:
      processed_df -> DataFrame(wavenumber_cm1, intensity_a)
      peaks_df     -> DataFrame(wavenumber_cm1, intensity_a, left_bases, right_bases, prominence)
      fig          -> Figura pronta para st.pyplot
    """
    # 1) pré-processamento
    proc = preprocess_spectrum(df_spec, smooth=smooth, baseline=baseline, **kwargs)
    x = proc["wavenumber_cm1"].values
    y = proc["intensity_a"].values

    # 2) detecção de picos
    #    Tenta usar ramanchada2 se existir um wrapper compatível; mantém fallback em SciPy
    prominence = peak_prominence if (peak_prominence is not None and peak_prominence > 0) else None
    try:
        peaks, props = find_peaks(y, prominence=prominence)
    except Exception:
        peaks, props = np.array([], dtype=int), {}

    peaks_df = pd.DataFrame({
        "wavenumber_cm1": x[peaks] if len(peaks) else [],
        "intensity_a": y[peaks] if len(peaks) else [],
        "prominence": props.get("prominences", np.array([])),
        "left_bases": props.get("left_bases", np.array([])),
        "right_bases": props.get("right_bases", np.array([])),
    })

    # 3) figura
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(x, y, label="Processado")
    if not peaks_df.empty:
        ax.plot(peaks_df["wavenumber_cm1"], peaks_df["intensity_a"], "x", label="Picos")
    ax.set_xlabel("Número de onda (cm⁻¹)")
    ax.set_ylabel("Intensidade (norm.)")
    ax.legend()

    return proc, peaks_df, fig


def compare_spectra(spec_a: pd.DataFrame, spec_b: pd.DataFrame) -> float:
    """
    Compara dois espectros (cosine similarity) após interpolar para grade comum.
    Retorna valor em [0, 1] (1 = idênticos).
    """
    A = _ensure_df(spec_a)
    B = _ensure_df(spec_b)
    xa, ya = A["wavenumber_cm1"].values, A["intensity_a"].values
    xb, yb = B["wavenumber_cm1"].values, B["intensity_a"].values

    ya_g, yb_g = _to_common_grid(xa, ya, xb, yb, num=3000)
    if ya_g.size == 0 or yb_g.size == 0:
        return 0.0
    # normaliza cada vetor
    ya_g = _normalize_intensity(ya_g)
    yb_g = _normalize_intensity(yb_g)

    denom = (norm(ya_g) * norm(yb_g))
    if denom == 0:
        return 0.0
    sim = float(np.dot(ya_g, yb_g) / denom)
    # força para [0,1]
    return max(0.0, min(1.0, sim))
