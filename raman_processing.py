# raman_processing/pipeline.py
from __future__ import annotations
import io
import numpy as np
import pandas as pd

# Tentativa de importar ramanchada2; se não houver, usa SciPy (fallback)
RC_AVAILABLE = True
try:
    # a API do rc2 pode mudar; encapsulamos em funções para isolar chamadas
    import ramanchada2 as rc2
except Exception:  # ImportError ou outros
    RC_AVAILABLE = False

from scipy.signal import find_peaks

def _df_from_csv(file_bytes: bytes) -> pd.DataFrame:
    buf = io.BytesIO(file_bytes)
    df = pd.read_csv(buf)
    # tenta detectar nomes de colunas comuns
    cols = [c.lower().strip() for c in df.columns]
    df.columns = cols
    # suportados: wavenumber_cm1/intensity_a OU x/y
    if "wavenumber_cm1" in df.columns and "intensity_a" in df.columns:
        x = df["wavenumber_cm1"].to_numpy(float)
        y = df["intensity_a"].to_numpy(float)
    elif "x" in df.columns and "y" in df.columns:
        x = df["x"].to_numpy(float)
        y = df["y"].to_numpy(float)
    else:
        # tentativa: primeiras duas colunas
        x = df.iloc[:, 0].to_numpy(float)
        y = df.iloc[:, 1].to_numpy(float)
    return pd.DataFrame({"x": x, "y": y})


# ---------------- RC2 wrappers (com fallback seguro) ---------------- #

def _rc2_make_spectrum(x: np.ndarray, y: np.ndarray):
    """
    Cria um 'Spectrum' do ramanchada2 se disponível; caso contrário,
    retorna tupla (x,y) para o caminho alternativo.
    """
    if RC_AVAILABLE:
        # Alguns exemplos do projeto usam rc2.spectrum.Spectrum
        # e operações decoradas (normalize, baseline, etc.)
        # Mantemos chamadas defensivas.
        if hasattr(rc2, "spectrum") and hasattr(rc2.spectrum, "Spectrum"):
            return rc2.spectrum.Spectrum(x=x, y=y)
    return (x, y)


def preprocess_spectrum(
    df_xy: pd.DataFrame,
    *,
    do_despike: bool = True,
    do_baseline: bool = True,
    do_smooth: bool = True,
    normalize: str | None = "l2",
    resample_step: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Retorna x_proc, y_proc prontos para exibir e detectar picos.
    Usa ramanchada2 quando disponível; senão aplica um fallback simples.
    """
    x = df_xy["x"].to_numpy()
    y = df_xy["y"].to_numpy()

    if RC_AVAILABLE:
        s = _rc2_make_spectrum(x, y)

        # Despike (cosmic rays)
        if do_despike and hasattr(rc2, "processing"):
            if hasattr(rc2.processing, "spikes"):
                try:
                    s = rc2.processing.spikes.remove_spikes(s)  # nome típico
                except Exception:
                    pass  # segue com o dado sem travar

        # Baseline
        if do_baseline:
            # Existem várias opções em rc2 (freq-domain baseline, ALS etc.)
            try:
                if hasattr(rc2.processing, "baseline"):
                    s = rc2.processing.baseline.subtract_baseline(s)
            except Exception:
                pass

        # Suavização
        if do_smooth:
            try:
                if hasattr(rc2.processing, "smooth"):
                    s = rc2.processing.smooth.savgol(s, window_length=11, polyorder=3)
            except Exception:
                pass

        # Normalização
        if normalize:
            try:
                if hasattr(rc2.spectrum, "normalize"):
                    s = rc2.spectrum.normalize.normalize(s, method=normalize)
            except Exception:
                pass

        # Reamostragem opcional (passo regular de cm-1)
        if resample_step:
            try:
                if hasattr(rc2.spectrum, "set_new_xaxis"):
                    s = rc2.spectrum.set_new_xaxis.set_new_xaxis(
                        s, new_x=np.arange(float(np.min(s.x)), float(np.max(s.x)), resample_step)
                    )
            except Exception:
                pass

        # Extrai arrays finais (API típica .x/.y)
        x_out = np.array(getattr(s, "x", x), dtype=float)
        y_out = np.array(getattr(s, "y", y), dtype=float)
        return x_out, y_out

    # ---------- Fallback leve (SciPy / NumPy), para não travar ----------
    # Suavização Savitzky-Golay como aproximação
    if do_smooth:
        try:
            from scipy.signal import savgol_filter
            win = 11 if len(y) >= 11 else (len(y) // 2) * 2 + 1
            y = savgol_filter(y, window_length=max(win, 5), polyorder=3, mode="interp")
        except Exception:
            pass

    # Baseline (ALS simplificado)
    if do_baseline:
        try:
            lam = 1e5
            p = 0.01
            # ALS baseline (padrão conhecido)
            L = len(y)
            D = np.diff(np.eye(L), 2)
            DTD = D.T @ D
            w = np.ones(L)
            for _ in range(10):
                W = np.diag(w)
                Z = W + lam * DTD
                baseline = np.linalg.solve(Z, w * y)
                w = p * (y > baseline) + (1 - p) * (y < baseline)
            y = y - baseline
        except Exception:
            pass

    # Normalização
    if normalize:
        denom = np.linalg.norm(y, ord=1 if normalize == "l1" else 2) or 1.0
        y = y / denom

    # Reamostragem
    if resample_step:
        new_x = np.arange(float(np.min(x)), float(np.max(x)), resample_step)
        new_y = np.interp(new_x, x, y)
        return new_x, new_y

    return x, y


def detect_peaks(x: np.ndarray, y: np.ndarray, *, prominence: float = 0.01, width: float | None = None) -> pd.DataFrame:
    """
    Detecta picos. Com rc2 usa o método próprio; sem rc2 usa scipy.find_peaks
    """
    if RC_AVAILABLE and hasattr(rc2, "processing") and hasattr(rc2.processing, "peaks"):
        try:
            peaks = rc2.processing.peaks.find_peaks(x=x, y=y, prominence=prominence, width=width)
            # Espera-se que retorne índices/posições; padronizamos numa DataFrame
            return pd.DataFrame({
                "x": np.array(peaks["x"], dtype=float),
                "y": np.array(peaks["y"], dtype=float),
                "prominence": np.array(peaks.get("prominence", [np.nan]*len(peaks["x"]))),
                "width": np.array(peaks.get("widths", [np.nan]*len(peaks["x"]))),
            })
        except Exception:
            pass

    # Fallback SciPy
    # prominence relativo ao range de y
    prom_abs = float(prominence) * (np.max(y) - np.min(y) + 1e-12)
    idx, props = find_peaks(y, prominence=prom_abs, width=width)
    return pd.DataFrame({
        "x": x[idx],
        "y": y[idx],
        "prominence": props.get("prominences", np.full(len(idx), np.nan)),
        "width": props.get("widths", np.full(len(idx), np.nan)),
    })


def process_bytes(file_bytes: bytes, *, resample_step: float | None = 1.0) -> dict:
    """
    Única função de alto nível que o Streamlit chama.
    Ela:
      1) lê CSV -> DataFrame
      2) pré-processa (despike/baseline/smooth/normalize)
      3) detecta picos
    Retorna dict com x, y, peaks_df (para exibição rápida).
    """
    df_xy = _df_from_csv(file_bytes)
    x_p, y_p = preprocess_spectrum(
        df_xy,
        do_despike=True,
        do_baseline=True,
        do_smooth=True,
        normalize="l2",
        resample_step=resample_step,
    )
    peaks = detect_peaks(x_p, y_p, prominence=0.02)
    return {"x": x_p, "y": y_p, "peaks": peaks}
