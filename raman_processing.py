"""
raman_processing.py
Pipeline modular para processamento e otimização de espectros Raman
usando biblioteca do MIT (ramanchada2) e técnicas de IA.
"""

import numpy as np
import pandas as pd
from ramanchada2 import rc2
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import io

# ---------------------- 1. Funções de Pré-Processamento ----------------------

def load_raman_dataframe(df: pd.DataFrame):
    """
    Converte DataFrame em objeto Spectrum do ramanchada2.
    """
    try:
        spectrum = rc2.spectrum.from_array(df["wavenumber_cm1"], df["intensity_a"])
        return spectrum
    except Exception as e:
        raise ValueError(f"Erro ao carregar espectro Raman: {e}")


def preprocess_spectrum(spectrum, smooth_window=9):
    """
    Corrige baseline, suaviza e normaliza o espectro.
    """
    processed = (
        spectrum.copy()
        .baseline_subtract()     # remove ruído de fundo
        .smooth(window_length=smooth_window)
        .normalize(mode="area")  # normalização pela área total
    )
    return processed


# ---------------------- 2. Identificação de Picos ----------------------

def detect_peaks(processed_spectrum, prominence=0.05):
    """
    Detecta picos e retorna posições e intensidades.
    """
    peaks = processed_spectrum.find_peaks(prominence=prominence)
    peak_positions = [p.x for p in peaks]
    peak_intensities = [p.y for p in peaks]
    return pd.DataFrame({"wavenumber_cm1": peak_positions, "intensity_a": peak_intensities})


# ---------------------- 3. Clustering e Similaridade ----------------------

def cluster_spectra(list_of_spectra, n_clusters=3):
    """
    Realiza clusterização de múltiplos espectros para encontrar padrões.
    Retorna rótulos e modelo KMeans.
    """
    all_data = []
    for spec in list_of_spectra:
        all_data.append(spec.y)
    X = np.vstack(all_data)
    X_scaled = StandardScaler().fit_transform(X)
    model = KMeans(n_clusters=n_clusters, random_state=0).fit(X_scaled)
    return model.labels_, model


def compare_spectra(spec1, spec2):
    """
    Calcula similaridade de cosseno entre dois espectros.
    Retorna valor entre 0 e 1.
    """
    y1 = spec1.y / np.linalg.norm(spec1.y)
    y2 = spec2.y / np.linalg.norm(spec2.y)
    sim = cosine_similarity([y1], [y2])[0][0]
    return sim


# ---------------------- 4. Visualização ----------------------

def plot_spectrum_with_peaks(processed_spectrum, peaks_df):
    """
    Gera gráfico matplotlib com picos detectados.
    Retorna objeto fig (para Streamlit).
    """
    fig, ax = plt.subplots()
    ax.plot(processed_spectrum.x, processed_spectrum.y, label="Espectro Raman (tratado)")
    ax.plot(peaks_df["wavenumber_cm1"], peaks_df["intensity_a"], "ro", label="Picos")
    ax.set_xlabel("Número de onda (cm⁻¹)")
    ax.set_ylabel("Intensidade (a.u.)")
    ax.legend()
    ax.grid(True)
    return fig


# ---------------------- 5. Pipeline Principal ----------------------

def process_raman_pipeline(df):
    """
    Executa o pipeline completo de:
    - Leitura do espectro
    - Pré-processamento
    - Detecção de picos
    - Retorna dados e gráfico
    """
    spectrum = load_raman_dataframe(df)
    processed = preprocess_spectrum(spectrum)
    peaks = detect_peaks(processed)
    fig = plot_spectrum_with_peaks(processed, peaks)
    return processed, peaks, fig
