# %%
import numpy as np
from scipy.stats import ttest_ind
import pandas as pd
import mne

epochs = mne.read_epochs(
    "C:/Users/tomas/Desktop/COG_MEEG_EXP1_RELEASE_OUTPUT/CA124/CA124_Preproc/04_epochs_FINAL/CA124_04_epochs_FINAL.fif",
    preload=False,
)

meta = epochs.metadata.copy()

# %%
# Define a janela temporal (ajusta conforme necessário)
tmin, tmax = 1.0, 1.5  # em segundos
# Índices da janela nos dados das epochs
times = epochs.times
mask = (times >= tmin) & (times <= tmax)

# Condições a comparar (garantir que são strings, sem None)
condicoes = sorted(
    epochs.metadata["duration"].dropna().unique()
)  # ex.: ['500ms', '1000ms', '1500ms']

# Dicionário para guardar as amplitudes médias por trial de cada condição
amplitudes = {}

for cond in condicoes:
    # Filtrar os epochs da condição
    epochs_cond = epochs[f"duration == '{cond}'"]
    # Extrair dados apenas dos canais 'mag' e na janela temporal
    data = epochs_cond.get_data(picks="mag")[
        :, :, mask
    ]  # shape: (n_trials, n_channels, n_times)
    # Média primeiro sobre o tempo (dentro da janela), depois sobre os canais
    mean_amp = data.mean(axis=(1, 2))  # shape: (n_trials,)
    amplitudes[cond] = mean_amp

# Mostrar as médias e desvios por condição
for cond, amps in amplitudes.items():
    print(
        f"Condição {cond}: média = {amps.mean():.3e}, DP = {amps.std():.3e}, N = {len(amps)}"
    )

# Testes t independentes entre todos os pares
from itertools import combinations

print("\nResultados dos testes t (não corrigidos):")
pares = list(combinations(condicoes, 2))
resultados = []
for c1, c2 in pares:
    t_stat, p_val = ttest_ind(amplitudes[c1], amplitudes[c2])
    print(f"{c1} vs {c2}: t = {t_stat:.3f}, p = {p_val:.4f}")
    resultados.append((c1, c2, t_stat, p_val))

# Correção de Bonferroni
num_comparacoes = len(pares)
print(f"\nCorreção de Bonferroni (multiplicar p por {num_comparacoes}):")
for c1, c2, t_stat, p_val in resultados:
    p_corr = p_val * num_comparacoes
    sig = (
        "***"
        if p_corr < 0.001
        else "**"
        if p_corr < 0.01
        else "*"
        if p_corr < 0.05
        else "n.s."
    )
    print(f"{c1} vs {c2}: p_corr = {p_corr:.4f} {sig}")
# %%
