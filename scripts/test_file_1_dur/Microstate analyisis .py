# Microstate analyisis (que o tomas precisa de entender melhor)
# %%
# imports
import mne
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

subject_id = "CA124"
method = "GRAD"

epochs = mne.read_epochs(
    rf"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE_OUTPUT\{subject_id}\{subject_id}_Preproc\04_epochs_FINAL\epochs_divided\{subject_id}_04_epochs_{method}.fif",
    preload=False,
)

# epochs.plot()

meta = epochs.metadata.copy()


# %% gemy
# Microstates analysis

import mne
import pycrostates
from pycrostates.cluster import ModKMeans

# --- NOTA: Microestados funcionam melhor em dados contínuos (Raw) ou Epochs longas. ---
# Vamos assumir que tens o teu objeto 'epochs' com a metadata.

# 1. Escolher uma categoria para testar
# (Para começar, vamos extrair os microestados gerais combinando as epochs dessa categoria)
cat_especifica = epochs.metadata["category"].unique()[0]
epochs_cat = epochs[epochs.metadata["category"] == cat_especifica]

# 2. Inicializar o algoritmo ModKMeans para encontrar 4 microestados
# 4 é o padrão da literatura (Mapas A, B, C, D)
n_clusters = 4
ModK = ModKMeans(n_clusters=n_clusters, random_state=42)

# 3. Treinar o modelo com os teus dados
# O pycrostates faz a extração automática dos picos de GFP para o clustering
ModK.fit(epochs_cat, picks="grad", n_jobs=-1)

# 4. Desenhar os 4 Microestados encontrados
# Isto vai mostrar as 4 topografias cerebrais dominantes desta categoria
ModK.plot()

# 5. Segmentar o sinal e extrair as métricas
# Agora aplicamos estes mapas de volta às epochs para ver a dinâmica temporal
segmentation = ModK.predict(epochs_cat, factor=0, rejection_by_segments=True)

# Extrair as métricas (Duração, Ocorrência, Cobertura)
metrics = segmentation.compute_parameters()

print("\nMétricas dos Microestados para a categoria:", cat_especifica)
for microstate_name, values in metrics.items():
    print(f"\n{microstate_name}:")
    print(f"  Duração Média: {values['meandurs'] * 1000:.2f} ms")
    print(f"  Frequência (Ocorrência): {values['occurrences']:.2f} por segundo")
    print(f"  Cobertura Total: {values['gcoverages'] * 100:.2f} %")
