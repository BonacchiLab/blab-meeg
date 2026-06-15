# OMG O TOMAS VAI TENTAR UM GLM
# %%
import mne
import pandas as pd
from eelbrain import *

# --- Configuração ---
subjects = ["CA124", "CA140", "CB013"]
base_path = "C:/Users/tomas/Desktop/COG_MEEG_EXP1_RELEASE_OUTPUT"

epochs_list = []

for sub in subjects:
    fname = f"{base_path}/{sub}/{sub}_Preproc/04_epochs_FINAL/{sub}_04_epochs_FINAL.fif"
    epochs = mne.read_epochs(
        fname, preload=True
    )  # preload=True porque vamos mexer nos dados

    # 1. Resetar a transformação da cabeça (identidade → todos iguais)
    epochs.info["dev_head_t"] = mne.Transform("meg", "head", trans=None)

    # 2. Interpolar canais maus (recupera-os com base nos vizinhos)
    epochs.interpolate_bads()

    # 3. Adicionar coluna 'subject' à metadata
    if epochs.metadata is None:
        epochs.metadata = pd.DataFrame(index=range(len(epochs)))
    epochs.metadata["subject"] = sub

    epochs_list.append(epochs)

# 4. Concatenar todos os sujeitos num único objeto Epochs
all_epochs = mne.concatenate_epochs(epochs_list, add_offset=False)

# 5. Criar o NDVar do Eelbrain (apenas MEG)
ds = load.mne.epochs_ndvar(all_epochs, "meg")

# 6. Adicionar os fatores a partir da metadata (já concatenada)
meta = all_epochs.metadata
for col in ["category", "relevance", "orientation", "duration"]:
    # Se 'duration' for uma variável contínua (ex: 0.5, 1.0), usa Var, senão Factor
    if col == "duration" and meta[col].dtype in ["float64", "int64"]:
        ds[col] = Var(meta[col].astype(float))
    else:
        ds[col] = Factor(meta[col])
ds["subject"] = Factor(meta["subject"])

# 7. Verificar se está tudo correto
print(ds)

# 6. Modelo linear (repensa a fórmula antes de correr!)
result = testnd.LM(
    "meg",
    "category * relevance * orientation * duration",
    data=ds,
    samples=1000,
)


# %%
# %%

import mne
from eelbrain import Dataset, Factor
from eelbrain import load
from eelbrain import testnd

# 1. Carregar épocas de um só sujeito
sub = "CA124"
fname = f"C:/Users/tomas/Desktop/COG_MEEG_EXP1_RELEASE_OUTPUT/{sub}/{sub}_Preproc/04_epochs_FINAL/{sub}_04_epochs_FINAL.fif"
epochs = mne.read_epochs(
    fname, preload=False
)  # preload=True para permitir interpolação

# 3. Criar NDVar (MEG)
meg = load.mne.epochs_ndvar(epochs, "meg")

ds = Dataset()
ds["meg"] = meg


ds["category"] = Factor(epochs.metadata["category"].values)
ds["relevance"] = Factor(epochs.metadata["relevance"].values)
ds["orientation"] = Factor(epochs.metadata["orientation"].values)
ds["duration"] = Factor(epochs.metadata["duration"].values)

"""# 4. Adicionar fatores a partir da metadata
meta = epochs.metadata
for col in ['category', 'relevance', 'orientation', 'duration']:
    # Ajusta o tipo: se 'duration' for contínua, usa Var
    if col == 'duration' and meta[col].dtype in ['float64', 'int64']:
        ds[col] = Var(meta[col].astype(float))
    else:
        ds[col] = Factor(meta[col])

# 5. (Opcional) Ver o dataset
print(ds)
"""
# 6. Modelo linear (ajusta a fórmula conforme as tuas hipóteses)
# Exemplo com efeitos principais + interações duplas (mais leve)
result = testnd.LM(
    "meg",
    "category * relevance * orientation * duration",
    data=ds,
    samples=1000,  # reduz se estiver muito lento
)

# Depois podes visualizar os resultados:
# p = plot.brain.cluster(result, ds=ds)
# %%
