#cluster based Permutations tests/Decoding/temporal generalization
 

#%%
#Import Room
import mne
import numpy as np
import os
import matplotlib 
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
from pathlib import Path
from mne.preprocessing import find_bad_channels_maxwell, maxwell_filter, ICA


#%%

raw = mne.io.read_raw_fif("C:\\Users\\tomas\\Desktop\\MEG_outputs\\raw_clean-ica.fif", preload=True)


#%%
#STI - Event Categorization
#1) Definir os Canal de triggers 
stim_channel = "STI101"
#2) Extrair eventos (ajustando min_duration para evitar erros) 
events = mne.find_events(
    raw,  
    stim_channel=stim_channel,
    shortest_event=1,  
    min_duration=0.001,  
    verbose=True
)
#3) Função para categorizar 
def categorize(event_id):
    if 1 <= event_id <= 20:
        return "faces"
    elif 21 <= event_id <= 40:
        return "objects"
    elif 41 <= event_id <= 60:
        return "fonts"
    elif 61 <= event_id <= 80:
        return "false_fonts"
    else:
        return None
#4) Criar dicionário com todos os IDs de cada categoria
event_dict = {}
for e in np.unique(events[:, 2]):
    label = categorize(e)
    if label is not None:
        if label not in event_dict:
            event_dict[label] = []
        event_dict[label].append(e)
#Verificar
for cat, ids in event_dict.items():
    print(f"{cat}: {ids}")
#5) Criar dicionário para plot (cada ID com label única)
plot_dict = {}
for cat, ids in event_dict.items():
    for e_id in ids:
        plot_dict[f"{cat}_{e_id}"] = e_id
#6) Plotar eventos
fig = mne.viz.plot_events(
    events,
    event_id=plot_dict,
    sfreq=raw.info["sfreq"],
    first_samp=raw.first_samp,
)
#7) Critérios de rejeição
reject_criteria = dict(
    mag=4000e-15,
    grad=4000e-13,
    eeg=400e-6, #O criterio de rejeiçao do EEG com MEG é diferente de um normal 
    
)
#8) Criar dicionário "flat" para todas as categorias
flat_event_id = plot_dict.copy()  # Mais eficiente


#%%
# --- 9) Criar epochs com todos os eventos ---
epochs = mne.Epochs(
    raw,
    events,
    event_id=flat_event_id,
    tmin=-0.2,
    tmax=0.8,
    baseline=(-0.2, 0),   # ou (None, 0)
    reject=reject_criteria,
    preload=True
)
#%%
print(raw.info['bads'])

#%% 
epochs.plot_drop_log()
 
#%%
# Criar evoked responses para cada categoria
evoked_dict = {}
for category in event_dict.keys():
    category_labels = [f"{category}_{i}" for i in event_dict[category] if f"{category}_{i}" in epochs.event_id]
    if category_labels and len(epochs[category_labels]) > 0:
        evoked_dict[category] = epochs[category_labels].average()
# Verificar que tipos de canais MEG temos disponíveis
first_evoked = evoked_dict[list(evoked_dict.keys())[0]]
available_ch_types = set(first_evoked.get_channel_types())
print(f"Types of Channels Available: {available_ch_types}")
# Plotar para cada tipo de canal separadamente
for ch_type in available_ch_types:
    print(f"Ploting {ch_type.upper()}...")
    try:
        mne.viz.plot_compare_evokeds(
            evoked_dict,
            picks=ch_type,
            legend="upper left",
            show_sensors="upper right",
            title=f"Comparison Between Categories - {ch_type.upper()}"
        )
    except Exception as e:
        print(f"Error ploting {ch_type}: {e}")








#%% beginning of cluster based permutation test

import numpy as np
import mne
import scipy.stats
from mne.stats import spatio_temporal_cluster_test
from mne.channels import find_ch_adjacency

# trabalhar apenas com magnetómetros
epochs_mag = epochs.copy().pick("mag")

# condições experimentais (categorias reais!)
conditions = ["faces", "objects", "fonts", "false_fonts"]

#%% Teste 
# obter todos os nomes de eventos existentes
all_event_names = list(epochs_mag.event_id.keys())

category_map = {
    "faces":       [e for e in all_event_names if e.startswith("faces_")],
    "objects":     [e for e in all_event_names if e.startswith("objects_")],
    "fonts":       [e for e in all_event_names if e.startswith("fonts_")],
    "false_fonts": [e for e in all_event_names if e.startswith("false_fonts_")]
}

# sanity check
for k, v in category_map.items():
    print(k, "→", len(v), "event IDs")


#%% teste 2 

# número mínimo de trials entre categorias
n_trials = min(
    sum(len(epochs_mag[e]) for e in category_map[cat])
    for cat in category_map
)

print("Trials por categoria (equalizados):", n_trials)



#%% Equalizar número de epochs e construir X
  
X = []

for cat, labels in category_map.items():
    epochs_cat = epochs_mag[labels]
    data = epochs_cat.get_data()[:n_trials]   # (epochs, channels, times)
    data = np.transpose(data, (0, 2, 1))      # (epochs, times, channels)
    X.append(data)

print("Shape por condição:", [x.shape for x in X])


#%% Definir adjencência espacial

adjacency, ch_names = find_ch_adjacency(
    epochs_mag.info,
    ch_type="mag"
)


#%% Definir threshold estatístico (F-test)

alpha_cluster = 0.001
n_conditions = len(X)
n_observations = n_trials

dfn = n_conditions - 1
dfd = n_observations - n_conditions

f_threshold = scipy.stats.f.ppf(
    1 - alpha_cluster,
    dfn=dfn,
    dfd=dfd
)

print("F-threshold:", f_threshold)


#%% Correr o cluster-based permutation test

F_obs, clusters, p_values, _ = spatio_temporal_cluster_test(
    X,
    threshold=f_threshold,
    n_permutations=1000,
    tail=1,  # F-test → apenas cauda superior
    adjacency=adjacency,
    n_jobs=None
)


#%% Identificar clusters significativos

alpha_cluster_accept = 0.05
good_cluster_inds = np.where(p_values < alpha_cluster_accept)[0]

print(f"Clusters significativos: {len(good_cluster_inds)}")


#%% Visualização simples de um cluster 
import matplotlib.pyplot as plt

if len(good_cluster_inds) > 0:
    clu = good_cluster_inds[0]
    time_inds, space_inds = clusters[clu]
    time_inds = np.unique(time_inds)
    ch_inds = np.unique(space_inds)

    f_map = F_obs[time_inds].mean(axis=0)

    evoked_f = mne.EvokedArray(
        f_map[:, np.newaxis],
        epochs_mag.info,
        tmin=0
    )

    evoked_f.plot_topomap(
        times=0,
        cmap="Reds",
        show=True
    )





#%% FAZER EVOCKED RESPONSES PARA CADA CATEGORIA

# escolher o primeiro cluster significativo (podes mudar o índice)
clu_idx = good_cluster_inds[0]

time_inds, space_inds = clusters[clu_idx]
time_inds = np.unique(time_inds)
ch_inds = np.unique(space_inds)

tmin_clu = epochs.times[time_inds[0]]
tmax_clu = epochs.times[time_inds[-1]]

print(f"Cluster window: {tmin_clu:.3f} – {tmax_clu:.3f} s")
print(f"N sensores no cluster: {len(ch_inds)}")

evoked_dict = {}

for cat, labels in category_map.items():
    if len(labels) > 0:
        evoked_dict[cat] = epochs_mag[labels].average()

evoked_clu = {
    cat: ev.copy().crop(tmin=tmin_clu, tmax=tmax_clu)
    for cat, ev in evoked_dict.items()
}

import mne

print("Plotting evokeds restricted to cluster sensors and time window")

mne.viz.plot_compare_evokeds(
    evoked_clu,
    picks=ch_inds,              # sensores do cluster
    combine="mean",             # média dos sensores do cluster
    legend="upper right",
    show_sensors=False,
    title=(
        f"Evoked responses in cluster "
        f"({tmin_clu*1000:.0f}–{tmax_clu*1000:.0f} ms)"
    )
)

import numpy as np

f_map = F_obs[time_inds].mean(axis=0)

evoked_f = mne.EvokedArray(
    f_map[:, np.newaxis],
    epochs_mag.info,
    tmin=0
)

mask = np.zeros((len(f_map), 1), dtype=bool)
mask[ch_inds, :] = True

evoked_f.plot_topomap(
    times=0,
    mask=mask,
    cmap="Reds",
    show=True,
    mask_params=dict(markersize=8)
)











#%% Começo da Decoding analtysis

#Preparar X e y

import numpy as np

# só magnetómetros
epochs_mag = epochs.copy().pick("mag")

# Todas as labels de evento e as tuas categorias reais
all_event_names = list(epochs_mag.event_id.keys())

# criar vetor y com categorias simplificadas
labels = []
for ev in epochs_mag.events:
    # evento original
    code = ev[2]
    # descobrir nome do evento
    name = [k for k,v in epochs_mag.event_id.items() if v == code][0]
    # simplificar (ex: "faces_1" -> "faces")
    cat = name.split("_")[0]
    labels.append(cat)

labels = np.array(labels)

# datão de X
X = epochs_mag.get_data()  # shape: n_epochs, n_channels, n_times
print("X:", X.shape, "y:", np.unique(labels))


#%% Mudar y para números (scikit-learn precisa disso)
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y = le.fit_transform(labels)  # e.g., faces->0, objects->1,...
print("classes:", le.classes_)


#%% Configurar o decodificador
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from mne.decoding import SlidingEstimator

# classificador linear com normalização
clf = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000, solver="liblinear")
)

time_decod = SlidingEstimator(
    clf,
    scoring="accuracy",  # acurácia em cada tempo
    n_jobs=1
)


#%%  Correr o decoding

time_decod.fit(X, y)
scores = time_decod.score(X, y)



#%% Visualizar resultados do decoding
import matplotlib.pyplot as plt

times = epochs_mag.times
plt.plot(times, scores, label="Decoding accuracy")
plt.axhline(1.0/len(le.classes_), color="k", linestyle="--",
            label="chance")
plt.xlabel("Time (s)")
plt.ylabel("Accuracy")
plt.legend()
plt.show()










#%% cOISAS EXTRAS -- DECODING BINARIO E TEMP GEN 
epochs_mag

#%% BINARY DECODING ALL PAIRWISE CONTRASTS
#PREPARAÇÃO COMUM (X e labels)

import numpy as np
from sklearn.preprocessing import LabelEncoder

# dados
X = epochs_mag.get_data()   # (n_epochs, n_channels, n_times)

# labels categóricas (faces / objects / fonts / false_fonts)
labels = []
for ev in epochs_mag.events:
    code = ev[2]
    name = [k for k, v in epochs_mag.event_id.items() if v == code][0]
    labels.append(name.split("_")[0])

labels = np.array(labels)

le = LabelEncoder()
y_all = le.fit_transform(labels)

print("Classes:", le.classes_)

"""DECODING BINÁRIO → TODOS OS CONTRASTES
faces vs objects
faces vs fonts
faces vs false_fonts
objects vs fonts
objects vs false_fonts
fonts vs false_fonts
"""

#Classificador (igual para todos)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from mne.decoding import SlidingEstimator, cross_val_multiscore

clf = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000, solver="liblinear")
)

time_decod = SlidingEstimator(
    clf,
    scoring="accuracy",
    n_jobs=1
)

#Loop pelos contrastes
import itertools
import matplotlib.pyplot as plt

conditions = le.classes_
times = epochs_mag.times

decoding_results = {}

for cond_a, cond_b in itertools.combinations(conditions, 2):

    print(f"Decoding: {cond_a} vs {cond_b}")

    mask = np.logical_or(labels == cond_a, labels == cond_b)
    X_pair = X[mask]
    y_pair = labels[mask]

    y_pair = LabelEncoder().fit_transform(y_pair)

    scores = cross_val_multiscore(
        time_decod,
        X_pair,
        y_pair,
        cv=5,
        n_jobs=1
    )

    decoding_results[(cond_a, cond_b)] = scores.mean(axis=0)

    plt.plot(times, scores.mean(axis=0), label=f"{cond_a} vs {cond_b}")

#PLOT FINAL 
plt.axhline(0.5, color="k", linestyle="--", label="chance")
plt.xlabel("Time (s)")
plt.ylabel("Decoding accuracy")
plt.legend()
plt.title("Pairwise category decoding (MEG)")
plt.show()




#%% TEMPORAL GENERALIZATION DECODING

#GeneralizingEstimator
from mne.decoding import GeneralizingEstimator

gen_clf = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000, solver="liblinear")
)

time_gen = GeneralizingEstimator(
    gen_clf,
    scoring="accuracy",
    n_jobs=1
)

#Exemplo: faces vs fonts
cond_a, cond_b = "faces", "fonts"

mask = np.logical_or(labels == cond_a, labels == cond_b)
X_pair = X[mask]
y_pair = LabelEncoder().fit_transform(labels[mask])

scores_gen = cross_val_multiscore(
    time_gen,
    X_pair,
    y_pair,
    cv=5,
    n_jobs=1
)

scores_gen = scores_gen.mean(axis=0)  # (train_time, test_time)

#Plot matriz de generalização
plt.figure(figsize=(6, 5))
plt.imshow(
    scores_gen,
    origin="lower",
    aspect="auto",
    extent=[times[0], times[-1], times[0], times[-1]],
    vmin=0.5,
    vmax=scores_gen.max()
)
plt.colorbar(label="Accuracy")
plt.xlabel("Test time (s)")
plt.ylabel("Train time (s)")
plt.title("Temporal generalization: faces vs fonts")
plt.show()

# %%
