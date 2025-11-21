
#%%
#Import Room
import mne
import numpy as np
import os
import matplotlib 
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from mne.preprocessing import find_bad_channels_maxwell, maxwell_filter, ICA


# %%
# STI - Event Categorization
# 1) Definir o canal de triggers 
stim_channel = "STI101"

# 2) Extrair eventos (ajustando min_duration para evitar erros) 
events = mne.find_events(
    raw,  
    stim_channel=stim_channel,
    shortest_event=1,  
    min_duration=0.001,  
    verbose=True
)

# 3) Função para categorizar estímulo (faces, objects, fonts, false_fonts)
def categorize_stimulus(event_id):
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

# 3b) Funções para categorizar orientação, duração e relevância

def categorize_orientation(event_id):
    mapping = {
        101: "center",
        102: "left",
        103: "right",
    }
    return mapping.get(event_id, None)

def categorize_duration(event_id):
    mapping = {
        151: "dur_500ms",
        152: "dur_1000ms",
        153: "dur_1500ms",
    }
    return mapping.get(event_id, None)

def categorize_relevance(event_id):
    mapping = {
        201: "target",
        202: "relevant",
        203: "irrelevant",
    }
    return mapping.get(event_id, None)

# 4) Criar dicionário de estímulo com todos os IDs de cada categoria
stimulus_dict = {}
for e in np.unique(events[:, 2]):
    label = categorize_stimulus(e)
    if label is not None:
        if label not in stimulus_dict:
            stimulus_dict[label] = []
        stimulus_dict[label].append(e)

print("=== Stimulus codes ===")
for cat, ids in stimulus_dict.items():
    print(f"{cat}: {ids}")

# 4b) Criar dicionários para orientação, duração e relevância

orientation_dict = {}
duration_dict = {}
relevance_dict = {}

for e in np.unique(events[:, 2]):
    # orientação
    label = categorize_orientation(e)
    if label is not None:
        if label not in orientation_dict:
            orientation_dict[label] = []
        orientation_dict[label].append(e)

    # duração
    label = categorize_duration(e)
    if label is not None:
        if label not in duration_dict:
            duration_dict[label] = []
        duration_dict[label].append(e)

    # relevância
    label = categorize_relevance(e)
    if label is not None:
        if label not in relevance_dict:
            relevance_dict[label] = []
        relevance_dict[label].append(e)

print("=== Orientation codes ===")
for cat, ids in orientation_dict.items():
    print(f"{cat}: {ids}")

print("=== Duration codes ===")
for cat, ids in duration_dict.items():
    print(f"{cat}: {ids}")

print("=== Relevance codes ===")
for cat, ids in relevance_dict.items():
    print(f"{cat}: {ids}")

# 5) Criar dicionário para plot dos estímulos (cada ID com label única)
stimulus_plot_dict = {}
for cat, ids in stimulus_dict.items():
    for e_id in ids:
        stimulus_plot_dict[f"{cat}_{e_id}"] = e_id

# 5b) Plot dicts para orientação, duração e relevância
orientation_plot_dict = {}
for cat, ids in orientation_dict.items():
    for e_id in ids:
        orientation_plot_dict[f"{cat}_{e_id}"] = e_id

duration_plot_dict = {}
for cat, ids in duration_dict.items():
    for e_id in ids:
        duration_plot_dict[f"{cat}_{e_id}"] = e_id

relevance_plot_dict = {}
for cat, ids in relevance_dict.items():
    for e_id in ids:
        relevance_plot_dict[f"{cat}_{e_id}"] = e_id

# 6) Plotar eventos (aqui só com os estímulos principais; se quiseres, podes usar full_event_id)
fig = mne.viz.plot_events(
    events,
    event_id=stimulus_plot_dict,
    sfreq=raw.info["sfreq"],
    first_samp=raw.first_samp,
)

# 7) Critérios de rejeição
reject_criteria = dict(
    mag=4000e-15,
    grad=4000e-13,
    eeg=400e-6,  # O critério de rejeição do EEG com MEG é diferente de um normal 
)

# 8) Criar dicionário "flat" com TODOS os eventos relevantes
full_event_id = {}
full_event_id.update(stimulus_plot_dict)
full_event_id.update(orientation_plot_dict)
full_event_id.update(duration_plot_dict)
full_event_id.update(relevance_plot_dict)

# --- 9) Criar epochs com todos os eventos ---
epochs = mne.Epochs(
    raw,
    events,
    event_id=full_event_id,
    tmin=-0.2,
    tmax=2.0,
    baseline=(-0.2, 0),   # ou (None, 0)
    reject=reject_criteria,
    preload=True
)

# 10) Criar evoked responses para cada categoria de estímulo (faces/objects/fonts/false_fonts)
evoked_stimulus = {}
for category in stimulus_dict.keys():
    category_labels = [
        f"{category}_{i}" 
        for i in stimulus_dict[category] 
        if f"{category}_{i}" in epochs.event_id
    ]
    if category_labels and len(epochs[category_labels]) > 0:
        evoked_stimulus[category] = epochs[category_labels].average()

# (Opcional) Evokeds por orientação
evoked_orientation = {}
for category in orientation_dict.keys():
    category_labels = [
        f"{category}_{i}"
        for i in orientation_dict[category]
        if f"{category}_{i}" in epochs.event_id
    ]
    if category_labels and len(epochs[category_labels]) > 0:
        evoked_orientation[category] = epochs[category_labels].average()

# (Opcional) Evokeds por duração
evoked_duration = {}
for category in duration_dict.keys():
    category_labels = [
        f"{category}_{i}"
        for i in duration_dict[category]
        if f"{category}_{i}" in epochs.event_id
    ]
    if category_labels and len(epochs[category_labels]) > 0:
        evoked_duration[category] = epochs[category_labels].average()

# (Opcional) Evokeds por relevância
evoked_relevance = {}
for category in relevance_dict.keys():
    category_labels = [
        f"{category}_{i}"
        for i in relevance_dict[category]
        if f"{category}_{i}" in epochs.event_id
    ]
    if category_labels and len(epochs[category_labels]) > 0:
        evoked_relevance[category] = epochs[category_labels].average()

# 11) Verificar que tipos de canais MEG temos disponíveis (usando estímulos)
first_evoked = evoked_stimulus[list(evoked_stimulus.keys())[0]]
available_ch_types = set(first_evoked.get_channel_types())
print(f"Types of Channels Available: {available_ch_types}")

# 12) Plotar comparação entre categorias de estímulo para cada tipo de canal
for ch_type in available_ch_types:
    print(f"Ploting {ch_type.upper()}...")
    try:
        mne.viz.plot_compare_evokeds(
            evoked_relevance,   #Mudei
            picks=ch_type,
            legend="upper left",
            show_sensors="upper right",
            title=f"Comparison Between Stimulus Categories - {ch_type.upper()}",
            ci=0.68
        )
    except Exception as e:
        print(f"Error ploting {ch_type}: {e}")










# %%
