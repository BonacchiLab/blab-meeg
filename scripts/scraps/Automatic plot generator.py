#Automatic plot generator 

#%% Import Room
import mne 
import numpy as np


#%% Load data
raw = mne.io.read_raw_fif("C:\\Users\\tomas\\Desktop\\MEG_outputs\\raw_clean-ica.fif", preload=True)



#%% Dictionary mapping experimental conditions to event IDs

# 1) Defining the stimulation channel
stim_channel = "STI101"

# 2) Extracting events from raw data 
events = mne.find_events(
    raw,  
    stim_channel=stim_channel,
    shortest_event=1,  
    min_duration=0.001,  
    verbose=True
)

#3) Categorizing event IDs 

#3a) Category (faces, objects, fonts, false_fonts)
def category(event_id):
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

#falta  oo big one para meter tdo na mesma caixa???



# 3b) Orientation (center, left, right)
def orientation(event_id):
    mapping = {
        101: "center",
        102: "left",
        103: "right",
    }
    return mapping.get(event_id, None)

# 3c) Duration (500ms, 1000ms, 1500ms)
def duration(event_id):
    mapping = {
        151: "dur_500ms",
        152: "dur_1000ms",
        153: "dur_1500ms",
    }
    return mapping.get(event_id, None)

# 3d) Relevance (target, relevant, irrelevant)
def relevance(event_id):
    mapping = {
        201: "target",
        202: "relevant",
        203: "irrelevant",
    }
    return mapping.get(event_id, None)



# 4) Create dictionary for stimulus categories

# 4a) Category Special dictionary 
stimulus_dict = {}
for e in np.unique(events[:, 2]):
    label = category(e)
    if label is not None:
        if label not in stimulus_dict:
            stimulus_dict[label] = []
        stimulus_dict[label].append(e)

print("=== Stimulus codes ===")
for cat, ids in stimulus_dict.items():
    print(f"{cat}: {ids}")

#TODO: seclahar aqui é que vem o outro dicionário


# 4b) Orientation, Duration and Relevance dictionaries
orientation_dict = {}
duration_dict = {}
relevance_dict = {}
for e in np.unique(events[:, 2]):
    # Orientation
    label = orientation(e)
    if label is not None:
        if label not in orientation_dict:
            orientation_dict[label] = []
        orientation_dict[label].append(e)

    # Duration
    label = duration(e)
    if label is not None:
        if label not in duration_dict:
            duration_dict[label] = []
        duration_dict[label].append(e)
    
    # relevância
    label = relevance(e)
    if label is not None:
        if label not in relevance_dict:
            relevance_dict[label] = []
        relevance_dict[label].append(e)


# 4c) Print orientation, duration and relevance dictionaries
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
    event_id=relevance_plot_dict,
    sfreq=raw.info["sfreq"],
    first_samp=raw.first_samp,
)

# 7) Critérios de rejeição
reject_criteria = dict(
    mag=4500e-15,
    grad=4500e-13,
    eeg=450e-6,  # O critério de rejeição do EEG com MEG é diferente de um normal 
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
    tmin=-0.9,
    tmax=1.5,
    baseline=(-0.9, 0),   # ou (None, 0)
    reject=reject_criteria,
    preload= True 
)

epochs.plot_drop_log()


#%%
print(events[:100])  # show the first 5

#%%
fig = mne.viz.plot_events(
    events,
    event_id=full_event_id,
    sfreq=raw.info["sfreq"],
    first_samp=raw.first_samp,
)

#%%
evoked_dict = {}

for category in full_event_id.keys():
    if category in epochs.event_id:
        evoked_dict[category] = epochs[category].average()
    else:
        print(f"WARNING: {category} not found in epochs.event_id")




#%%
plot_dict = {}
for cat, ids in full_event_id.items():
    for e_id in ids:
        plot_dict[f"{cat}_{e_id}"] = e_id

#%%
 print("Epochs event_id keys:")
print(list(epochs.event_id.keys()))

print("\nCategorias esperadas:")
print(list(full_event_id.keys()))

for category in full_event_id.keys():
    category_labels = [
        key for key in epochs.event_id.keys()
        if key.startswith(f"{category}_")
    ]
    print(f"{category}: {category_labels}")




#%%
# Criar evoked responses para cada categoria
evoked_dict = {}
for category in full_event_id.keys():
    category_labels = [
        key for key in epochs.event_id.keys()
        if key.startswith(f"{category}_")
    ]
    if category_labels:
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


#%%


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
            picks="mag",
            legend="upper left",
            show_sensors="upper right",
            title=f"Comparison Between Stimulus Categories - {ch_type.upper()}"
        )
    except Exception as e:
        print(f"Error ploting {ch_type}: {e}")


# %%
