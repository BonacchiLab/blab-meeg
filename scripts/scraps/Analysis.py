#Analises


#%%
#Import Room
import mne
import os
import matplotlib 
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt

#%%
#FALTA CHAMAR TODOS OS DICIONARIOS 

#event_dics.py
from event_dics import flat_event_id
print(flat_event_id)

#%%
#chamar o ficheiro 
sample_data_raw_file = r"C:\Users\tomas\Desktop\a pasta das aulas\Mestrado\The Last Year aka Tese\A pasta do codigo organizado\Participantes\Random peps\CA112_MEEG_1\My Results aka Processed files\CA112_MEEG_1_Dur_1_PreProc.fif"
raw = mne.io.read_raw_fif(sample_data_raw_file, preload=True)


#%%

# --- 9) Criar epochs com todos os eventos ---
epochs = mne.Epochs(
    raw,
    events,
    event_id=flat_event_id,
    tmin=-0.2,
    tmax=1,
    reject=reject_criteria,
    preload=True
)

#%%
#9) Criar epochs com todos os eventos
epochs = mne.Epochs(
    raw,
    events,
    event_id=flat_event_id,
    tmin=-0.2,
    tmax=1,
    reject=reject_criteria,
    preload=True
)

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
print(f"Tipos de canais disponíveis: {available_ch_types}")

# Plotar para cada tipo de canal separadamente
for ch_type in available_ch_types:
    print(f"Plotando {ch_type.upper()}...")
    try:
        mne.viz.plot_compare_evokeds(
            evoked_dict,
            picks=ch_type,
            legend="upper left",
            show_sensors="upper right",
            title=f"Comparação entre Categorias - {ch_type.upper()}"
        )
    except Exception as e:
        print(f"Erro ao plotar {ch_type}: {e}")



#%%

faces_evoked = evoked_dict["faces"] 
objects_evoked = evoked_dict["objects"] 
fonts_evoked = evoked_dict["fonts"]




#%%
# --- 1) Calcular média de epochs para a categoria "faces" ---
faces_epochs = epochs["faces"]  # assume que já tens category_epochs
faces_evoked = faces_epochs.average()

# --- 2) Plotar gráfico conjunto (ERPs + topografia) ---
faces_evoked.plot_joint(picks="eeg")

# --- 3) Plotar mapas topográficos em tempos específicos ---
# ajusta os tempos de acordo com o que queres visualizar
faces_evoked.plot_topomap(times=[0.0, 0.08, 0.1, 0.12, 0.2], ch_type="eeg")



#%%

# --- 1) Calcular médias de duas categorias ---
faces_evoked = epochs["faces"].average()
fonts_evoked = epochs["fonts"].average()

# --- 2) Combinar as evoked (diferença faces - objects) ---
evoked_diff = mne.combine_evoked([faces_evoked, fonts_evoked], weights=[1, -1])

# --- 3) Selecionar apenas canais magnetômetros e plotar topografia ---
evoked_diff.pick(picks="mag").plot_topo(color="r", legend=False)

evoked_diff_eeg = mne.combine_evoked([faces_evoked, fonts_evoked], weights=[1, -1])
evoked_diff_eeg.plot_topomap(times=[0.170, 0.250, 0.400], ch_type='eeg', title='Faces - Fonts (EEG)')



#%%



#%%



#%%




#%%




#%%




#%%




#%%







#%%
# Time-frequency analysis -> Precisamos de mais um codigo para definir um eletrodo de interesse e o mostrar 
frequencies = np.arange(7, 30, 3)
n_cycles = 2

from mne.time_frequency import tfr_morlet

power = tfr_morlet(
    epochs,
    freqs=frequencies,  
    n_cycles=n_cycles,
    return_itc=False,
    decim=3,
    average=True
)

power.plot(picks=["MEG0121"])



#%%



#%%


#%%



