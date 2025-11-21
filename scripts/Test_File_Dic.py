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



#%%
#Calling the file
base_dir = r"C:\Users\tomas\Desktop\a pasta das aulas\Mestrado\The Last Year aka Tese\A pasta do codigo organizado"

#insert the file 
sample_data_raw_file = fr"{base_dir}\Participantes\Random peps\CA112_MEEG_1\scans\DurR3_DurR3\FIF\CA112_MEEG_1_DurR3.fif"

cal_file = fr"{base_dir}\Participantes\Random peps\CA112_MEEG_1\resources\METADATA\calibration_crosstalk_coreg\sub-CA112_ses-1_acq-calibration_meg.dat"
ct_file  = fr"{base_dir}\Participantes\Random peps\CA112_MEEG_1\resources\METADATA\calibration_crosstalk_coreg\sub-CA112_ses-1_acq-crosstalk_meg.fif"

raw = mne.io.read_raw_fif(sample_data_raw_file, preload=True)


# %%
#Detetar bad channels automaticamente (MEG)

raw.info['bads'] = []  # começa sem nada marcado

auto_noisy, auto_flat, auto_scores = find_bad_channels_maxwell(
    raw.copy(),
    calibration=cal_file,
    cross_talk=ct_file,
    return_scores=True,
    verbose=True,
)

print("Noisy channels:", auto_noisy)
print("Flat channels :", auto_flat)

raw.info['bads'].extend(auto_noisy + auto_flat)

raw.info['bads'].append('MEG0131')
#raw.info['bads'].append('MEGXXXX')  #para meter mais se quiser


#So para chekar é lixo depois 

print("Bads ANTES do SSS:", raw.info['bads'])

# guardar lista para reaplicar depois
bads_for_sss = raw.info['bads'].copy()



# 4) Corrigir tipos de bobina dos magnetómetros (obrigatório em Elekta)
raw.fix_mag_coil_types()


# Maxwell filter – SSS (sem tSSS, igual ao Cogitate)

raw_sss = maxwell_filter(
    raw,
    calibration=cal_file,
    cross_talk=ct_file,
    st_duration=None,      # SSS simples  --- desligar pra fazer tSSS
    #st_duration=10.0,      # tSSS com janela de 10 s --- se quiseres ativar tSSS liga estes 2 
    #st_correlation=0.98,   # default razoável
    origin='auto',
    coord_frame='head',
    verbose=True,
)

# re-meter a lista de bads no Raw filtrado
raw_sss.info['bads'] = bads_for_sss

print("Bads DEPOIS do SSS:", raw_sss.info['bads'])

raw = raw_sss




#%%
#filtragem 
raw.notch_filter(freqs=[50, 100, 150, 200, 250, 300], phase='zero', fir_design='firwin')  # rede elétrica



#%%

# Caminho para treinar ICA: 1–80 Hz @ 250 Hz
raw_ica = raw.copy()

raw_ica.filter(1., 80., fir_design='firwin')
raw_ica.resample(250.)

print(raw_ica)

# ICA para MEG
ica_meg = ICA(
    n_components=0.99,
    method='fastica',      # 'picard' se tiveres instalado
    random_state=97,
    max_iter='auto',
)

print(">>> Fitting ICA (MEG)...")
ica_meg.fit(raw_ica, picks='meg')

# Detectar componentes EOG/ECG
eog_inds_meg, eog_scores_meg = ica_meg.find_bads_eog(
    raw_ica,
    ch_name=['EOG001', 'EOG002']
)
ecg_inds_meg, ecg_scores_meg = ica_meg.find_bads_ecg(
    raw_ica,
    ch_name='ECG003'
)

ica_meg.exclude = sorted(set(eog_inds_meg + ecg_inds_meg))
print("MEG ICs marcadas para exclusão:", ica_meg.exclude)

# Inspeção manual (OLHA MESMO PARA ISTO!)
ica_meg.plot_sources(raw_ica, picks=ica_meg.exclude)
ica_meg.plot_components(picks=ica_meg.exclude)
plt.show()

# Se vires que alguma componente não é EOG/ECG → tira dos exclude
# Exemplo:
# ica_meg.exclude = [0, 3, 7]

# Aplicar ICA MEG ao sinal de análise (0.1–330)
raw_meg_clean = ica_meg.apply(raw.copy())
print(raw_meg_clean)



# ICA para EEG
ica_eeg = ICA(
    n_components=0.99,
    method='fastica',
    random_state=97,
    max_iter='auto',
)

print(">>> Fitting ICA (EEG)...")
ica_eeg.fit(raw_ica, picks='eeg')

# Detectar componentes EOG/ECG vistas no EEG
eog_inds_eeg, eog_scores_eeg = ica_eeg.find_bads_eog(
    raw_ica,
    ch_name=['EOG001', 'EOG002']
)
ecg_inds_eeg, ecg_scores_eeg = ica_eeg.find_bads_ecg(
    raw_ica,
    ch_name='ECG003'
)

ica_eeg.exclude = sorted(set(eog_inds_eeg + ecg_inds_eeg))
print("EEG ICs marcadas para exclusão:", ica_eeg.exclude)

# Inspeção manual só se houver ICs marcadas
if len(ica_eeg.exclude) > 0:
    ica_eeg.plot_sources(raw_ica, picks=ica_eeg.exclude)
    ica_eeg.plot_components(picks=ica_eeg.exclude)
    plt.show()
else:
    print("Nenhuma IC EEG foi automaticamente marcada como EOG/ECG.")
    # Se quiseres, podes inspecionar manualmente:
    # ica_eeg.plot_components()  # sem picks, para veres todas

# Ajusta se for preciso:
# ica_eeg.exclude = [1, 5, 9]

# Aplicar ICA EEG em cima do MEG-clean
raw_clean = ica_eeg.apply(raw_meg_clean.copy())
print(raw_clean)

raw = raw_clean 





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
