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
#Meter a pasta do sujeito
base_dir = r"C:\Users\tomas\Desktop\a pasta das aulas\Mestrado\The Last Year aka Tese\A pasta do codigo organizado\Participantes\Random peps"

# --- caminhos dos ficheiros ---
file_paths = [
    fr"{base_dir}\CA112_MEEG_1\scans\DurR1_DurR1\FIF\CA112_MEEG_1_DurR1.fif",
    fr"{base_dir}\CA112_MEEG_1\scans\DurR2_DurR2\FIF\CA112_MEEG_1_DurR2.fif",
    fr"{base_dir}\CA112_MEEG_1\scans\DurR3_DurR3\FIF\CA112_MEEG_1_DurR3.fif",
    fr"{base_dir}\CA112_MEEG_1\scans\DurR4_DurR4\FIF\CA112_MEEG_1_DurR4.fif",
    fr"{base_dir}\CA112_MEEG_1\scans\DurR5_DurR5\FIF\CA112_MEEG_1_DurR5.fif"
]
names = ["dur1", "dur2", "dur3", "dur4", "dur5"]


# ficheiros de calibração e cross-talk
cal_file = file_Path = fr"{base_dir}\CA112_MEEG_1\resources\METADATA\calibration_crosstalk_coreg\sub-CA112_ses-1_acq-calibration_meg.dat"
ct_file = file_path = fr"{base_dir}\CA112_MEEG_1\resources\METADATA\calibration_crosstalk_coreg\sub-CA112_ses-1_acq-crosstalk_meg.fif"

raws = [mne.io.read_raw_fif(str(p), preload=True) for p in file_paths]


#%%"
# Detetar bads por run e juntar tudo
all_bads = set()
# all_bads.add('MEG0131')  # se souberes de algum canal mau de antemão, mete aqui

for i, raw in enumerate(raws, start=1):
    raw.info['bads'] = []  # começa sem nada marcado neste run

    auto_noisy, auto_flat, auto_scores = find_bad_channels_maxwell(
        raw.copy(),
        calibration=cal_file,
        cross_talk=ct_file,
        return_scores=True,
        verbose=True,
    )

    print(f"Run {i} - Noisy channels:", auto_noisy)
    print(f"Run {i} - Flat channels :", auto_flat)

    all_bads.update(auto_noisy + auto_flat)

print("Bads combinados de todos os runs:", all_bads)
bads_for_sss = list(all_bads)


#%%
# Aplicar bads + corrigir coil types em todos os runs
for raw in raws:
    raw.info['bads'] = bads_for_sss.copy()
    raw.fix_mag_coil_types()



#%% ---------------------------------------------------------
# 4) Maxwell filter run-a-run com realinhamento da cabeça
# ---------------------------------------------------------
dest = raws[0].info['dev_head_t']  # posição de cabeça de referência

raws_sss = []
for i, raw in enumerate(raws, start=1):
    print(f">>> Maxwell filter no run {i}...")
    raw_sss = maxwell_filter(
        raw,
        calibration=cal_file,
        cross_talk=ct_file,
        st_duration=None,      # SSS simples (sem tSSS)
        # st_duration=10.0,    # ativa estes 2 se quiseres tSSS
        # st_correlation=0.98,
        origin='auto',
        coord_frame='head',
        destination=dest,      # realinhar todos os runs à mesma posição
        verbose=True,
    )
    raws_sss.append(raw_sss)

print("Bads DEPOIS do SSS (run 1):", raws_sss[0].info['bads'])



#%% ---------------------------------------------------------
# 5) ICA VERSÃO ÓSCAR – treinar em dados LEVES
# ---------------------------------------------------------


raws_for_ica = []

for i, raw_sss in enumerate(raws_sss, start=1):
    print(f">>> Preparar run {i} para treino de ICA...")
    r = raw_sss.copy()
    r.pick(['meg', 'eog', 'ecg'])          # só o que interessa para artefactos
    r.filter(1., 80., fir_design='firwin')
    r.resample(250., npad="auto")         # poupar RAM
    raws_for_ica.append(r)

raw_ica = mne.concatenate_raws(raws_for_ica)
print(raw_ica)

ica_meg = ICA(
    n_components=0.99,
    method='fastica',   # 'picard' se tiveres instalado
    random_state=97,
    max_iter='auto',
)

print(">>> Fitting ICA (MEG) na versão leve...")
ica_meg.fit(raw_ica, picks='meg')

# Detectar componentes EOG/ECG
eog_inds_meg, eog_scores_meg = ica_meg.find_bads_eog(
    raw_ica,
    ch_name=['EOG001', 'EOG002']   # confirma se estes nomes batem certo
)
ecg_inds_meg, ecg_scores_meg = ica_meg.find_bads_ecg(
    raw_ica,
    ch_name='ECG003'
)

ica_meg.exclude = sorted(set(eog_inds_meg + ecg_inds_meg))
print("MEG ICs marcadas para exclusão:", ica_meg.exclude)

# Inspeção manual das ICs marcadas
ica_meg.plot_sources(raw_ica, picks=ica_meg.exclude)
ica_meg.plot_components(picks=ica_meg.exclude)
plt.show()

# Se vires que alguma componente não é artefacto, ajusta aqui:
# ica_meg.exclude = [0, 3, 7]

# Libertar a versão leve da memória — já não é precisa
del raw_ica
del raws_for_ica

# ---------------------------------------------------------
# 6) Aplicar ICA aos dados ORIGINAIS (full band), run-a-run
# ---------------------------------------------------------
for i in range(len(raws_sss)):
    print(f">>> Aplicar ICA ao run {i+1} (dados full-band)...")
    raws_sss[i] = ica_meg.apply(raws_sss[i])

# ---------------------------------------------------------
# 7) Concatenar todos os runs já limpos
# ---------------------------------------------------------
raw_all = mne.concatenate_raws(raws_sss)

# ---------------------------------------------------------
# 8) Criar anotações para cada duração (dur1–dur5)
# ---------------------------------------------------------
start = 0.0
for i, raw_sss in enumerate(raws_sss, start=1):
    dur = raw_sss.times[-1]
    raw_all.annotations.append(
        onset=start,
        duration=dur,
        description=f"dur{i}"
    )
    start += dur

# ---------------------------------------------------------
# 9) Notch filter no contínuo final (opcional, ajusta freqs se quiseres)
# ---------------------------------------------------------
raw_all.notch_filter(freqs=[50, 100, 150, 200, 250, 300], phase='zero', fir_design='firwin')

# Este será o Raw contínuo que usas no resto do pipeline
raw = raw_all
print(raw)
print(raw.info)



#%%
#prints EEG, Gradiometers and Magnetometers power function
raw.compute_psd(fmax=50).plot(picks="data", exclude="bads", amplitude=False)
#prints all channels signals
raw.plot(duration=5, n_channels=30)
plt.show()



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
        202: "nontarget",
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



reject_criteria_meg = dict(
    mag=4000e-15,
    grad=4000e-13,
    # sem eeg aqui
)


epochs = mne.Epochs(
    raw,
    events,
    event_id=full_event_id,
    tmin=-0.2,
    tmax=2.0,
    baseline=(-0.2, 0),
    reject=reject_criteria_meg,  # 👈 trocado
    picks='meg',
    decim=2,
    preload=False
)

epochs.drop_bad()  # isto marca e remove os trials maus, sem enfiar 38 GB em RAM
print(f"Número de epochs bons depois de rejeição: {len(epochs.events)}")

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
            evoked_stimulus,   #Mudei
            picks=ch_type,
            legend="upper left",
            show_sensors="upper right",
            title=f"Comparison Between Stimulus Categories - {ch_type.upper()}",
            ci=0.68
        )
    except Exception as e:
        print(f"Error ploting {ch_type}: {e}")

# %%
