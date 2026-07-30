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

raw_ica.filter(1., 40., fir_design='firwin')  # changeeeeeeeeeeeeeeeeeeeeeeeeeed
raw_ica.resample(200.)                        # changeeeeeeeeeeeeeeeeeeeeeeeeeed

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
    tmax=0.5,
    baseline=(-0.2, 0),   # ou (None, 0)
    reject=reject_criteria,
    preload=True
)



#%%
print(raw.info['bads'])
#%% 
epochs.plot_drop_log()
 
#%%



# --- parâmetros da TFR ---
freqs = np.arange(2, 31, 1)   # 2–30 Hz
n_cycles = freqs / 2
time_bandwidth = 2.0

# --- apanhar todas as condições que começam por 'faces_' ---
face_keys = [k for k in epochs.event_id.keys() if k.startswith("faces_")]
print("Eventos de faces encontrados:", face_keys)

# selecionar epochs de todas essas condições
face_epochs = epochs[face_keys]

# --- TFR multitaper ---
tfr_faces = mne.time_frequency.tfr_multitaper(
    face_epochs,
    freqs=freqs,
    n_cycles=n_cycles,
    time_bandwidth=time_bandwidth,
    picks='grad',        # muda para 'mag' ou 'eeg' se quiseres
    use_fft=True,
    return_itc=False,
    average=True,
    decim=2,
    n_jobs=-1,
    verbose=True,
)

# --- plot topo ---
tfr_faces.plot_topo(
    tmin=-0.5, tmax=1.0,
    baseline=(-0.5, -0.3),
    mode="percent",
    fig_facecolor='w',
    font_color='k',
    vmin=-1, vmax=1,
    title="TFR of power <30 Hz – faces",
)
plt.show()



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
