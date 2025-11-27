#Fisrt try cluster Test 

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
#ica_meg.plot_sources(raw_ica, picks=ica_meg.exclude)
#ica_meg.plot_components(picks=ica_meg.exclude)
#plt.show()
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
# Caminho da pasta onde queres guardar
save_folder = r"C:\Users\tomas\Desktop\MEG_outputs"

# Nome do ficheiro
filename = "raw_clean-ica.fif"

# Caminho completo
full_path = os.path.join(save_folder, filename)

# Salvar Raw
raw_clean.save(full_path, overwrite=True)

#%%
#Call the cleaned file

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










#%%
builtin_montages = mne.channels.get_builtin_montages(descriptions=True)
for montage_name, montage_description in builtin_montages:
    print(f"{montage_name}: {montage_description}")

#%%

easycap_montage = mne.channels.make_standard_montage("easycap-M1")
print(easycap_montage)

easycap_montage.plot()  # 2D
fig = easycap_montage.plot(kind="3d", show=False)  # 3D
fig = fig.gca().view_init(azim=70, elev=15)  # set view angle for tutorial



#%%

epochs.compute_psd().plot_topomap(ch_type="grad", normalize=False, contours=0)






#%%
# 0) ver que eventos tens disponíveis
print("Event keys disponíveis:", list(epochs.event_id.keys()))

# 1) construir listas de rótulos por prefixo
def keys_with_prefix(prefix):
    keys = [k for k in epochs.event_id.keys() if k.startswith(prefix)]
    if not keys:
        raise ValueError(f"Nenhuma etiqueta encontrada com prefixo '{prefix}'")
    return keys

faces_keys = keys_with_prefix("faces")
fonts_keys = keys_with_prefix("fonts")   # ou "objects" conforme necessário

# 2) calcular médias (evoked) a partir dessas listas
faces_evoked = epochs[faces_keys].average()
fonts_evoked = epochs[fonts_keys].average()

# 3) combinar para obter diferença (faces - fonts)
evoked_diff = mne.combine_evoked([faces_evoked, fonts_evoked], weights=[1, -1])

# 4) seleccionar magnetómetros (sem sobrescrever o objeto original)
evoked_diff_mag = evoked_diff.copy().pick(picks="mag")  # pick modifica in-place, por isso usamos copy()

# 5) plot topo (usa-se plot_topo sem argumentos de cor se provocar erro)
evoked_diff_mag.plot_topo(legend=False, title="Faces - Fonts (mag)")

#%%

faces_evoked.plot()

#%%
faces_evoked.copy().pick("mag").plot_topo()


#%%%
mne.viz.plot_compare_evokeds([faces_evoked, fonts_evoked])

#%%
faces_data = epochs[faces_keys].get_data()   # shape: (n_trials, n_channels, n_times)
print(faces_data.shape)




#%%
import numpy as np
from scipy.stats import ttest_rel
import mne

# parâmetros
tmin_win = -0.2
tmax_win = 0.0
tmin_win2 = 0.0
tmax_win2 = 0.8
alpha = 0.05  # threshold FDR

# --- 0) obter trials das faces (já tens faces_keys) ---
faces_trials = epochs[faces_keys].get_data()  # shape: (n_trials, n_channels, n_times)
times = epochs.times  # eixo temporal compartilhado

# --- 1) índices temporais das janelas ---
idx1 = np.where((times >= tmin_win) & (times < tmax_win))[0]
idx2 = np.where((times >= tmin_win2) & (times <= tmax_win2))[0]

if len(idx1) == 0 or len(idx2) == 0:
    raise ValueError("Uma das janelas temporais não contém pontos de tempo — verifica os limites ou o sampling rate.")

# --- 2) calcular média dentro das janelas por trial e por canal ---
# shape resultante: (n_trials, n_channels)
win1_mean = faces_trials[:, :, idx1].mean(axis=2)
win2_mean = faces_trials[:, :, idx2].mean(axis=2)

# --- 3) fazer t-test emparelhado (paired) sensor-wise ---
# ttest_rel espera (n_obs, n_meas), aqui testamos entre as janelas ao longo dos trials
tvals = np.zeros(win1_mean.shape[1])
pvals = np.ones(win1_mean.shape[1])
for ch in range(win1_mean.shape[1]):
    t, p = ttest_rel(win2_mean[:, ch], win1_mean[:, ch])
    tvals[ch] = t
    pvals[ch] = p

# --- 4) correção FDR (Benjamini-Hochberg) ---
def bh_fdr(pvals, alpha=0.05):
    p = np.asarray(pvals)
    n = p.size
    order = np.argsort(p)
    sorted_p = p[order]
    thresh = (np.arange(1, n+1) * alpha) / n
    below = sorted_p <= thresh
    if not np.any(below):
        return np.zeros(n, dtype=bool)
    max_idx = np.max(np.where(below)[0])
    p_cutoff = sorted_p[max_idx]
    return p <= p_cutoff

significant_mask = bh_fdr(pvals, alpha=alpha)  # boolean array length n_channels
sig_channels = [epochs.ch_names[i] for i, sig in enumerate(significant_mask) if sig]

print(f"Number of significant channels (FDR q={alpha}): {len(sig_channels)}")
print("Significant channel names:", sig_channels)

# --- 5) Plot topo dos t-values apenas para magnetómetros e marcar os significativos ---
# Preparar evoked-like object só para obter info e nomes dos mags na ordem certa
evoked_faces = epochs[faces_keys].average()           # evoked de todas as faces (para info/times)
evoked_mag = evoked_faces.copy().pick_types(meg='mag')  # só magnetómetros

# Precisamos dos tvals e do mask ordenados só para estes canais mags
# Encontrar índices dos canais magnéticos dentro do array global
mag_chs = evoked_mag.ch_names
# mapear cada mag_ch ao índice correspondente no epochs.ch_names
mag_indices_global = [epochs.ch_names.index(ch) for ch in mag_chs]

tvals_mag = tvals[mag_indices_global]
mask_mag = significant_mask[mag_indices_global]

import matplotlib.pyplot as plt
from mne.viz import plot_topomap

# plot topomap dos t-values com máscara para significância

fig, ax = plt.subplots()

im, cm = plot_topomap(
    tvals_mag,          # dados (t-values)
    evoked_mag.info,    # info do objeto evoked
    mask=mask_mag,      # máscara de significância
    mask_params=dict(marker='o', markersize=8, markerfacecolor='w'),
    contours=0,
    axes=ax,
    show=False          # plotamos depois com plt.show()
)

# ajustar cores manualmente
im.set_clim(vmin=np.min(tvals_mag), vmax=np.max(tvals_mag))

plt.colorbar(im, ax=ax)
plt.show()





#%%
import matplotlib.pyplot as plt
from mne.viz import plot_topomap
import numpy as np

fig, ax = plt.subplots(figsize=(10,6)) 

# plot topo
im, cm = plot_topomap(
    tvals_mag,
    evoked_mag.info,
    mask=mask_mag,
    mask_params=dict(marker='o', markersize=8, markerfacecolor='w'),
    contours=0,
    axes=ax,
    show=False
)
im.set_clim(vmin=np.min(tvals_mag), vmax=np.max(tvals_mag))
plt.colorbar(im, ax=ax)

# layout do topo
layout = mne.find_layout(evoked_mag.info)
pos = layout.pos[:, :2]  # pegar x, y
pos = pos - np.mean(pos, axis=0)  # centralizar

# aplicar escala diferente para x e y (esticado horizontalmente)
scale_x = 0.24   # aumenta largura
scale_y = 0.21  # mantém altura
pos[:, 0] = pos[:, 0] / np.max(np.abs(pos[:, 0])) * scale_x
pos[:, 1] = pos[:, 1] / np.max(np.abs(pos[:, 1])) * scale_y

# adicionar nomes dos canais
for i, ch_name in enumerate(evoked_mag.ch_names):
    x, y = pos[i]
    ax.text(x, y, ch_name, ha='center', va='center', fontsize=6)

plt.show()


#%%

#%%
import numpy as np
from scipy.stats import ttest_rel
import mne
import matplotlib.pyplot as plt
from mne.viz import plot_topomap

# --- parâmetros ---
tmin_win = -0.2
tmax_win = 0.0
tmin_win2 = 0.0
tmax_win2 = 0.8
alpha = 0.05  # threshold FDR

# --- 0) obter trials das faces ---
faces_trials = epochs[faces_keys].get_data()  # (n_trials, n_channels, n_times)
times = epochs.times

# --- 1) índices temporais ---
idx1 = np.where((times >= tmin_win) & (times < tmax_win))[0]
idx2 = np.where((times >= tmin_win2) & (times <= tmax_win2))[0]

if len(idx1) == 0 or len(idx2) == 0:
    raise ValueError("Uma das janelas temporais não contém pontos de tempo — verifica limites ou sampling rate.")

# --- 2) média por trial e canal ---
win1_mean = faces_trials[:, :, idx1].mean(axis=2)
win2_mean = faces_trials[:, :, idx2].mean(axis=2)

# --- 3) t-test emparelhado por canal ---
tvals = np.zeros(win1_mean.shape[1])
pvals = np.ones(win1_mean.shape[1])
for ch in range(win1_mean.shape[1]):
    t, p = ttest_rel(win2_mean[:, ch], win1_mean[:, ch])
    tvals[ch] = t
    pvals[ch] = p

# --- 4) correção FDR ---
def bh_fdr(pvals, alpha=0.05):
    p = np.asarray(pvals)
    n = p.size
    order = np.argsort(p)
    sorted_p = p[order]
    thresh = (np.arange(1, n+1) * alpha) / n
    below = sorted_p <= thresh
    if not np.any(below):
        return np.zeros(n, dtype=bool)
    max_idx = np.max(np.where(below)[0])
    p_cutoff = sorted_p[max_idx]
    return p <= p_cutoff

significant_mask = bh_fdr(pvals, alpha=alpha)
sig_channels = [epochs.ch_names[i] for i, sig in enumerate(significant_mask) if sig]

print(f"Number of significant channels (FDR q={alpha}): {len(sig_channels)}")
print("Significant channel names:", sig_channels)

# --- 5) preparar objeto evoked EEG ---
evoked_faces = epochs[faces_keys].average()
evoked_eeg = evoked_faces.copy().pick_types(eeg=True)  # só EEG

# tvals e mask apenas para canais EEG
eeg_chs = evoked_eeg.ch_names
eeg_indices_global = [epochs.ch_names.index(ch) for ch in eeg_chs]

tvals_eeg = tvals[eeg_indices_global]
mask_eeg = significant_mask[eeg_indices_global]

# --- 6) plot topo ---
fig, ax = plt.subplots(figsize=(10, 6))

im, cm = plot_topomap(
    tvals_eeg,
    evoked_eeg.info,
    mask=mask_eeg,
    mask_params=dict(marker='o', markersize=8, markerfacecolor='w'),
    contours=0,
    axes=ax,
    show=False
)
im.set_clim(vmin=np.min(tvals_eeg), vmax=np.max(tvals_eeg))
plt.colorbar(im, ax=ax)

# layout e nomes dos canais
layout = mne.find_layout(evoked_eeg.info)
pos = layout.pos[:, :2]
pos = pos - np.mean(pos, axis=0)  # centralizar

scale_x = 0.10
scale_y = 0.135
pos[:, 0] = pos[:, 0] / np.max(np.abs(pos[:, 0])) * scale_x
pos[:, 1] = pos[:, 1] / np.max(np.abs(pos[:, 1])) * scale_y

for i, ch_name in enumerate(evoked_eeg.ch_names):
    x, y = pos[i]
    ax.text(x, y, ch_name, ha='center', va='center', fontsize=6)

plt.show()

#%%
import pandas as pd

# --- EEG ---
evoked_eeg = evoked_faces.copy().pick_types(eeg=True)
eeg_chs = evoked_eeg.ch_names
eeg_indices_global = [epochs.ch_names.index(ch) for ch in eeg_chs]
tvals_eeg = tvals[eeg_indices_global]
pvals_eeg = pvals[eeg_indices_global]
mask_eeg = significant_mask[eeg_indices_global]

df_eeg = pd.DataFrame({
    'channel': eeg_chs,
    'type': 'EEG',
    't_value': tvals_eeg,
    'p_value': pvals_eeg,
    'significant': mask_eeg
})

# --- MEG ---
# Escolhendo magnetómetros
evoked_mag = evoked_faces.copy().pick_types(meg='mag')
mag_chs = evoked_mag.ch_names
mag_indices_global = [epochs.ch_names.index(ch) for ch in mag_chs]
tvals_mag = tvals[mag_indices_global]
pvals_mag = pvals[mag_indices_global]
mask_mag = significant_mask[mag_indices_global]

df_meg = pd.DataFrame({
    'channel': mag_chs,
    'type': 'MEG',
    't_value': tvals_mag,
    'p_value': pvals_mag,
    'significant': mask_mag
})

# --- Juntar ambos ---
df_all = pd.concat([df_eeg, df_meg], ignore_index=True)
print(df_all)

# Se quiser, salvar em CSV
 df_all.to_csv("C:\\Users\\tomas\\Desktop\\MEG_outputs\\t_test_channels.csv", index=False)



#%%
# EEG significativos
sig_eeg_chs = df_eeg[df_eeg['significant']]['channel'].tolist()
evoked_sig_eeg = evoked_faces.copy().pick_channels(sig_eeg_chs)

# MEG significativos (magnetómetros)
sig_mag_chs = df_meg[df_meg['significant']]['channel'].tolist()
evoked_sig_mag = evoked_faces.copy().pick_channels(sig_mag_chs)

# EEG
evoked_sig_eeg.plot(
    titles='Evoked EEG - canais significativos',
    spatial_colors=True
)

# MEG
evoked_sig_mag.plot(
    titles='Evoked MEG - magnetómetros significativos',
    spatial_colors=True
)



#%% 
import matplotlib.pyplot as plt

# --- EEG ---
if len(sig_eeg_chs) > 0:
    evoked_sig_eeg = evoked_faces.copy().pick_channels(sig_eeg_chs)
    mean_eeg = evoked_sig_eeg.data.mean(axis=0)  # média ao longo dos canais
    times = evoked_sig_eeg.times

    plt.figure(figsize=(8,4))
    plt.plot(times, mean_eeg, color='blue', label='EEG - canais significativos')
    plt.axvline(0, color='k', linestyle='--')  # tempo zero (stimulus)
    plt.xlabel('Tempo (s)')
    plt.ylabel('Amplitude (uV)')
    plt.title('Resposta média EEG - canais significativos')
    plt.legend()
    plt.show()
else:
    print("Nenhum canal EEG significativo encontrado.")

# --- MEG (magnetómetros) ---
if len(sig_mag_chs) > 0:
    evoked_sig_mag = evoked_faces.copy().pick_channels(sig_mag_chs)
    mean_mag = evoked_sig_mag.data.mean(axis=0)
    times = evoked_sig_mag.times

    plt.figure(figsize=(8,4))
    plt.plot(times, mean_mag, color='red', label='MEG - canais significativos')
    plt.axvline(0, color='k', linestyle='--')
    plt.xlabel('Tempo (s)')
    plt.ylabel('Amplitude (fT)')
    plt.title('Resposta média MEG - canais significativos')
    plt.legend()
    plt.show()
else:
    print("Nenhum canal MEG significativo encontrado.")


#%%
bands = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 12),
    'beta':  (12, 30),
    'gamma': (30, 100)  # podes ajustar o limite superior se quiseres
}
# --- Exemplo para Alpha EEG ---
l_freq, h_freq = bands['alpha']
epochs_alpha = epochs.copy().filter(l_freq, h_freq, fir_design='firwin')

# Extrair trials
faces_trials_alpha = epochs_alpha[faces_keys].get_data()
times = epochs_alpha.times

# Janelas temporais
idx1 = np.where((times >= tmin_win) & (times < tmax_win))[0]
idx2 = np.where((times >= tmin_win2) & (times <= tmax_win2))[0]

win1_mean = faces_trials_alpha[:, :, idx1].mean(axis=2)
win2_mean = faces_trials_alpha[:, :, idx2].mean(axis=2)

# t-test sensor-wise
tvals_alpha = np.zeros(win1_mean.shape[1])
pvals_alpha = np.ones(win1_mean.shape[1])
for ch in range(win1_mean.shape[1]):
    t, p = ttest_rel(win2_mean[:, ch], win1_mean[:, ch])
    tvals_alpha[ch] = t
    pvals_alpha[ch] = p

# Filtrar apenas canais EEG do epochs_alpha
epochs_alpha_eeg = epochs_alpha.copy().pick_types(eeg=True)

# Selecionar apenas os canais significativos que existem no EEG
sig_chs_alpha_eeg = [ch for ch in sig_chs_alpha if ch in epochs_alpha_eeg.ch_names]

if sig_chs_alpha_eeg:
    # Criar evoked apenas com canais significativos EEG
    evoked_sig_alpha = epochs_alpha_eeg[faces_keys].average().copy().pick_channels(sig_chs_alpha_eeg)
    mean_alpha = evoked_sig_alpha.data.mean(axis=0)

    # Plot
    plt.figure(figsize=(8,4))
    plt.plot(evoked_sig_alpha.times, mean_alpha, color='blue', label='Alpha EEG - canais significativos')
    plt.axvline(0, color='k', linestyle='--')
    plt.xlabel('Tempo (s)')
    plt.ylabel('Amplitude (uV)')
    plt.title('Resposta média EEG - Alpha (8-12 Hz)')
    plt.legend()
    plt.show()
else:
    print("Nenhum canal EEG significativo na banda Alpha.")



#%%
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
import matplotlib.pyplot as plt
import mne

# --- Parâmetros de janelas ---
tmin_win = -0.2
tmax_win = 0.0
tmin_win2 = 0.0
tmax_win2 = 0.8
alpha_fdr = 0.05

# --- Bandas clássicas ---
bands = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 12),
    'beta':  (12, 30),
    'gamma': (30, 100)
}

# --- Função de FDR ---
def bh_fdr(pvals, alpha=0.05):
    p = np.asarray(pvals)
    n = p.size
    order = np.argsort(p)
    sorted_p = p[order]
    thresh = (np.arange(1, n+1) * alpha) / n
    below = sorted_p <= thresh
    if not np.any(below):
        return np.zeros(n, dtype=bool)
    max_idx = np.max(np.where(below)[0])
    p_cutoff = sorted_p[max_idx]
    return p <= p_cutoff

# --- Função genérica para análise por banda ---
def analyze_band(epochs, faces_keys, l_freq, h_freq, channel_type='eeg', band_name='band', tmin_win=-0.2, tmax_win=0.0, tmin_win2=0.0, tmax_win2=0.8, alpha_fdr=0.05):

    # 1) filtrar dados
    epochs_band = epochs.copy().filter(l_freq, h_freq, fir_design='firwin')
    epochs_band = epochs_band.pick_types(eeg=True) if channel_type.lower() == 'eeg' else epochs_band.pick_types(meg='mag')

    # 2) obter trials
    trials = epochs_band[faces_keys].get_data()
    times = epochs_band.times

    # 3) índices das janelas
    idx1 = np.where((times >= tmin_win) & (times < tmax_win))[0]
    idx2 = np.where((times >= tmin_win2) & (times <= tmax_win2))[0]

    if len(idx1) == 0 or len(idx2) == 0:
        raise ValueError("Janelas temporais sem pontos de tempo válidos.")

    # 4) média por janela
    win1_mean = trials[:, :, idx1].mean(axis=2)
    win2_mean = trials[:, :, idx2].mean(axis=2)

    # 5) t-test emparelhado sensor-wise
    tvals = np.zeros(win1_mean.shape[1])
    pvals = np.ones(win1_mean.shape[1])
    for ch in range(win1_mean.shape[1]):
        t, p = ttest_rel(win2_mean[:, ch], win1_mean[:, ch])
        tvals[ch] = t
        pvals[ch] = p

    # 6) FDR
    sig_mask = bh_fdr(pvals, alpha=alpha_fdr)
    sig_chs = [epochs_band.ch_names[i] for i, sig in enumerate(sig_mask) if sig]

    # 7) Criar evoked médio apenas para canais significativos
    if sig_chs:
        evoked_sig = epochs_band[faces_keys].average().copy().pick_channels(sig_chs)
        mean_data = evoked_sig.data.mean(axis=0)

        # Plot resposta média
        plt.figure(figsize=(8,4))
        plt.plot(times, mean_data, label=f'{band_name.upper()} {channel_type.upper()} - canais significativos')
        plt.axvline(0, color='k', linestyle='--')
        plt.xlabel('Tempo (s)')
        plt.ylabel('Amplitude (uV)' if channel_type.lower()=='eeg' else 'Amplitude (fT)')
        plt.title(f'Resposta média {channel_type.upper()} - {band_name.upper()}')
        plt.legend()
        plt.show()
    else:
        print(f"Nenhum canal {channel_type.upper()} significativo na banda {band_name.upper()}.")
        evoked_sig = None

    # 8) Guardar CSV
    df = pd.DataFrame({
        'channel': epochs_band.ch_names,
        'type': channel_type.upper(),
        't_value': tvals,
        'p_value': pvals,
        'significant': sig_mask
    })
    csv_filename = f"{channel_type}_{band_name}_t_test.csv"
    df.to_csv(csv_filename, index=False)
    print(f"CSV guardado: {csv_filename}")

    return evoked_sig, df

# --- Rodar todas as bandas para EEG e MEG ---
results = {}
for band_name, (l_freq, h_freq) in bands.items():
    print(f"\n=== Analisando banda {band_name.upper()} EEG ===")
    evoked_eeg, df_eeg = analyze_band(epochs, faces_keys, l_freq, h_freq, channel_type='eeg', band_name=band_name,
                                      tmin_win=tmin_win, tmax_win=tmax_win, tmin_win2=tmin_win2, tmax_win2=tmax_win2, alpha_fdr=alpha_fdr)
    results[f'EEG_{band_name}'] = (evoked_eeg, df_eeg)

    print(f"\n=== Analisando banda {band_name.upper()} MEG ===")
    evoked_meg, df_meg = analyze_band(epochs, faces_keys, l_freq, h_freq, channel_type='meg', band_name=band_name,
                                      tmin_win=tmin_win, tmax_win=tmax_win, tmin_win2=tmin_win2, tmax_win2=tmax_win2, alpha_fdr=alpha_fdr)
    results[f'MEG_{band_name}'] = (evoked_meg, df_meg)










#%%
faces_keys = [k for k in epochs.event_id.keys() if k.startswith("faces")]
faces_evoked = epochs[faces_keys].average()
faces_evoked.plot_topo()

faces_evoked.plot()

#%%
faces_evoked = evoked_dict["faces"] 
fonts_evoked = evoked_dict["fonts"] 

# --- 1) Calcular média de epochs para a categoria "faces" ---
faces_epochs = epochs["faces"]  # assume que já tens category_epochs
faces_evoked = faces_epochs.average()
# --- 2) Plotar gráfico conjunto (ERPs + topografia) ---
faces_evoked.plot_joint(picks="mag")
# --- 3) Plotar mapas topográficos em tempos específicos ---
# ajusta os tempos de acordo com o que queres visualizar
faces_evoked.plot_topomap(times=[0.0, 0.08, 0.1, 0.12, 0.2], ch_type="mag")



#%%
evoked_diff = mne.combine_evoked([faces, fonts], weights=[1, -1])
evoked_diff.pick(picks="mag").plot_topo(color="r", legend=False)



#%%

faces_evoked = evoked_dict["faces"] 
fonts_evoked = evoked_dict["fonts"] 

# --- 1) Calcular médias de duas categorias ---
faces_evoked = epochs["faces"].average()
fonts_evoked = epochs["fonts"].average()

# --- 2) Combinar as evoked (diferença faces - objects) ---
evoked_diff = mne.combine_evoked([faces_evoked, fonts_evoked], weights=[1, -1])
# --- 3) Selecionar apenas canais magnetômetros e plotar topografia ---
evoked_diff.pick(picks="mag").plot_topo(color="r", legend=False)
# %%
