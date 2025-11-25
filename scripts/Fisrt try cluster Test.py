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
ssvep_folder = mne.datasets.ssvep.data_path()
ssvep_data_raw_path = (
    ssvep_folder / "sub-02" / "ses-01" / "eeg" / "sub-02_ses-01_task-ssvep_eeg.vhdr"
)
ssvep_raw = mne.io.read_raw_brainvision(ssvep_data_raw_path, verbose=False)

# Use the preloaded montage
ssvep_raw.set_montage(easycap_montage)
fig = ssvep_raw.plot_sensors(show_names=True)

# Apply a template montage directly, without preloading
ssvep_raw.set_montage("easycap-M1")
fig = ssvep_raw.plot_sensors(show_names=True)



#%%
layout_from_raw = mne.channels.make_eeg_layout(raw.info)
# same result as mne.channels.find_layout(raw.info, ch_type='eeg')
layout_from_raw.plot()


#Another try 

#%%
epochs.compute_psd(fmin=2.0, fmax=40.0).plot(
    average=True, amplitude=False, picks="data", exclude="bads"
)


#%%

epochs.compute_psd().plot_topomap(ch_type="grad", normalize=False, contours=0)

#%% NO WORKING
_, ax = plt.subplots()
spectrum = epochs.compute_psd(fmin=0.1, fmax=330 , tmax=None, n_jobs=-0.1, method='multitaper')
# average across epochs first
mean_spectrum = spectrum.average()
psds, freqs = mean_spectrum.get_data(return_freqs=True)
# then convert to dB and take mean & standard deviation across channels
psds = 10 * np.log10(psds)
psds_mean = psds.mean(axis=0)
psds_std = psds.std(axis=0)

ax.plot(freqs, psds_mean, color="k")
ax.fill_between(
    freqs,
    psds_mean - psds_std,
    psds_mean + psds_std,
    color="k",
    alpha=0.5,
    edgecolor="none",
)
ax.set(
    title="Multitaper PSD (gradiometers)",
    xlabel="Frequency (Hz)",
    ylabel="Power Spectral Density (dB)",
)

#%%
# Estimate PSDs based on "mean" and "median" averaging for comparison.
kwargs = dict(fmin=0.1, fmax=330, n_jobs=-0.1)
psds_welch_mean, freqs_mean = epochs.compute_psd(
    "welch", average="mean", **kwargs
).get_data(return_freqs=True)
psds_welch_median, freqs_median = epochs.compute_psd(
    "welch", average="median", **kwargs
).get_data(return_freqs=True)

# Convert power to dB scale.
psds_welch_mean = 10 * np.log10(psds_welch_mean)
psds_welch_median = 10 * np.log10(psds_welch_median)

# We will only plot the PSD for a single sensor in the first epoch.
ch_name = "MEG 0122"
ch_idx = epochs.info["ch_names"].index(ch_name)
epo_idx = 0

_, ax = plt.subplots()
ax.plot(
    freqs_mean,
    psds_welch_mean[epo_idx, ch_idx, :],
    color="k",
    ls="-",
    label="mean of segments",
)
ax.plot(
    freqs_median,
    psds_welch_median[epo_idx, ch_idx, :],
    color="k",
    ls="--",
    label="median of segments",
)

ax.set(
    title=f"Welch PSD ({ch_name}, Epoch {epo_idx})",
    xlabel="Frequency (Hz)",
    ylabel="Power Spectral Density (dB)",
)
ax.legend(loc="upper right")



#%%

freqs = np.logspace(*np.log10([6, 35]), num=8)
n_cycles = freqs / 2.0  # different number of cycle per frequency
power, itc = epochs.compute_tfr(
    method="morlet",
    freqs=freqs,
    n_cycles=n_cycles,
    average=True,
    return_itc=True,
    decim=3,
)

#%%

power.plot_topo(baseline=(-0.5, 0), mode="logratio", title="Average power")
power.plot(picks=[82], baseline=(-0.5, 0), mode="logratio", title=power.ch_names[82])

fig, axes = plt.subplots(1, 2, figsize=(7, 4), layout="constrained")
topomap_kw = dict(
    ch_type="grad", tmin=0.5, tmax=1.0, baseline=(-0.5, 0), mode="logratio", show=False
)
plot_dict = dict(Alpha=dict(fmin=8, fmax=12), Beta=dict(fmin=13, fmax=25))
for ax, (title, fmin_fmax) in zip(axes, plot_dict.items()):
    power.plot_topomap(**fmin_fmax, axes=ax, **topomap_kw)
    ax.set_title(title)

#%%
itc.plot_topo(title="Inter-Trial coherence", vmin=0.0, vmax=1.0, cmap="Reds")





#%%

